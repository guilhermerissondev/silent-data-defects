"""Camada 1 — agregados vetorizados em DuckDB.

O desenho inicial era "uma query com tudo, uma passada so". A medicao derrubou isso.
Ver `bench/batching.py` para os numeros; o resumo, sobre 5 M x 26 colunas:

    400 count(*) FILTER numa unica query .............. 16,7 s
    400 count(*) FILTER em lotes de 16 (25 queries) ....  1,8 s   9x mais rapido
    24 approx_top_k numa unica query .................. 11,7 s
    24 approx_top_k em lotes de 2 (12 queries) ........  1,1 s   10x mais rapido
    104 agregados simples em 1 query ..................  1,1 s
    104 agregados simples em lotes de 16 ..............  2,4 s   2x mais LENTO

Ou seja: "numero de passadas" e a metrica errada. Agregado barato e limitado por
I/O e deve ser empacotado ao maximo, porque varios dividem a mesma varredura;
agregado pesado e com estado (Space-Saving, t-digest, cadeia de FILTER) degrada
de forma super-linear quando empilhado na mesma query. O lote otimo e **por classe
de custo da expressao**, nao por coluna.

Ressalva honesta: com a base cabendo em cache de pagina, revarrer e quase de graca
e o custo de expressao domina. Em base maior que a RAM o balanco muda e os lotes
das classes pesadas devem crescer. `--batch-profile io` existe para isso.
"""

from __future__ import annotations

import datetime
import decimal
import time
from dataclasses import dataclass
from typing import Any, Iterable

import duckdb

from ..types import ColumnStats, ScanResult

DEFAULT_QUANTILES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
TOP_K = 24            # itens pedidos ao Space-Saving em coluna larga
CANDIDATES = 8        # quantos deles vao para a contagem exata
NARROW_MAX = 4096     # ate esta cardinalidade, histogram() exato sai mais barato

# Teto de valores distintos que aceitamos materializar num count(DISTINCT) exato.
# Acima disso a tabela hash e que vira o problema de memoria, e fica o HLL.
NDV_BUDGET = 5_000_000
HLL_SLACK = 1.25      # folga sobre a estimativa do HLL: o erro medido chega a 17%

# Lote por classe de custo. Calibrado em bench/batching.py.
BATCH_CPU = {"cheap": 512, "quantile": 4, "topk": 2, "histogram": 64,
             "filter": 16, "distinct": 4}
# Perfil para base maior que a RAM: revarrer passa a doer, entao empacota mais.
BATCH_IO = {"cheap": 512, "quantile": 16, "topk": 8, "histogram": 64,
            "filter": 64, "distinct": 8}

_NUMERIC = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
            "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE",
            "REAL", "DECIMAL", "NUMERIC")
_TEMPORAL = ("DATE", "TIMESTAMP", "TIME", "INTERVAL")


def classify(sql_type: str) -> str:
    t = sql_type.upper()
    if t.startswith("BOOLEAN"):
        return "boolean"
    if t.startswith(_NUMERIC):
        return "numeric"
    if t.startswith(_TEMPORAL):
        return "temporal"
    if t.startswith(("VARCHAR", "CHAR", "TEXT", "STRING", "UUID")):
        return "string"
    return "other"


def q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _py(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


@dataclass(frozen=True)
class Expr:
    alias: str
    sql: str
    klass: str


class Runner:
    """Executa expressoes agrupadas por classe de custo, com o lote de cada classe."""

    def __init__(self, con, src: str, batch: dict[str, int]):
        self.con, self.src, self.batch = con, src, batch
        self.queries = 0
        self.by_class: dict[str, float] = {}

    def run(self, exprs: Iterable[Expr]) -> dict[str, Any]:
        buckets: dict[str, list[Expr]] = {}
        for e in exprs:
            buckets.setdefault(e.klass, []).append(e)
        out: dict[str, Any] = {}
        for klass, group in buckets.items():
            size = self.batch.get(klass, 32)
            t0 = time.perf_counter()
            for s in range(0, len(group), size):
                chunk = group[s:s + size]
                sel = ", ".join(f"{e.sql} AS {e.alias}" for e in chunk)
                row = self.con.execute(f"SELECT {sel} FROM {self.src}").fetchone()
                out.update(zip((e.alias for e in chunk), row))
                self.queries += 1
            self.by_class[klass] = self.by_class.get(klass, 0.0) + time.perf_counter() - t0
        return out


def connect(threads: int | None = None, memory_limit: str | None = None):
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def source_expr(paths: list[str], limit: int | None = None) -> str:
    lit = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    if all(p.lower().endswith((".parquet", ".pq")) for p in paths):
        src = f"read_parquet([{lit}], union_by_name = true)"
    else:
        src = f"read_csv([{lit}], union_by_name = true, sample_size = -1)"
    return f"(SELECT * FROM {src} LIMIT {int(limit)})" if limit else src


# ---------------------------------------------------------------------------
# Estagio A — agregados baratos, todos numa query so
# ---------------------------------------------------------------------------

def _stage_a(idx: int, name: str, kind: str) -> list[Expr]:
    c = q(name)
    out = [Expr(f"c{idx}_nn", f"count({c})", "cheap")]
    if kind == "other":
        return out
    out += [
        Expr(f"c{idx}_ndv", f"approx_count_distinct({c})", "cheap"),
        Expr(f"c{idx}_min", f"min({c})", "cheap"),
        Expr(f"c{idx}_max", f"max({c})", "cheap"),
    ]
    if kind == "numeric":
        out += [Expr(f"c{idx}_avg", f"avg({c})", "cheap"),
                Expr(f"c{idx}_std", f"stddev_pop({c})", "cheap")]
    elif kind == "string":
        out.append(Expr(f"c{idx}_len", f"avg(length({c}))", "cheap"))
    elif kind == "boolean":
        out.append(Expr(f"c{idx}_avg", f"avg(CASE WHEN {c} THEN 1.0 ELSE 0.0 END)", "cheap"))
    return out


# ---------------------------------------------------------------------------
# Estagio B — quantis, e contagem de valores por rota conforme a cardinalidade
# ---------------------------------------------------------------------------

def _stage_b(idx: int, cs: ColumnStats, quantiles: list[float]) -> list[Expr]:
    c = q(cs.name)
    out: list[Expr] = []
    if cs.kind == "numeric":
        qs = ", ".join(str(x) for x in quantiles)
        out.append(Expr(f"c{idx}_q", f"approx_quantile({c}, [{qs}])", "quantile"))
    if cs.kind in ("numeric", "string", "temporal", "boolean"):
        ndv = cs.ndv_approx or 0
        if 0 < ndv <= NARROW_MAX:
            # Coluna estreita: histogram() e exato, completo e mais barato que
            # approx_top_k + contagem. Dispensa o estagio C inteiro.
            out.append(Expr(f"c{idx}_hist", f"histogram({c})", "histogram"))
        elif cs.kind != "boolean":
            out.append(Expr(f"c{idx}_top", f"approx_top_k({c}, {TOP_K})", "topk"))
    return out


# ---------------------------------------------------------------------------
# Estagio C — contagem exata dos candidatos, so das colunas largas
# ---------------------------------------------------------------------------

def _literal(v: Any, kind: str) -> str | None:
    if v is None:
        return None
    if kind == "numeric":
        if isinstance(v, bool):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN nunca casa com '='
            return None
        return repr(int(v)) if isinstance(v, int) else repr(f)
    if kind == "string":
        return "'" + str(v).replace("'", "''") + "'"
    if kind == "temporal":
        if isinstance(v, datetime.datetime):
            return f"TIMESTAMP '{v.isoformat(sep=' ')}'"
        if isinstance(v, datetime.date):
            return f"DATE '{v.isoformat()}'"
    return None


def _candidates(cs: ColumnStats) -> list[Any]:
    """Sem filtro semantico: os primeiros do Space-Saving, mais os extremos.

    Nao ha lista de constantes canonicas aqui de proposito. Sentinela de verdade
    tem massa alta e portanto aparece no topo; singleton de coluna continua entra
    junto e e descartado pela massa na contagem exata. Filtrar por `-999`, `9999`
    e afins aqui seria regex disfarcada de detector.
    """
    cand = list(cs.top_values[:CANDIDATES])
    for extreme in (cs.min, cs.max):
        if extreme is not None and extreme not in cand:
            cand.append(extreme)
    seen, out = set(), []
    for v in cand:
        k = str(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


# ---------------------------------------------------------------------------

def scan(paths: list[str], *, threads: int | None = None, memory_limit: str | None = None,
         quantiles: list[float] = DEFAULT_QUANTILES, value_counts: bool = True,
         batch_profile: str = "cpu", limit: int | None = None,
         exact_ndv: bool = True, ndv_budget: int = NDV_BUDGET,
         materialize: bool | None = None) -> ScanResult:
    t0 = time.perf_counter()
    con = connect(threads, memory_limit)
    src = source_expr(paths, limit)

    # O plano de varias queries so se paga onde revarrer e barato. Parquet e colunar,
    # tem projection pushdown e fica em cache de pagina: revarrer custa quase nada.
    # CSV nao — cada query reparseia o arquivo inteiro. Medido em 200 mil linhas:
    # uma varredura do CSV 0,71 s, e o perfil inteiro (23 queries) 43,6 s; materializar
    # em tabela temporaria custa 0,79 s uma vez e derruba a varredura para ~0 s.
    if materialize is None:
        materialize = not all(p.lower().endswith((".parquet", ".pq")) for p in paths)
    if materialize:
        con.execute(f"CREATE TEMP TABLE _sdd_src AS SELECT * FROM {src}")
        src = "_sdd_src"
    runner = Runner(con, src, BATCH_IO if batch_profile == "io" else BATCH_CPU)

    desc = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    cols = [(r[0], r[1]) for r in desc]
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    runner.queries += 2

    # --- A ---
    got = runner.run(e for i, (name, ty) in enumerate(cols)
                     for e in _stage_a(i, name, classify(ty)))
    stats: dict[str, ColumnStats] = {}
    for i, (name, ty) in enumerate(cols):
        kind = classify(ty)
        cs = ColumnStats(name=name, kind=kind, sql_type=ty, n=n,
                         non_null=got.get(f"c{i}_nn") or 0)
        cs.ndv_approx = got.get(f"c{i}_ndv")
        cs.min = _py(got.get(f"c{i}_min"))
        cs.max = _py(got.get(f"c{i}_max"))
        if kind in ("numeric", "boolean"):
            cs.mean = _py(got.get(f"c{i}_avg"))
            cs.stddev = _py(got.get(f"c{i}_std"))
        elif kind == "string":
            cs.avg_len = _py(got.get(f"c{i}_len"))
        stats[name] = cs

    if value_counts:
        # --- B ---
        got = runner.run(e for i, (name, _) in enumerate(cols)
                         for e in _stage_b(i, stats[name], quantiles))
        for i, (name, _) in enumerate(cols):
            cs = stats[name]
            qv = got.get(f"c{i}_q")
            if qv:
                cs.quantiles = {f"p{x:g}": _py(v) for x, v in zip(quantiles, qv)}
            hist = got.get(f"c{i}_hist")
            if hist:
                cs.value_counts = {str(_py(k)): int(v) for k, v in hist.items()}
                cs.counts_source = "histogram"
                # o histogram ja enumerou todos os valores: a cardinalidade sai exata
                cs.ndv_approx = len(hist)
                cs.ndv_source = "exact_histogram"
            top = got.get(f"c{i}_top")
            if top:
                cs.top_values = [_py(v) for v in top]

        # --- B2: cardinalidade exata onde a tabela hash cabe no orcamento ---
        # O HLL do DuckDB nao tem parametro de precisao e erra ate 17% nesta base
        # (bench/accuracy.py). Como count(DISTINCT) custa ~1,3 s para 26 colunas em
        # 5 M linhas, o exato vale a pena sempre que couber; o HLL fica para a coluna
        # larga demais, onde o limite passa a ser memoria e nao tempo.
        if exact_ndv:
            pend = [(i, name) for i, (name, _) in enumerate(cols)
                    if stats[name].ndv_source == "hll"
                    and stats[name].kind != "other"
                    and (stats[name].ndv_approx or 0) * HLL_SLACK <= ndv_budget]
            if pend:
                got = runner.run(Expr(f"c{i}_dist", f"count(DISTINCT {q(name)})", "distinct")
                                 for i, name in pend)
                for i, name in pend:
                    v = got.get(f"c{i}_dist")
                    if v is not None:
                        stats[name].ndv_approx = int(v)
                        stats[name].ndv_source = "exact_distinct"

        # --- C ---
        exprs, back = [], []
        for name, cs in stats.items():
            if cs.counts_source == "histogram" or not cs.top_values:
                continue
            for v in _candidates(cs):
                lit = _literal(v, cs.kind)
                if lit is None:
                    continue
                exprs.append(Expr(f"k{len(back)}",
                                  f"count(*) FILTER (WHERE {q(name)} = {lit})", "filter"))
                back.append((name, v))
        if exprs:
            got = runner.run(exprs)
            for j, (name, v) in enumerate(back):
                cnt = got.get(f"k{j}")
                if cnt:
                    stats[name].value_counts[str(v)] = int(cnt)
                stats[name].counts_source = "top_k"

    res = ScanResult(
        source=paths[0] if len(paths) == 1 else f"{len(paths)} arquivos",
        rows=n, columns=stats,
        elapsed_s=time.perf_counter() - t0,
        queries=runner.queries,
        engine=f"duckdb {duckdb.__version__}",
        timing_by_class={k: round(v, 3) for k, v in runner.by_class.items()},
    )
    con.close()
    return res
