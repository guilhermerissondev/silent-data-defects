"""D1/D2 — sentinela disfarcada de valor.

`-999`, `9999`, `0`, `1900-01-01`, `"N/A"`. Passam por qualquer validador de tipo e
envenenam media, escala e corte de arvore.

O detector e o **teste de gap**, nao a lista de constantes. Sentinela e um ponto
*isolado* no suporte: renda com `0` e depois nada ate `800` e codigo de ausente; renda
com `0, 12, 40, 95, ...` e valor legitimo. Mesmo valor, veredito oposto — e a distancia
ate o vizinho, medida em unidades da dispersao da propria coluna, que separa os dois.

    gap_ratio = distancia(v, vizinho mais proximo) / escala

**A escala nao pode ser o IQR.** Medido nesta base (5 M linhas):

    coluna                  v       gap/IQR   gap/amplitude   lift    verdade
    renda                -999,0       0,59        0,011      33.657   sentinela
    idade                    0        0,46        0,234           1   sentinela
    dias_desde_ultimo_pgto 9999      220,2      111,3             1   sentinela
    qtd_produtos             0        1,00        0,091           3   LEGITIMO
    valor_recuperado       0,0        0,00        0,000     251.597   zero estrutural

Com IQR, a armadilha `qtd_produtos = 0` (1,00) pontua **mais alto** que a sentinela
`idade = 0` (0,46): o teste inverte o veredito. E `gap/amplitude` tambem nao serve
sozinha — `renda` (0,011) fica abaixo da armadilha (0,091).

O que separa e usar **escalas diferentes conforme a coluna seja discreta ou continua**,
que e exatamente a divisao de rotas que a camada 1 ja produziu:

* **coluna estreita** (`histogram()` completo, discreta) — escala = **espacamento
  mediano entre valores distintos adjacentes**. Mede buraco no suporte, que e o que
  "ponto isolado" significa. Separa limpo: idade 18x e dias 9.910x contra a armadilha
  em 1x. Massa aqui e inutil como criterio — `idade = 0` tem 1,2% contra 1,27% dos
  valores legitimos, ou seja, **ranquear candidato por massa perderia a sentinela**.
  Tudo sai da tabela de frequencias em Python, sem uma query sequer.
* **coluna larga** (top-k do Space-Saving, continua) — escala = **amplitude do suporte
  restante**, combinada com o *lift* (massa x ndv), que detecta massa pontual numa
  distribuicao continua. Separa `renda` (gap/amplitude 0,011) de `valor_recuperado`
  (0,0000001) por cinco ordens de grandeza. Vizinhos e quantis vem de uma query por
  coluna, com `quantile_cont` exato: o t-digest nao e deterministico com threads > 1
  e erra justamente ao lado de massa pontual. Ver `sdd/scan/exact.py`.

Massa pontual em coluna continua **sem** isolamento nao e sentinela: e inflacao de zero
(`valor_recuperado` e 72% zeros porque so inadimplente tem recuperacao). Sai como achado
proprio, `massa_pontual_em_continua`, porque quebra premissa de modelo do mesmo jeito —
mas o veredito de sentinela exige o gap, e distinguir os dois casos exige semantica que
a ferramenta nao tem.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from ..scan.duckdb_pass import q
from ..types import ColumnStats
from .base import Finding, is_canonical

# Rota discreta: gap medido em espacamentos medianos entre valores adjacentes.
GAP_ESPACAMENTO = 8.0
# Rota continua: gap medido em fracao da amplitude do suporte restante.
GAP_AMPLITUDE = 0.005
# Rota continua: massa x ndv. Sob continuidade nenhum valor repete muito, entao um
# lift alto significa massa pontual — condicao necessaria, nao suficiente.
LIFT_MIN = 50.0

MIN_MASS = 0.001       # abaixo disto nao move nada a jusante
MAX_CANDIDATES = 8     # so na rota de coluna larga, onde massa e criterio valido

# Codigos textuais de ausencia. Isto NAO e um detector: e uma tabela de consulta,
# e o relatorio precisa dizer isso. Fica em D2, separado do D1, por honestidade.
TEXT_NULLS = {"", " ", "-", "--", "?", "n/a", "na", "n.a.", "null", "none", "nil",
              "nan", "#n/a", "#null!", "undefined", "unknown", "desconhecido",
              "nao informado", "não informado", "sem informacao", "sem informação",
              "vazio", "9999", "99999", "-999"}
DATE_NULLS = {"1900-01-01", "1901-01-01", "1970-01-01", "0001-01-01",
              "9999-12-31", "2999-12-31", "1899-12-30"}


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _quantiles_excluding(pairs: list[tuple[float, int]], drop: float,
                         probs=(0.01, 0.25, 0.5, 0.75, 0.99)) -> list[float]:
    """Quantis exatos a partir da tabela de frequencias, ignorando um valor.

    Excluir o candidato importa: a sentinela nao pode distorcer a escala com que ela
    mesma sera julgada. Como a tabela e completa, isto e exato e custa nada.
    """
    tot = sum(c for v, c in pairs if v != drop)
    if tot <= 0:
        return []
    out, acc, i = [], 0, 0
    alvos = [p * tot for p in probs]
    vivos = [(v, c) for v, c in pairs if v != drop]
    for alvo in alvos:
        while i < len(vivos) and acc + vivos[i][1] < alvo:
            acc += vivos[i][1]
            i += 1
        out.append(vivos[min(i, len(vivos) - 1)][0])
    return out


def _finding(cs: ColumnStats, v: float, massa: float, d: float, escala: float,
             viz: Any, rota: str, limiar: float, extra: dict,
             diag: list | None = None) -> Finding | None:
    if not escala:
        return None
    ratio = d / escala
    if diag is not None:
        # Todo candidato avaliado entra aqui, aceito ou nao. Sem os rejeitados nao da
        # para medir a margem entre o positivo mais fraco e o negativo mais forte, que
        # e o numero que diz se a precisao foi detector ou sorte.
        diag.append({"coluna": cs.name, "valor": v, "massa": massa, "rota": rota,
                     "razao": ratio, "limiar": limiar, "aceito": ratio >= limiar})
    if ratio < limiar:
        return None  # ponto colado no suporte: valor legitimo, nao sentinela

    # o gap decide; o resto so reforca a confianca
    conf = 0.40 + 0.45 * min(1.0, ratio / (2 * limiar))
    reforcos: list[str] = []
    if is_canonical(v):
        conf += 0.12
        reforcos.append("constante_canonica")
    if cs.min is not None and v == _num(cs.min):
        conf += 0.06
        reforcos.append("no_minimo")
    if cs.max is not None and v == _num(cs.max):
        conf += 0.06
        reforcos.append("no_maximo")

    if rota == "discreta":
        dist = f"{ratio:,.0f}x o espacamento tipico entre valores ({escala:g})"
    else:
        dist = f"{ratio:.2%} da amplitude do suporte ({escala:,.0f})"
    return Finding(
        coluna=cs.name, detector="D1", tipo="sentinela_numerica",
        valor=v, massa=massa, confianca=round(min(0.97, conf), 3),
        evidencia={"rota": rota, "gap_ratio": round(ratio, 6), "limiar": limiar,
                   "vizinho_mais_proximo": viz, "distancia": d, "escala": escala,
                   "reforcos": reforcos, **extra},
        nota=(f"ponto isolado: vizinho mais proximo {viz!r}, a {dist}; "
              f"tratar como ausente afeta {massa:.2%} das linhas"),
    )


# ---------------------------------------------------------------------------
# Rota A — coluna estreita: tudo sai do histograma, sem query
# ---------------------------------------------------------------------------

def _detect_from_histogram(cs: ColumnStats, min_mass: float,
                           limiar: float = GAP_ESPACAMENTO,
                           diag: list | None = None) -> list[Finding]:
    pairs = []
    for v, c in cs.value_counts.items():
        f = _num(v)
        if f is not None:
            pairs.append((f, c))
    if len(pairs) < 4:
        return []
    pairs.sort()
    vals = [v for v, _ in pairs]
    tot = sum(c for _, c in pairs)

    out = []
    for idx, (v, c) in enumerate(pairs):
        massa = c / tot
        if massa < min_mass:
            continue
        viz_l = vals[idx - 1] if idx > 0 else None
        viz_r = vals[idx + 1] if idx + 1 < len(vals) else None
        dists = [(abs(v - x), x) for x in (viz_l, viz_r) if x is not None]
        if not dists:
            continue
        d, viz = min(dists)

        # escala = espacamento tipico entre valores adjacentes, ja sem o candidato.
        # Mede buraco no suporte; e o unico jeito de distinguir `idade = 0` (gap 18
        # onde o tipico e 1) de `qtd_produtos = 0` (gap 1 onde o tipico e 1).
        resto = [x for x in vals if x != v]
        if len(resto) < 3:
            continue
        espacos = [b - a for a, b in zip(resto, resto[1:])]
        escala = median(espacos)
        qs = _quantiles_excluding(pairs, v)
        f = _finding(cs, v, massa, d, escala, viz, "discreta", limiar,
                     {"espacamento_mediano": escala, "ndv": len(vals),
                      "quantis_sem_candidato": dict(zip(("p01", "p25", "p50", "p75", "p99"), qs))},
                     diag)
        if f:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Rota B — coluna larga: uma query por coluna
# ---------------------------------------------------------------------------

def _detect_from_topk(con, src: str, cs: ColumnStats, min_mass: float,
                      limiar: float = GAP_AMPLITUDE,
                      lift_min: float = LIFT_MIN,
                      diag: list | None = None) -> list[Finding]:
    cands = []
    for v, c in cs.value_counts.items():
        f, massa = _num(v), c / cs.non_null
        if f is not None and massa >= min_mass:
            cands.append((f, massa))
    cands.sort(key=lambda t: -t[1])
    cands = cands[:MAX_CANDIDATES]
    if not cands:
        return []

    c = q(cs.name)
    excl = ", ".join(repr(v) for v, _ in cands)
    sel = [f"min({c}) FILTER (WHERE {c} NOT IN ({excl})) AS lo_g",
           f"max({c}) FILTER (WHERE {c} NOT IN ({excl})) AS hi_g"]
    for i, (v, _) in enumerate(cands):
        sel.append(f"max({c}) FILTER (WHERE {c} < {v!r} AND {c} NOT IN ({excl})) AS lo{i}")
        sel.append(f"min({c}) FILTER (WHERE {c} > {v!r} AND {c} NOT IN ({excl})) AS hi{i}")
    row = con.execute(f"SELECT {', '.join(sel)} FROM {src}").fetchone()
    if row[0] is None or row[1] is None:
        return []
    # amplitude do suporte legitimo: a escala natural de "longe" numa coluna continua
    escala = float(row[1]) - float(row[0])

    out = []
    for i, (v, massa) in enumerate(cands):
        lift = massa * (cs.ndv_approx or 0)
        if lift < lift_min:
            continue  # sem massa pontual nao ha o que julgar numa coluna continua
        lo, hi = row[2 + 2 * i], row[3 + 2 * i]
        dists = [(abs(v - float(x)), float(x)) for x in (lo, hi) if x is not None]
        if not dists:
            continue
        d, viz = min(dists)
        extra = {"lift": round(lift, 1), "amplitude_suporte": escala,
                 "ndv": cs.ndv_approx, "ndv_exato": cs.ndv_exact}
        f = _finding(cs, v, massa, d, escala, viz, "continua", limiar, extra, diag)
        if f:
            out.append(f)
            continue
        # Massa pontual sem isolamento nao e sentinela — e distribuicao mista
        # (inflacao de zero e o caso tipico). Quebra premissa de modelo do mesmo
        # jeito, mas o veredito de sentinela exige o gap.
        out.append(Finding(
            coluna=cs.name, detector="D1", tipo="massa_pontual_em_continua",
            valor=v, massa=massa, confianca=round(min(0.90, 0.45 + 0.10 * (lift ** 0.25)), 3),
            evidencia={**extra, "rota": "continua", "gap_ratio": round(d / escala, 8),
                       "limiar": limiar, "vizinho_mais_proximo": viz},
            nota=(f"massa pontual de {massa:.2%} em coluna continua (lift {lift:,.0f}x), "
                  f"mas colada no suporte: distribuicao mista, nao sentinela"),
        ))
    return out


# ---------------------------------------------------------------------------

def detect_text_and_date(cs: ColumnStats, min_mass: float = MIN_MASS) -> list[Finding]:
    """D2 — consulta a uma tabela de codigos conhecidos. Barato, mas nao e um detector:
    so encontra o que ja esta na lista, e a lista e cultural."""
    out = []
    if not cs.non_null:
        return out
    for v, c in cs.value_counts.items():
        massa = c / cs.non_null
        if massa < min_mass:
            continue
        s = str(v).strip()
        if cs.kind == "string" and s.lower() in TEXT_NULLS:
            out.append(Finding(coluna=cs.name, detector="D2", tipo="sentinela_textual",
                               valor=v, massa=massa, confianca=0.90,
                               evidencia={"metodo": "tabela_de_consulta"},
                               nota=f"codigo textual de ausencia em {massa:.2%} das linhas"))
        elif cs.kind == "temporal" and s[:10] in DATE_NULLS:
            extremo = s[:10] in (str(cs.min)[:10], str(cs.max)[:10])
            out.append(Finding(coluna=cs.name, detector="D2", tipo="sentinela_temporal",
                               valor=v, massa=massa, confianca=0.90 if extremo else 0.75,
                               evidencia={"metodo": "tabela_de_consulta",
                                          "no_extremo": extremo},
                               nota=f"data sentinela em {massa:.2%} das linhas"))
    return out


def detect(con, src: str, columns: dict[str, ColumnStats],
           min_mass: float = MIN_MASS, diag: list | None = None) -> list[Finding]:
    out: list[Finding] = []
    for cs in columns.values():
        if cs.kind == "numeric":
            if cs.counts_complete:
                out += _detect_from_histogram(cs, min_mass, diag=diag)
            elif cs.value_counts:
                out += _detect_from_topk(con, src, cs, min_mass, diag=diag)
        elif cs.kind in ("string", "temporal"):
            out += detect_text_and_date(cs, min_mass)
    return out
