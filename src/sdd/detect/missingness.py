"""D3 — nulo com estrutura.

"A coluna tem 12% de nulos" nao diz nada. O nulo raramente e aleatorio, e a estrutura
dele e que carrega a informacao:

* **por tempo** — `score_bureau` com 2% de nulos ate 2025-03 e 100% depois nao e uma
  coluna com 39% de nulos: e um pipeline que quebrou numa data. E o achado de maior
  valor pratico do projeto e o mais barato de calcular: um `GROUP BY` na coluna de data.
* **por categoria** — nulo concentrado num nivel (`valor_fatura` ausente so em
  `produto = consignado`) aponta a origem que falhou, nao uma falha difusa.
* **entre colunas** — nulos que coincidem vem do mesmo join que falhou.
* **contra o alvo** — se a taxa de evento difere entre nulo e nao-nulo, o nulo *e*
  sinal, e imputar apaga informacao. Aqui o relatorio diz "imputar apaga sinal", nao
  "12% de nulos".
"""

from __future__ import annotations

from typing import Any

from ..scan.duckdb_pass import q
from ..types import ColumnStats
from .base import Finding

MIN_SALTO = 0.30       # diferenca de taxa de nulo que caracteriza degrau
MIN_LADO = 0.05        # cada lado do corte precisa de pelo menos 5% das linhas
MIN_PHI = 0.50         # correlacao entre mascaras de nulo
MIN_DELTA_ALVO = 0.05  # diferenca absoluta na taxa de evento
MAX_NIVEIS = 512       # cardinalidade maxima de uma categorica usada como recorte


def _null_cols(columns: dict[str, ColumnStats]) -> list[ColumnStats]:
    return [c for c in columns.values() if 0 < c.null_rate < 1]


def pick_time_column(columns: dict[str, ColumnStats]) -> str | None:
    """Data de referencia = a coluna temporal de menor cardinalidade.

    Snapshot mensal tem dezenas de valores; data de nascimento tem dezenas de milhares.
    A heuristica erra em base sem safra — por isso `--time-column` existe.
    """
    temporais = [c for c in columns.values() if c.kind == "temporal" and (c.ndv_approx or 0) > 1]
    if not temporais:
        return None
    return min(temporais, key=lambda c: c.ndv_approx or 10**9).name


def por_tempo(con, src: str, columns: dict[str, ColumnStats], time_col: str,
              min_salto: float = MIN_SALTO) -> list[Finding]:
    alvos = _null_cols(columns)
    if not alvos:
        return []
    sel = ", ".join([f"{q(time_col)} AS t", "count(*) AS n"]
                    + [f"count({q(c.name)}) AS c{i}" for i, c in enumerate(alvos)])
    rows = con.execute(f"SELECT {sel} FROM {src} GROUP BY 1 ORDER BY 1").fetchall()
    if len(rows) < 3:
        return []

    total = sum(r[1] for r in rows)
    out = []
    for i, cs in enumerate(alvos):
        taxas = [(r[0], r[1], 1 - (r[2 + i] / r[1] if r[1] else 0)) for r in rows]
        # melhor ponto de corte: o que maximiza a diferenca de taxa entre os dois lados
        melhor = None
        acum_n = 0
        acum_nulo = 0.0
        tot_nulo = sum(n * t for _, n, t in taxas)
        for k in range(len(taxas) - 1):
            _, n, t = taxas[k]
            acum_n += n
            acum_nulo += n * t
            resto_n = total - acum_n
            if acum_n < total * MIN_LADO or resto_n < total * MIN_LADO:
                continue
            antes = acum_nulo / acum_n
            depois = (tot_nulo - acum_nulo) / resto_n
            d = abs(depois - antes)
            if melhor is None or d > melhor[0]:
                melhor = (d, taxas[k + 1][0], antes, depois)
        if melhor is None or melhor[0] < min_salto:
            continue
        d, corte, antes, depois = melhor
        out.append(Finding(
            coluna=cs.name, detector="D3", tipo="nulo_por_tempo",
            massa=cs.null_rate, confianca=round(min(0.97, 0.55 + 0.45 * d), 3),
            evidencia={"coluna_tempo": time_col, "corte": corte,
                       "taxa_antes": round(antes, 4), "taxa_depois": round(depois, 4),
                       "salto": round(d, 4), "buckets": len(rows)},
            nota=(f"taxa de nulo salta de {antes:.1%} para {depois:.1%} em {corte}; "
                  f"nao e uma coluna com {cs.null_rate:.0%} de nulos, e uma quebra de origem"),
        ))
    return out


def por_categoria(con, src: str, columns: dict[str, ColumnStats],
                  min_salto: float = MIN_SALTO) -> list[Finding]:
    alvos = _null_cols(columns)
    cats = [c for c in columns.values()
            if c.kind in ("string", "boolean") and 1 < (c.ndv_approx or 0) <= MAX_NIVEIS]
    if not alvos or not cats:
        return []

    out = []
    for cat in cats:
        sel = ", ".join([f"{q(cat.name)} AS g", "count(*) AS n"]
                        + [f"count({q(c.name)}) AS c{i}" for i, c in enumerate(alvos)])
        rows = con.execute(f"SELECT {sel} FROM {src} GROUP BY 1").fetchall()
        total = sum(r[1] for r in rows)
        for i, cs in enumerate(alvos):
            niveis = [(r[0], r[1], 1 - (r[2 + i] / r[1] if r[1] else 0))
                      for r in rows if r[1] >= total * 0.01]
            if len(niveis) < 2:
                continue
            alto = max(niveis, key=lambda t: t[2])
            baixo = min(niveis, key=lambda t: t[2])
            d = alto[2] - baixo[2]
            if d < min_salto:
                continue
            out.append(Finding(
                coluna=cs.name, detector="D3", tipo="nulo_por_categoria",
                massa=cs.null_rate, confianca=round(min(0.95, 0.50 + 0.45 * d), 3),
                evidencia={"coluna_recorte": cat.name, "nivel_alto": alto[0],
                           "taxa_alto": round(alto[2], 4), "nivel_baixo": baixo[0],
                           "taxa_baixo": round(baixo[2], 4), "spread": round(d, 4)},
                nota=(f"nulo concentrado em {cat.name} = {alto[0]!r} ({alto[2]:.1%}) "
                      f"contra {baixo[2]:.1%} em {baixo[0]!r}"),
            ))
    return out


def entre_colunas(con, src: str, columns: dict[str, ColumnStats],
                  min_phi: float = MIN_PHI) -> list[Finding]:
    alvos = _null_cols(columns)
    if len(alvos) < 2:
        return []
    pares = [(a, b) for i, a in enumerate(alvos) for b in alvos[i + 1:]]
    sel, back = [], []
    for a, b in pares:
        ia = f"CASE WHEN {q(a.name)} IS NULL THEN 1.0 ELSE 0.0 END"
        ib = f"CASE WHEN {q(b.name)} IS NULL THEN 1.0 ELSE 0.0 END"
        sel.append(f"corr({ia}, {ib}) AS p{len(back)}")
        back.append((a, b))
    out = []
    for s in range(0, len(sel), 16):
        row = con.execute(f"SELECT {', '.join(sel[s:s + 16])} FROM {src}").fetchone()
        for (a, b), phi in zip(back[s:s + 16], row):
            if phi is None or abs(phi) < min_phi:
                continue
            out.append(Finding(
                coluna=f"{a.name} + {b.name}", detector="D3", tipo="nulo_entre_colunas",
                confianca=round(min(0.95, abs(phi)), 3),
                evidencia={"phi": round(float(phi), 4), "colunas": [a.name, b.name],
                           "taxa_a": round(a.null_rate, 4), "taxa_b": round(b.null_rate, 4)},
                nota=(f"nulos coincidem (phi = {phi:.2f}): as duas colunas provavelmente "
                      f"vem da mesma origem, e ela falhou junto"),
            ))
    return out


def contra_alvo(con, src: str, columns: dict[str, ColumnStats], target: str,
                min_delta: float = MIN_DELTA_ALVO) -> list[Finding]:
    alvos = [c for c in _null_cols(columns) if c.name != target]
    if not alvos or target not in columns:
        return []
    sel, back = [], []
    for c in alvos:
        sel.append(f"avg({q(target)}) FILTER (WHERE {q(c.name)} IS NULL) AS a{len(back)}")
        sel.append(f"avg({q(target)}) FILTER (WHERE {q(c.name)} IS NOT NULL) AS b{len(back)}")
        back.append(c)
    out = []
    row = con.execute(f"SELECT {', '.join(sel)} FROM {src}").fetchone()
    for i, c in enumerate(back):
        com, sem = row[2 * i], row[2 * i + 1]
        if com is None or sem is None:
            continue
        d = abs(float(com) - float(sem))
        if d < min_delta:
            continue
        out.append(Finding(
            coluna=c.name, detector="D3", tipo="nulo_informativo",
            massa=c.null_rate, confianca=round(min(0.95, 0.50 + 2 * d), 3),
            evidencia={"alvo": target, "taxa_evento_com_nulo": round(float(com), 4),
                       "taxa_evento_sem_nulo": round(float(sem), 4), "delta": round(d, 4)},
            nota=(f"taxa de evento {float(com):.1%} quando nulo contra {float(sem):.1%} "
                  f"quando preenchido: o nulo e sinal, imputar apaga informacao"),
        ))
    return out


def detect(con, src: str, columns: dict[str, ColumnStats], time_col: str | None = None,
           target: str | None = None) -> list[Finding]:
    out: list[Finding] = []
    tc = time_col or pick_time_column(columns)
    if tc and tc in columns:
        out += por_tempo(con, src, columns, tc)
    out += por_categoria(con, src, columns)
    out += entre_colunas(con, src, columns)
    if target:
        out += contra_alvo(con, src, columns, target)
    return out
