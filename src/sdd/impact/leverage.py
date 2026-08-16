"""Nivel A — heuristica de priorizacao.

    impacto_A = massa_afetada x alavancagem_univariada

Serve **so para ordenar a lista de defeitos**. Nao e percentual de decisao alterada, e
o relatorio precisa dizer isso com essas palavras. Uma coluna com 30% de nulos e AUC
0,51 e ruido; uma com 2% de sentinelas e AUC 0,78 muda decisao — e essa comparacao e
tudo o que o nivel A autoriza a fazer.

A alavancagem e `2 x |AUC - 0,5|`, com a AUC univariada calculada exata (Mann-Whitney
via tabela de contingencia por valor distinto, que trata empate corretamente). Custa
2,3 s para 16 colunas numericas em 5 M linhas — barato o bastante para nao aproximar.

Ressalva em coluna categorica: a AUC sai de ordenar os niveis pela taxa de evento
observada na propria base. E o encoding univariado otimo *em amostra*, entao a AUC e
otimista por construcao. Fica marcada como tal.
"""

from __future__ import annotations

from ..scan.duckdb_pass import classify, q
from ..types import ColumnStats

# Acima disto a coluna praticamente responde o alvo sozinha: e vazamento, nao feature.
# O detector proprio e o D4 (marco 5); aqui so se registra o sinal, que sai de graca.
AUC_VAZAMENTO = 0.95


def _sql_auc(src: str, col: str, alvo: str, ordenar_por_taxa: bool) -> str:
    c, a = q(col), q(alvo)
    ordem = "n1 / (n1 + n0)::DOUBLE" if ordenar_por_taxa else "x"
    return f"""
    WITH g AS (
        SELECT {c} AS x,
               count(*) FILTER (WHERE {a} = 1) AS n1,
               count(*) FILTER (WHERE {a} = 0) AS n0
        FROM {src} WHERE {c} IS NOT NULL AND {a} IS NOT NULL GROUP BY 1),
         c AS (
        SELECT n1, n0,
               sum(n0) OVER (ORDER BY {ordem}
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n0_antes
        FROM g)
    SELECT sum(n1 * coalesce(n0_antes, 0) + 0.5 * n1 * n0) / nullif(sum(n1) * sum(n0), 0),
           sum(n1), sum(n0)
    FROM c"""


def alavancagem(con, src: str, columns: dict[str, ColumnStats], alvo: str,
                ) -> dict[str, dict]:
    """AUC univariada e alavancagem por coluna. Uma query por coluna: a agregacao por
    valor distinto nao se combina entre colunas."""
    out: dict[str, dict] = {}
    for nome, cs in columns.items():
        if nome == alvo or cs.kind == "other":
            continue
        por_taxa = cs.kind in ("string", "boolean")
        if por_taxa and (cs.ndv_approx or 0) > 4096:
            continue  # categoria de altissima cardinalidade: AUC em amostra nao diz nada
        try:
            auc, n1, n0 = con.execute(_sql_auc(src, nome, alvo, por_taxa)).fetchone()
        except Exception:
            continue
        if auc is None or not n1 or not n0:
            continue
        auc = float(auc)
        out[nome] = {
            "auc": round(auc, 6),
            "alavancagem": round(2 * abs(auc - 0.5), 6),
            "metodo": "auc_por_taxa_de_evento" if por_taxa else "auc_univariada",
            "otimista_em_amostra": por_taxa,
            "suspeita_de_vazamento": max(auc, 1 - auc) >= AUC_VAZAMENTO,
        }
    return out


def impacto_a(massa: float | None, lev: dict | None) -> dict | None:
    """Combina massa e alavancagem. O produto e ordinal, nao tem unidade."""
    if massa is None or not lev:
        return None
    return {
        "nivel": "A",
        "metrica": "massa x alavancagem",
        "valor": round(massa * lev["alavancagem"], 8),
        "massa": round(massa, 6),
        "alavancagem": lev["alavancagem"],
        "auc": lev["auc"],
        "significa": ("ordem relativa entre defeitos, nada mais. NAO e percentual de "
                      "decisao alterada"),
    }
