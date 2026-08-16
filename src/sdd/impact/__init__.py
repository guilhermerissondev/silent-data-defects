"""Impacto na decisao — a escada de evidencia.

    A  heuristica de ordenacao      sempre disponivel, sem alvo nem modelo
    B  regra de decisao declarada   exato no que mede, model-free
    C  modelo plugado               numero verdadeiro (marco 5)

A regra que faz a escada valer alguma coisa: **cada achado carrega a etiqueta do nivel
que produziu sua severidade**. Sem etiqueta, o numero mente — porque A e B nao medem a
mesma coisa e nem a mesma unidade. Fingir que o nivel A e o nivel C e exatamente o tipo
de desonestidade que este projeto existe para nao cometer.
"""

from __future__ import annotations

from ..detect.base import Finding
from . import boundary, leverage

ORDEM_NIVEL = {"B": 0, "A": 1, None: 2}


def avaliar(con, src: str, achados: list[Finding], columns: dict,
            alvo: str | None = None, decisao: dict | None = None) -> list[dict]:
    """Anexa o melhor nivel de evidencia disponivel a cada achado."""
    lev = leverage.alavancagem(con, src, columns, alvo) if alvo else {}
    b = boundary.medir(con, src, achados, decisao, columns) if decisao else {}

    linhas = []
    for f in achados:
        a = leverage.impacto_a(f.massa, lev.get(f.coluna))
        imp = b.get(id(f)) or a
        linhas.append({"achado": f, "impacto": imp, "nivel_a": a,
                       "nivel_b": b.get(id(f)),
                       "alavancagem": lev.get(f.coluna)})

    def chave(l):
        imp = l["impacto"]
        nivel = imp["nivel"] if imp else None
        return (ORDEM_NIVEL.get(nivel, 2), -(imp["valor"] if imp else 0))

    linhas.sort(key=chave)
    return linhas


def severidade(imp: dict | None) -> str:
    """Rotulo textual, sempre acompanhado do nivel que o produziu."""
    if not imp:
        return "sem evidencia"
    v = imp["valor"]
    if imp["nivel"] == "B":
        if v >= 0.01:
            return "alta"
        if v >= 0.001:
            return "media"
        return "baixa"
    # nivel A e ordinal: nao ha corte absoluto que signifique alguma coisa
    return "ordinal"
