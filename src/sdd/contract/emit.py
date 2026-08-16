"""Emite o contrato de validacao a partir do perfil e dos achados.

E o passo que o Great Expectations deixa manual — escrever a suite de expectativas e
o trabalho dificil, e e onde a ferramenta morre na vida real. Aqui o contrato sai do
que foi medido.

O que o contrato guarda por coluna: a referencia (p_ref, n_ref, faixa, quantis, ndv,
niveis) e os checks derivados dela. O que ele **nao** guarda: severidade. Ela vem do
impacto na decisao (marco 4), e um contrato que se auto-atribui severidade ordena por
prevalencia — o erro que este projeto existe para nao cometer. Ate la o campo sai como
`indefinida`, explicitamente.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..detect.base import Finding
from ..types import ColumnStats, ScanResult, _plain
from .stats import wilson

CONTRACT_VERSION = 1

# Piso de efeito pratico. Constante declarada, nao derivada — a versao principiada
# (piso vindo do impacto na decisao) e o marco 4, e o contrato diz isso de si mesmo.
EFEITO_TAXA = 0.005        # 0,5 p.p. em qualquer proporcao
EFEITO_AMPLITUDE = 0.02    # 2% da amplitude, para deslocamento numerico
PSI_LIMITE = 0.10
ALPHA = 0.01


NDV_RATIO_ID = 0.5     # acima disto a coluna e identificador: distribuicao nao diz nada


def _bordas(cs: ColumnStats) -> list[float] | None:
    """Bordas de bin a partir dos quantis de referencia.

    Bordas repetidas aparecem sempre que ha massa pontual — em coluna discreta, metade
    dos quantis costuma ser o mesmo valor. Colapsa-las e obrigatorio; o que NAO se pode
    fazer e reaproveitar os niveis de probabilidade como proporcao esperada depois do
    colapso. Ver `proporcoes_de_referencia`.
    """
    if not cs.quantiles:
        return None
    # identificador nao tem distribuicao que valha checar: o PSI dispara so porque o
    # lote novo tem outras chaves, e o alarme nao informa nada
    if (cs.ndv_ratio or 0) > NDV_RATIO_ID:
        return None
    edges: list[float] = []
    for v in cs.quantiles.values():
        f = float(v)
        if not edges or f > edges[-1]:
            edges.append(f)
    return edges if len(edges) >= 3 else None


def proporcoes_de_referencia(con, src: str, bordas_por_coluna: dict[str, list[float]],
                             ) -> dict[str, list[float]]:
    """Mede a proporcao de cada bin **na propria base de referencia**.

    Este era o bug: as proporcoes esperadas vinham dos niveis de probabilidade dos
    quantis, o que so vale se as bordas forem todas distintas. Em coluna discreta as
    bordas colapsam e o esperado deixa de descrever os bins. Medido em `n_atrasos_12m`:

        esperado inferido  [0,0010, 0,7490, 0,2000, 0,0400, 0,0090, 0,0010]
        proporcao real     [0,0000, 0,5488, 0,3290, 0,0989, 0,0199, 0,0034]

    PSI da referencia contra ela mesma: ~0,13, acima do limite de 0,10. O contrato
    reprovaria a base que o gerou. Nenhum lote precisa mudar para o alarme tocar — e
    exatamente assim que uma suite de validacao vira ruido e e desligada.

    A correcao e medir, nao inferir. Custa uma query em lote no momento da emissao.
    """
    from ..scan.duckdb_pass import BATCH_CPU, Expr, Runner, q

    exprs, mapa = [], {}
    for col, bordas in bordas_por_coluna.items():
        c = q(col)
        for i in range(len(bordas) + 1):
            if i == 0:
                cond = f"{c} < {bordas[0]!r}"
            elif i == len(bordas):
                cond = f"{c} >= {bordas[-1]!r}"
            else:
                cond = f"{c} >= {bordas[i-1]!r} AND {c} < {bordas[i]!r}"
            alias = f"r{len(mapa)}"
            exprs.append(Expr(alias, f"count(*) FILTER (WHERE {cond})", "filter"))
            mapa[alias] = (col, i)
    if not exprs:
        return {}
    got = Runner(con, src, BATCH_CPU).run(exprs)

    bruto: dict[str, dict[int, int]] = {}
    for alias, (col, i) in mapa.items():
        bruto.setdefault(col, {})[i] = got.get(alias) or 0
    out = {}
    for col, bins in bruto.items():
        n = len(bordas_por_coluna[col]) + 1
        v = [bins.get(i, 0) for i in range(n)]
        tot = sum(v)
        if tot:
            out[col] = [x / tot for x in v]
    return out


def emitir(res: ScanResult, achados: list[Finding], *, fonte: str, con=None, src: str = "",
           alpha: float = ALPHA, efeito_taxa: float = EFEITO_TAXA,
           efeito_amplitude: float = EFEITO_AMPLITUDE,
           psi_limite: float = PSI_LIMITE,
           decisao: dict | None = None) -> dict:
    colunas: dict[str, Any] = {}
    checks: list[dict] = []

    # As proporcoes esperadas do PSI sao MEDIDAS na base de referencia, nunca inferidas
    # dos niveis de probabilidade dos quantis. Sem `con` nao ha como medir, e entao nao
    # se emite check de distribuicao — melhor um contrato menor que um contrato errado.
    bordas = {nome: b for nome, cs in res.columns.items()
              if cs.kind == "numeric" and (b := _bordas(cs))}
    esperados = proporcoes_de_referencia(con, src, bordas) if (con is not None and bordas) else {}

    por_coluna: dict[str, list[Finding]] = {}
    for f in achados:
        por_coluna.setdefault(f.coluna, []).append(f)

    for nome, cs in res.columns.items():
        ref: dict[str, Any] = {"tipo": cs.sql_type, "classe": cs.kind}

        lo, hi = wilson(cs.null_rate, cs.n)
        ref["nulo"] = {
            "p_ref": round(cs.null_rate, 8), "n_ref": cs.n,
            "wilson_95_na_emissao": [round(lo, 8), round(hi, 8)],
        }
        checks.append({
            "id": f"{nome}.nulo", "coluna": nome, "tipo": "proporcao",
            "campo": "taxa_de_nulo", "p_ref": round(cs.null_rate, 8), "n_ref": cs.n,
            "efeito_minimo": efeito_taxa, "severidade": "indefinida",
            "porque": "tolerancia por teste de duas proporcoes com o n do lote novo",
        })

        if cs.kind in ("numeric", "temporal") and cs.min is not None:
            ref["faixa"] = {"min": _plain(cs.min), "max": _plain(cs.max)}
            checks.append({
                "id": f"{nome}.faixa", "coluna": nome, "tipo": "faixa",
                "min": _plain(cs.min), "max": _plain(cs.max),
                "severidade": "indefinida",
                "porque": "valor fora da faixa observada indica dominio novo ou unidade trocada",
            })

        if cs.ndv_approx is not None:
            ref["ndv"] = {"valor": cs.ndv_approx, "exato": cs.ndv_exact}

        # Niveis conhecidos: so quando o histograma cobriu a coluna inteira. Com top-k
        # nao da para afirmar que um nivel e novo — a lista de referencia e parcial.
        if cs.counts_complete and cs.kind in ("string", "boolean"):
            niveis = sorted(cs.value_counts)
            ref["niveis"] = niveis
            checks.append({
                "id": f"{nome}.niveis", "coluna": nome, "tipo": "niveis",
                "conhecidos": niveis, "severidade": "indefinida",
                "porque": "nivel novo em producao quebra encoding treinado no historico",
            })

        if cs.quantiles:
            ref["quantis"] = {k: _plain(v) for k, v in cs.quantiles.items()}
            ref["quantis_sao_aproximados"] = True
            if nome in esperados:
                checks.append({
                    "id": f"{nome}.distribuicao", "coluna": nome, "tipo": "psi",
                    "bordas": bordas[nome],
                    "esperado": [round(x, 8) for x in esperados[nome]],
                    "esperado_medido": True,
                    "limite": psi_limite, "severidade": "indefinida",
                    "porque": ("PSI por feature; agregado sobre o escore e pouco sensivel. "
                               "Proporcoes esperadas medidas na base de referencia, nao "
                               "inferidas dos niveis dos quantis"),
                })

        fs = por_coluna.get(nome, [])
        sent = [f for f in fs if f.tipo.startswith("sentinela")]
        if sent:
            ref["sentinelas"] = [{"valor": _plain(f.valor), "massa_ref": round(f.massa or 0, 8),
                                  "detector": f.detector,
                                  "evidencia": _plain(f.evidencia)} for f in sent]
            for f in sent:
                checks.append({
                    "id": f"{nome}.sentinela[{_plain(f.valor)}]", "coluna": nome,
                    "tipo": "sentinela", "valor": _plain(f.valor),
                    "massa_ref": round(f.massa or 0, 8), "n_ref": cs.n,
                    "efeito_minimo": efeito_taxa, "severidade": "indefinida",
                    "porque": ("massa da sentinela nao deve crescer. O alvo presumivel e "
                               "zero depois que a origem for corrigida, mas quem decide "
                               "isso e o dono do pipeline, nao o contrato"),
                })

        outros = [f for f in fs if not f.tipo.startswith("sentinela")]
        if outros:
            ref["achados_na_emissao"] = [{"detector": f.detector, "tipo": f.tipo,
                                          "nota": f.nota} for f in outros]
        colunas[nome] = ref

    contrato = {
        "contract_version": CONTRACT_VERSION,
        "emitido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": {"caminho": fonte, "linhas": res.rows, "colunas": len(res.columns),
                  "motor": res.engine},
        "politica": {
            "alpha": alpha,
            "correcao_multiplicidade": "bonferroni",
            "n_checks": len(checks),
            "efeito_minimo": {"taxa": efeito_taxa, "amplitude": efeito_amplitude},
            "psi_limite": psi_limite,
            "nota": ("Todo teste exige significancia E efeito minimo. Com milhoes de "
                     "linhas a significancia sozinha alarma por diferenca que nao muda "
                     "decisao nenhuma. O piso de efeito aqui e constante declarada; "
                     "deriva-lo do impacto na decisao e o marco 4."),
        },
        "decisao": decisao,
        "severidade": ("Ausente de proposito. Deriva do impacto na decisao (marco 4); "
                       "atribui-la aqui seria ordenar por prevalencia."),
        "colunas": colunas,
        "checks": checks,
    }
    return _plain(contrato)
