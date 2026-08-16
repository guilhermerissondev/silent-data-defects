"""Nivel B — regra de decisao declarada, medida exata, sem modelo.

A resposta honesta primeiro: para afirmar "corrigir a coluna X muda a decisao" e preciso
a funcao de X para decisao, e essa funcao e o modelo. Nao existe atalho que de o numero
exato sem ela. O que existe, com a regra declarada, sao duas medidas exatas — e elas
medem coisas diferentes conforme o defeito esteja ou nao na coluna que decide.

**B1 — o defeito esta na coluna da decisao.** Ai nao falta nada: repara-se o valor,
recomputa-se a decisao, conta-se quem atravessou. E taxa de inversao de verdade.

**B2 — o defeito esta em outra coluna.** Sem modelo nao da para propagar. O que da,
exato, e medir a **exposicao da decisao ao defeito**: onde estao os registros afetados
em relacao a fronteira. Um defeito em 3% das contas que estao todas no fundo da regiao
de aprovacao nao inverte decisao nenhuma por mais grave que seja; o mesmo defeito
concentrado na margem e o pior caso possivel. A medida e:

    exposicao = P(defeito E perto da fronteira)

com "perto" definido em percentil do escore (banda de +-5 p.p. em volta do corte, entao
a taxa-base e 10% por construcao e o lift sai direto). Isso e um **limite superior** das
decisoes que poderiam inverter, nunca a taxa de inversao — e o relatorio diz isso.
"""

from __future__ import annotations

from typing import Any

from ..detect.base import Finding
from ..scan.duckdb_pass import BATCH_CPU, Expr, Runner, classify, q

BANDA = 0.05     # meia-largura da margem, em percentil do escore


def _condicao(f: Finding, kind: str) -> str | None:
    """Condicao SQL que isola as linhas afetadas pelo defeito."""
    c = q(f.coluna)
    if f.tipo in ("sentinela_numerica", "massa_pontual_em_continua",
                  "sentinela_textual", "sentinela_temporal"):
        if f.valor is None:
            return None
        if kind == "numeric":
            return f"{c} = {float(f.valor)!r}"
        if kind == "temporal":
            return f"{c} = DATE '{str(f.valor)[:10]}'"
        return f"{c} = '" + str(f.valor).replace("'", "''") + "'"
    if f.tipo in ("nulo_por_tempo", "nulo_por_categoria", "nulo_informativo"):
        return f"{c} IS NULL"
    return None  # coluna constante, nulo entre colunas: nao ha subconjunto afetado


def fronteira(con, src: str, decisao: dict) -> dict:
    """Traduz a regra declarada em valores de corte e na banda de margem."""
    col = q(decisao["coluna"])
    tipo = decisao.get("tipo", "corte")

    if tipo == "corte":
        corte = float(decisao["corte"])
        pct = con.execute(
            f"SELECT count(*) FILTER (WHERE {col} < {corte!r}) / count({col})::DOUBLE "
            f"FROM {src}").fetchone()[0] or 0.0
        cortes = [corte]
    elif tipo == "top_k":
        pct = 1.0 - float(decisao["fracao"])
        corte = con.execute(
            f"SELECT quantile_cont({col}, {pct!r}) FROM {src}").fetchone()[0]
        cortes = [float(corte)]
    else:
        raise ValueError(f"tipo de decisao nao suportado: {tipo}")

    lo_p, hi_p = max(0.0, pct - BANDA), min(1.0, pct + BANDA)
    lo, hi = con.execute(
        f"SELECT quantile_cont({col}, {lo_p!r}), quantile_cont({col}, {hi_p!r}) "
        f"FROM {src}").fetchone()
    return {"tipo": tipo, "coluna": decisao["coluna"], "cortes": cortes,
            "percentil_do_corte": round(float(pct), 6),
            "banda_percentil": [round(lo_p, 6), round(hi_p, 6)],
            "banda_valor": [float(lo), float(hi)],
            "taxa_base_na_margem": round(hi_p - lo_p, 6)}


def medir(con, src: str, achados: list[Finding], decisao: dict,
          columns: dict) -> dict[int, dict]:
    """Uma query em lote para todos os achados. Devolve {id(achado): impacto}."""
    fr = fronteira(con, src, decisao)
    dcol = q(decisao["coluna"])
    lo, hi = fr["banda_valor"]
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    exprs, mapa = [], {}
    for f in achados:
        cs = columns.get(f.coluna)
        if cs is None:
            continue
        cond = _condicao(f, cs.kind)
        if cond is None:
            continue
        i = len(mapa)
        exprs.append(Expr(f"a{i}", f"count(*) FILTER (WHERE {cond})", "filter"))
        exprs.append(Expr(f"m{i}", f"count(*) FILTER (WHERE ({cond}) "
                                   f"AND {dcol} >= {lo!r} AND {dcol} <= {hi!r})", "filter"))
        mapa[i] = (f, cond, cs)
    if not exprs:
        return {}
    got = Runner(con, src, BATCH_CPU).run(exprs)

    out: dict[int, dict] = {}
    for i, (f, cond, cs) in mapa.items():
        afet = got.get(f"a{i}") or 0
        marg = got.get(f"m{i}") or 0
        if not afet:
            continue
        p_afet = afet / n
        p_exp = marg / n
        p_marg_dado = marg / afet
        base = fr["taxa_base_na_margem"]
        lift = p_marg_dado / base if base else 0.0

        if f.coluna == decisao["coluna"]:
            out[id(f)] = _b1(con, src, f, cond, decisao, fr, n, afet)
        else:
            out[id(f)] = {
                "nivel": "B",
                "sub": "B2_exposicao",
                "metrica": "P(defeito E perto da fronteira)",
                "valor": round(p_exp, 8),
                "massa_afetada": round(p_afet, 6),
                "afetados_na_margem": round(p_marg_dado, 6),
                "taxa_base_na_margem": base,
                "lift_na_margem": round(lift, 3),
                "fronteira": fr,
                "significa": (f"{p_exp:.2%} das linhas tem o defeito E estao na margem "
                              f"da decisao. E LIMITE SUPERIOR das decisoes que poderiam "
                              f"inverter, nao taxa de inversao — propagar o defeito ate "
                              f"a decisao exige o modelo (nivel C)"),
            }
    return out


def _b1(con, src: str, f: Finding, cond: str, decisao: dict, fr: dict,
        n: int, afet: int) -> dict:
    """Defeito na propria coluna de decisao: da para contar a inversao exata.

    O reparo precisa ser declarado, porque "corrigir" nao e uma operacao unica. Aqui se
    mede o reparo mais conservador — tratar o valor como ausente, ou seja, a linha deixa
    de ser decidida — e o mais comum na pratica, imputar pela mediana do restante.
    """
    c = q(decisao["coluna"])
    corte = fr["cortes"][0]
    lado_atual = f"{c} >= {corte!r}"
    mediana = con.execute(
        f"SELECT quantile_cont({c}, 0.5) FROM {src} WHERE NOT ({cond})").fetchone()[0]
    mediana = float(mediana) if mediana is not None else None

    aprovados_defeito = con.execute(
        f"SELECT count(*) FILTER (WHERE ({cond}) AND {lado_atual}) FROM {src}").fetchone()[0]
    if mediana is None:
        invertidos = 0
    else:
        # apos imputar pela mediana, todo afetado cai do mesmo lado
        lado_novo = mediana >= corte
        invertidos = (afet - aprovados_defeito) if lado_novo else aprovados_defeito

    return {
        "nivel": "B",
        "sub": "B1_inversao_exata",
        "metrica": "taxa de inversao de decisao",
        "valor": round(invertidos / n, 8),
        "massa_afetada": round(afet / n, 6),
        "reparo": f"imputar pela mediana do restante ({mediana})",
        "decisoes_invertidas": invertidos,
        "fronteira": fr,
        "significa": (f"{invertidos / n:.4%} das linhas mudam de lado do corte "
                      f"{corte} ao reparar o defeito. Exato: o defeito esta na propria "
                      f"coluna que decide, entao nao falta modelo nenhum"),
    }
