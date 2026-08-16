"""Roda o contrato contra um lote novo. E o passo que fecha o loop.

Sem isto o projeto e um relatorio; com isto e uma ferramenta de CI. Tudo em uma
passada por classe de expressao, como no resto do motor, e com codigo de saida != 0
quando um check falha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..scan.duckdb_pass import Expr, Runner, BATCH_CPU, classify, q
from .stats import alpha_corrigida, duas_proporcoes, psi, wilson


@dataclass
class Resultado:
    id: str
    coluna: str
    tipo: str
    passou: bool
    observado: Any = None
    referencia: Any = None
    detalhe: str = ""
    evidencia: dict = field(default_factory=dict)


def _lit(v: Any, kind: str) -> str | None:
    if v is None:
        return None
    if kind == "numeric":
        try:
            return repr(float(v))
        except (TypeError, ValueError):
            return None
    if kind == "temporal":
        return f"DATE '{str(v)[:10]}'"
    return "'" + str(v).replace("'", "''") + "'"


def validar(con, src: str, contrato: dict) -> list[Resultado]:
    checks = contrato["checks"]
    politica = contrato["politica"]
    alpha = alpha_corrigida(politica["alpha"], len(checks))

    desc = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    tipos = {r[0]: r[1] for r in desc}
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    out: list[Resultado] = []
    runner = Runner(con, src, BATCH_CPU)
    exprs: list[Expr] = []
    mapa: dict[str, dict] = {}

    def pedir(chave: str, sql: str, klass: str, ck: dict, extra: dict | None = None):
        alias = f"v{len(mapa)}"
        exprs.append(Expr(alias, sql, klass))
        mapa[alias] = {"check": ck, "chave": chave, **(extra or {})}

    # --- schema primeiro: check em coluna ausente nao tem o que medir --------
    ausentes = set()
    for ck in checks:
        col = ck["coluna"]
        if col in tipos:
            continue
        if col not in ausentes:
            ausentes.add(col)
            out.append(Resultado(id=f"{col}.presente", coluna=col, tipo="schema",
                                 passou=False, observado="ausente",
                                 referencia=contrato["colunas"].get(col, {}).get("tipo"),
                                 detalhe="coluna do contrato nao existe no lote novo"))

    for ck in checks:
        col = ck["coluna"]
        if col in ausentes:
            continue
        c = q(col)
        kind = classify(tipos[col])
        if ck["tipo"] == "proporcao":
            pedir("nn", f"count({c})", "cheap", ck)
        elif ck["tipo"] == "faixa":
            lo = _lit(ck["min"], kind)
            hi = _lit(ck["max"], kind)
            if lo and hi:
                pedir("fora", f"count(*) FILTER (WHERE {c} < {lo} OR {c} > {hi})", "filter", ck)
                pedir("novo_min", f"min({c})", "cheap", ck)
                pedir("novo_max", f"max({c})", "cheap", ck)
        elif ck["tipo"] == "sentinela":
            lit = _lit(ck["valor"], kind)
            if lit:
                pedir("cnt", f"count(*) FILTER (WHERE {c} = {lit})", "filter", ck)
        elif ck["tipo"] == "niveis":
            conhecidos = ", ".join(_lit(v, kind) or "NULL" for v in ck["conhecidos"])
            pedir("novos", f"count(*) FILTER (WHERE {c} IS NOT NULL "
                           f"AND {c} NOT IN ({conhecidos}))", "filter", ck)
            pedir("amostra", f"min({c}) FILTER (WHERE {c} IS NOT NULL "
                             f"AND {c} NOT IN ({conhecidos}))", "cheap", ck)
        elif ck["tipo"] == "psi":
            bordas = ck["bordas"]
            for i in range(len(bordas) + 1):
                if i == 0:
                    cond = f"{c} < {bordas[0]!r}"
                elif i == len(bordas):
                    cond = f"{c} >= {bordas[-1]!r}"
                else:
                    cond = f"{c} >= {bordas[i-1]!r} AND {c} < {bordas[i]!r}"
                pedir(f"bin{i}", f"count(*) FILTER (WHERE {cond})", "filter", ck, {"bin": i})

    got = runner.run(exprs) if exprs else {}
    por_check: dict[str, dict] = {}
    for alias, meta in mapa.items():
        cid = meta["check"]["id"]
        d = por_check.setdefault(cid, {"check": meta["check"], "bins": {}})
        if meta["chave"].startswith("bin"):
            d["bins"][meta["bin"]] = got.get(alias) or 0
        else:
            d[meta["chave"]] = got.get(alias)

    for cid, d in por_check.items():
        ck = d["check"]
        col, tipo = ck["coluna"], ck["tipo"]

        if tipo == "proporcao":
            p_novo = 1 - ((d.get("nn") or 0) / n if n else 0)
            p_ref = ck["p_ref"]
            z, pval = duas_proporcoes(p_ref, ck["n_ref"], p_novo, n)
            efeito = abs(p_novo - p_ref)
            passou = not (pval < alpha and efeito >= ck["efeito_minimo"])
            lo, hi = wilson(p_novo, n)
            out.append(Resultado(
                id=cid, coluna=col, tipo=tipo, passou=passou,
                observado=round(p_novo, 6), referencia=round(p_ref, 6),
                detalhe=(f"taxa de nulo {p_ref:.2%} -> {p_novo:.2%} "
                         f"(delta {p_novo - p_ref:+.2%}, p={pval:.2g})"),
                evidencia={"z": round(z, 2), "p_valor": pval, "alpha_corrigida": alpha,
                           "efeito": round(efeito, 6), "efeito_minimo": ck["efeito_minimo"],
                           "wilson_95_no_lote": [round(lo, 6), round(hi, 6)],
                           "n_ref": ck["n_ref"], "n_novo": n}))

        elif tipo == "faixa":
            fora = d.get("fora") or 0
            out.append(Resultado(
                id=cid, coluna=col, tipo=tipo, passou=fora == 0,
                observado=[d.get("novo_min"), d.get("novo_max")],
                referencia=[ck["min"], ck["max"]],
                detalhe=(f"{fora:,} linhas fora de [{ck['min']}, {ck['max']}]"
                         if fora else "dentro da faixa"),
                evidencia={"linhas_fora": fora, "fracao": fora / n if n else 0}))

        elif tipo == "sentinela":
            cnt = d.get("cnt") or 0
            p_novo = cnt / n if n else 0
            p_ref = ck["massa_ref"]
            z, pval = duas_proporcoes(p_ref, ck["n_ref"], p_novo, n)
            cresceu = p_novo > p_ref
            passou = not (cresceu and pval < alpha and (p_novo - p_ref) >= ck["efeito_minimo"])
            out.append(Resultado(
                id=cid, coluna=col, tipo=tipo, passou=passou,
                observado=round(p_novo, 6), referencia=round(p_ref, 6),
                detalhe=(f"massa de {ck['valor']!r}: {p_ref:.2%} -> {p_novo:.2%} "
                         f"({'cresceu' if cresceu else 'nao cresceu'})"),
                evidencia={"valor": ck["valor"], "linhas": cnt, "p_valor": pval,
                           "alpha_corrigida": alpha}))

        elif tipo == "niveis":
            novos = d.get("novos") or 0
            out.append(Resultado(
                id=cid, coluna=col, tipo=tipo, passou=novos == 0,
                observado=d.get("amostra"), referencia=f"{len(ck['conhecidos'])} niveis",
                detalhe=(f"{novos:,} linhas com nivel fora do contrato "
                         f"(ex.: {d.get('amostra')!r})" if novos else "sem nivel novo"),
                evidencia={"linhas": novos, "exemplo": d.get("amostra")}))

        elif tipo == "psi":
            bins = [d["bins"].get(i, 0) for i in range(len(ck["bordas"]) + 1)]
            tot = sum(bins)
            if not tot:
                continue
            obs = [b / tot for b in bins]
            valor = psi(ck["esperado"], obs)
            out.append(Resultado(
                id=cid, coluna=col, tipo=tipo, passou=valor < ck["limite"],
                observado=round(valor, 4), referencia=ck["limite"],
                detalhe=(f"PSI {valor:.4f} contra limite {ck['limite']}" +
                         ("" if valor < ck["limite"] else "  <- distribuicao mudou")),
                evidencia={"esperado": ck["esperado"], "observado": [round(x, 6) for x in obs],
                           "bordas": ck["bordas"]}))

    ordem = {"schema": 0, "niveis": 1, "sentinela": 2, "proporcao": 3, "faixa": 4, "psi": 5}
    out.sort(key=lambda r: (r.passou, ordem.get(r.tipo, 9), r.id))
    return out
