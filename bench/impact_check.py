"""A tese do projeto, em uma tabela: prevalencia e impacto ordenam diferente.

Toda ferramenta de qualidade de dados ordena por prevalencia. Este script coloca as
duas ordenacoes lado a lado sobre a mesma base e mede o deslocamento. Se as duas
listas forem iguais, o projeto inteiro nao se justifica.

    python bench/impact_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdd import impact
from sdd.detect import missingness, sentinels
from sdd.detect.base import Finding
from sdd.scan import duckdb_pass as dp
from sdd.scan import metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/contas.parquet")
    ap.add_argument("--target", default="inadimplente")
    ap.add_argument("--decision-column", default="score")
    ap.add_argument("--decision-cutoff", type=float, default=0.70)
    ap.add_argument("--out", default="reports/impact_check.json")
    args = ap.parse_args()

    paths = metadata.resolve(args.path)
    res = dp.scan(paths)
    con = dp.connect()
    src = dp.source_expr(paths)

    achados: list[Finding] = []
    for f in metadata.metadata_findings(metadata.scan_metadata(paths)):
        if f["achado"] in ("coluna_constante", "coluna_100pct_nula"):
            achados.append(Finding(coluna=f["coluna"], detector="D0", tipo=f["achado"],
                                   confianca=f["confianca"], evidencia={}, nota=f["detalhe"]))
    achados += sentinels.detect(con, src, res.columns)
    achados += missingness.detect(con, src, res.columns, target=args.target)

    decisao = {"tipo": "corte", "coluna": args.decision_column, "corte": args.decision_cutoff}
    linhas = impact.avaliar(con, src, achados, res.columns, alvo=args.target, decisao=decisao)

    com_massa = [l for l in linhas if l["impacto"] and
                 l["impacto"].get("massa_afetada") is not None]
    por_massa = sorted(com_massa, key=lambda l: -l["impacto"]["massa_afetada"])
    por_impacto = sorted(com_massa, key=lambda l: -l["impacto"]["valor"])

    def rot(l) -> str:
        f = l["achado"]
        return f"{f.coluna}" + ("" if f.valor is None else f" = {f.valor!r}")

    pos_massa = {rot(l): i for i, l in enumerate(por_massa)}

    print(f"{args.path}  {res.rows:,} linhas | decisao: "
          f"{args.decision_column} >= {args.decision_cutoff}\n")
    print(f"{'#':>2}  {'POR PREVALENCIA (o que toda ferramenta faz)':<44}"
          f"{'#':>3}  {'POR IMPACTO NA DECISAO':<40}{'move':>7}")
    print("-" * 104)
    for i, (a, b) in enumerate(zip(por_massa, por_impacto)):
        ra, rb = rot(a), rot(b)
        delta = pos_massa[rb] - i
        seta = f"{delta:+d}" if delta else "="
        print(f"{i+1:>2}. {ra:<41}{a['impacto']['massa_afetada']:>6.1%}"
              f"{i+1:>3}. {rb:<32}{b['impacto']['valor']:>8.5f}{seta:>7}")

    print("\nO QUE MUDA DE LUGAR")
    for i, l in enumerate(por_impacto):
        r = rot(l)
        d = pos_massa[r] - i
        if abs(d) < 2:
            continue
        imp = l["impacto"]
        lift = imp.get("lift_na_margem")
        motivo = (f"lift {lift:.2f}x na margem" if lift is not None
                  else f"{imp['sub']}")
        print(f"  {r:<40}posicao {pos_massa[r]+1} -> {i+1}  ({motivo})")

    zerados = [l for l in por_impacto if l["impacto"]["valor"] == 0]
    if zerados:
        print("\nDEFEITOS COM IMPACTO MEDIDO IGUAL A ZERO")
        for l in zerados:
            imp = l["impacto"]
            print(f"  {rot(l):<40}{imp['massa_afetada']:>6.1%} das linhas, "
                  f"{imp['decisoes_invertidas']:,} decisoes invertidas ({imp['sub']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "por_massa": [{"achado": rot(l), "massa": l["impacto"]["massa_afetada"]}
                      for l in por_massa],
        "por_impacto": [{"achado": rot(l), "valor": l["impacto"]["valor"],
                         "nivel": l["impacto"]["nivel"], "sub": l["impacto"].get("sub"),
                         "lift": l["impacto"].get("lift_na_margem"),
                         "deslocamento": pos_massa[rot(l)] - i}
                        for i, l in enumerate(por_impacto)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
