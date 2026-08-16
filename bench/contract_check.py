"""Verifica o contrato em tres cenarios, e mede o que a tolerancia adaptativa evita.

Um contrato precisa passar em tres testes antes de merecer confianca:

    1. contra a **propria base** que o gerou       -> zero falhas (senao esta quebrado)
    2. contra uma **amostra limpa e pequena**      -> zero falhas (senao alarma a toa)
    3. contra uma **safra deteriorada**            -> pega todas as mudancas conhecidas

O cenario 2 e o que separa este contrato de uma suite de tolerancia fixa. O contrato
foi emitido de 5 M linhas, onde o intervalo de Wilson sobre a taxa de nulo tem meia
largura de ~0,03 p.p. Guardar esse intervalo e compara-lo contra um lote de 10 mil
linhas — cujo ruido amostral proprio e ~20x maior — produz alarme sem que nada tenha
mudado. Este script conta quantos.

    python bench/contract_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdd.contract import emit as emit_mod
from sdd.contract import validate as val
from sdd.contract.stats import wilson
from sdd.detect import missingness, sentinels
from sdd.detect.base import Finding
from sdd.scan import duckdb_pass as dp
from sdd.scan import metadata


def construir(paths, target):
    res = dp.scan(paths)
    con = dp.connect()
    src = dp.source_expr(paths)
    achados: list[Finding] = []
    for f in metadata.metadata_findings(metadata.scan_metadata(paths)):
        if f["achado"] in ("coluna_constante", "coluna_100pct_nula"):
            achados.append(Finding(coluna=f["coluna"], detector="D0", tipo=f["achado"],
                                   confianca=f["confianca"], evidencia={}, nota=f["detalhe"]))
    achados += sentinels.detect(con, src, res.columns)
    achados += missingness.detect(con, src, res.columns, target=target)
    contrato = emit_mod.emitir(res, achados, fonte=str(paths[0]), con=con, src=src)
    return contrato, con, src


def rodar(contrato, src, con, rotulo) -> list:
    falhas = [r for r in val.validar(con, src, contrato) if not r.passou]
    marca = "OK  " if not falhas else "FALHA"
    print(f"  {marca} {rotulo:<44}{len(falhas)} falha(s)")
    for r in falhas:
        print(f"        [{r.tipo}] {r.id}: {r.detalhe}")
    return falhas


def tolerancia_fixa(contrato, con, src, n) -> list[str]:
    """Quantos checks de proporcao falhariam sob o desenho ingenuo.

    Ingenuo = guardar o intervalo de Wilson calculado na emissao e exigir que a taxa do
    lote novo caia dentro dele. E o que quase toda suite de validacao faz.
    """
    alarmes = []
    for ck in contrato["checks"]:
        if ck["tipo"] != "proporcao":
            continue
        col = ck["coluna"]
        try:
            nn = con.execute(f'SELECT count("{col}") FROM {src}').fetchone()[0]
        except Exception:
            continue
        p_novo = 1 - (nn / n if n else 0)
        lo, hi = wilson(ck["p_ref"], ck["n_ref"])
        if not (lo <= p_novo <= hi):
            alarmes.append(f"{ck['id']}: {p_novo:.4%} fora de [{lo:.4%}, {hi:.4%}]")
    return alarmes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/contas.parquet")
    ap.add_argument("--drift", default="data/safra_nova.parquet")
    ap.add_argument("--target", default="inadimplente")
    ap.add_argument("--amostra", type=int, default=10_000)
    ap.add_argument("--out", default="reports/contract_check.json")
    args = ap.parse_args()

    paths = metadata.resolve(args.base)
    contrato, con, src = construir(paths, args.target)
    print(f"contrato de {contrato['fonte']['linhas']:,} linhas | "
          f"{len(contrato['checks'])} checks\n")

    print("CENARIO 1  a propria base de referencia")
    f1 = rodar(contrato, src, con, "contas.parquet (5 M linhas)")

    print(f"\nCENARIO 2  amostra limpa de {args.amostra:,} linhas (nada mudou)")
    con.execute(f"CREATE OR REPLACE TEMP TABLE _amostra AS "
                f"SELECT * FROM {src} USING SAMPLE {args.amostra} ROWS")
    n_am = con.execute("SELECT count(*) FROM _amostra").fetchone()[0]
    f2 = rodar(contrato, "_amostra", con, f"amostra ({n_am:,} linhas)")

    ingenuo = tolerancia_fixa(contrato, con, "_amostra", n_am)
    print(f"\n  Sob tolerancia FIXA (intervalo de Wilson da emissao, o desenho ingenuo):")
    print(f"  {len(ingenuo)} alarme(s) na mesma amostra limpa")
    for a in ingenuo[:6]:
        print(f"        {a}")
    if len(ingenuo) > 6:
        print(f"        ... e mais {len(ingenuo) - 6}")

    print(f"\nCENARIO 3  safra deteriorada")
    dpaths = metadata.resolve(args.drift)
    dsrc = dp.source_expr(dpaths)
    f3 = rodar(contrato, dsrc, con, Path(args.drift).name)

    mud = json.loads(Path(args.drift).with_suffix(".mudancas.json").read_text(encoding="utf-8"))
    ids = {r.id for r in [type("R", (), {"id": x.id})() for x in f3]} if False else {r.id for r in f3}
    print("\n  mudancas conhecidas x checks que dispararam:")
    faltou = 0
    for m in mud["mudancas"]:
        ok = m["check"] in ids
        faltou += 0 if ok else 1
        print(f"        {'pego  ' if ok else 'PERDEU'} {m['check']:<32}{m['detalhe'][:58]}")
    extras = sorted(ids - {m["check"] for m in mud["mudancas"]})
    if extras:
        print(f"\n  {len(extras)} check(s) alem do manifesto — consequencias das mesmas "
              f"transformacoes:")
        for e in extras:
            print(f"        {e}")

    print("\n" + "=" * 66)
    print(f"cenario 1 (propria base)      {len(f1)} falhas   esperado 0")
    print(f"cenario 2 (amostra limpa)     {len(f2)} falhas   esperado 0   "
          f"| tolerancia fixa daria {len(ingenuo)}")
    print(f"cenario 3 (safra deteriorada) {len(f3)} falhas   "
          f"{len(mud['mudancas']) - faltou}/{len(mud['mudancas'])} mudancas conhecidas pegas")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checks": len(contrato["checks"]),
        "cenario_1_falhas": [r.id for r in f1],
        "cenario_2_falhas": [r.id for r in f2],
        "cenario_2_tolerancia_fixa": ingenuo,
        "cenario_3_falhas": sorted(ids),
        "mudancas_perdidas": faltou,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
