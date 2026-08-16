"""Benchmark de deteccao — precisao, recall e margem, contra o gabarito injetado.

Velocidade sem taxa de acerto nao vale nada. Este script roda os detectores sobre a
base com defeitos conhecidos e responde as duas perguntas que decidem se alguem usa a
ferramenta ou desliga o alerta:

    1. quanto ele acha (recall) e quanto do que ele acha presta (precisao)?
    2. quantos falsos positivos por 100 colunas limpas?

E uma terceira, que e a mais informativa e quase nunca aparece: **qual a margem**
entre o verdadeiro positivo mais fraco e o verdadeiro negativo mais forte. Precisao
1,00 com margem de 5% e sorte; com margem de duas ordens de grandeza e detector.

    python bench/evaluate.py data/contas.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdd.detect import missingness, sentinels
from sdd.detect.base import Finding
from sdd.scan import duckdb_pass as dp
from sdd.scan import metadata

IMPLEMENTADOS = {"D0", "D1", "D2", "D3"}


def coletar(paths, manifesto) -> tuple[list[Finding], list[dict]]:
    res = dp.scan(paths)
    con = dp.connect()
    src = dp.source_expr(paths)

    achados: list[Finding] = []
    for f in metadata.metadata_findings(metadata.scan_metadata(paths)):
        if f["achado"] in ("coluna_constante", "coluna_100pct_nula"):
            achados.append(Finding(coluna=f["coluna"], detector="D0", tipo=f["achado"],
                                   confianca=f["confianca"],
                                   evidencia={"camada": 0, "custo": "zero linhas lidas"},
                                   nota=f["detalhe"]))
    diag: list[dict] = []
    achados += sentinels.detect(con, src, res.columns, diag=diag)
    achados += missingness.detect(con, src, res.columns,
                                  time_col=manifesto.get("time_column"),
                                  target=manifesto.get("target"))
    return achados, diag


def avaliar(achados, manifesto) -> dict:
    defeitos = manifesto["defects"]
    limpas = set(manifesto["clean_columns"])

    esperados = [d for d in defeitos if d["esperado"]]
    armadilhas = [d for d in defeitos if not d["esperado"]]

    def casa(f, d) -> bool:
        if d["coluna"] not in f.coluna:          # D3 entre colunas usa "a + b"
            return False
        return f.detector == d["detector"] and f.tipo == d["tipo"]

    tp, fn, fp_armadilha, fp_limpa, outros = [], [], [], [], []
    usados = set()
    for d in esperados:
        hit = next((f for f in achados if casa(f, d) and id(f) not in usados), None)
        if hit is not None:
            usados.add(id(hit))
            tp.append((d, hit))
        else:
            fn.append(d)
    for f in achados:
        if id(f) in usados:
            continue
        arm = next((d for d in armadilhas if casa(f, d)), None)
        if arm is not None:
            fp_armadilha.append((arm, f))
        elif any(c == f.coluna for c in limpas):
            fp_limpa.append(f)
        else:
            outros.append(f)

    por_det: dict[str, dict] = {}
    for d, _ in tp:
        por_det.setdefault(d["detector"], {"tp": 0, "fn": 0, "fp": 0})["tp"] += 1
    for d in fn:
        por_det.setdefault(d["detector"], {"tp": 0, "fn": 0, "fp": 0})["fn"] += 1
    for _, f in fp_armadilha:
        por_det.setdefault(f.detector, {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1
    for f in fp_limpa:
        por_det.setdefault(f.detector, {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1

    return {"tp": tp, "fn": fn, "fp_armadilha": fp_armadilha, "fp_limpa": fp_limpa,
            "outros": outros, "por_detector": por_det, "n_limpas": len(limpas)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/contas.parquet")
    ap.add_argument("--out", default="reports/evaluate.json")
    args = ap.parse_args()

    paths = metadata.resolve(args.path)
    manifesto = json.loads(Path(args.path).with_suffix(".manifest.json").read_text(encoding="utf-8"))
    achados, diag = coletar(paths, manifesto)
    r = avaliar(achados, manifesto)

    print(f"{args.path}  {manifesto['rows']:,} linhas | "
          f"{len([d for d in manifesto['defects'] if d['esperado']])} defeitos esperados, "
          f"{len([d for d in manifesto['defects'] if not d['esperado']])} armadilha, "
          f"{r['n_limpas']} colunas limpas\n")

    print(f"{'detector':<12}{'TP':>4}{'FN':>4}{'FP':>4}{'precisao':>11}{'recall':>9}   estado")
    print("-" * 62)
    for det in sorted(set(list(r["por_detector"]) + [d["detector"] for d in manifesto["defects"]])):
        v = r["por_detector"].get(det, {"tp": 0, "fn": 0, "fp": 0})
        esp = sum(1 for d in manifesto["defects"] if d["detector"] == det and d["esperado"])
        if det not in IMPLEMENTADOS:
            print(f"{det:<12}{'-':>4}{'-':>4}{'-':>4}{'-':>11}{'-':>9}   nao implementado "
                  f"({esp} no gabarito)")
            continue
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else float("nan")
        rc = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else float("nan")
        print(f"{det:<12}{v['tp']:>4}{v['fn']:>4}{v['fp']:>4}{p:>10.0%}{rc:>9.0%}   ok")

    impl = {k: v for k, v in r["por_detector"].items() if k in IMPLEMENTADOS}
    TP = sum(v["tp"] for v in impl.values())
    FN = sum(v["fn"] for v in impl.values())
    FP = sum(v["fp"] for v in impl.values())
    print("-" * 62)
    print(f"{'total':<12}{TP:>4}{FN:>4}{FP:>4}"
          f"{TP / (TP + FP) if TP + FP else 0:>10.0%}{TP / (TP + FN) if TP + FN else 0:>9.0%}")
    print(f"\nfalsos positivos por 100 colunas limpas: "
          f"{100 * len(r['fp_limpa']) / r['n_limpas']:.1f}")

    if r["fn"]:
        print("\nNAO ENCONTRADOS")
        for d in r["fn"]:
            marca = "" if d["detector"] in IMPLEMENTADOS else "  (detector nao implementado)"
            print(f"  [{d['detector']}] {d['coluna']:<26}{d['tipo']}{marca}")
    if r["fp_armadilha"]:
        print("\nCAIU NA ARMADILHA")
        for d, f in r["fp_armadilha"]:
            print(f"  {f}")
    if r["fp_limpa"]:
        print("\nFALSO POSITIVO EM COLUNA LIMPA")
        for f in r["fp_limpa"]:
            print(f"  {f}")
    if r["outros"]:
        print("\nACHADOS FORA DO GABARITO (nem defeito conhecido, nem coluna limpa)")
        for f in r["outros"]:
            print(f"  {f}")

    # --- margem: o numero que diz se a precisao foi detector ou sorte ---------
    # Inclui os candidatos REJEITADOS. Sem eles a tabela so mostra o que passou, e a
    # pergunta "por quanto passou" fica sem resposta.
    tipo_de = {(f.coluna, f.valor): f.tipo for f in achados if f.detector == "D1"}
    sentinelas = {k for k, t in tipo_de.items() if t == "sentinela_numerica"}
    # (coluna, valor) exatos que o gabarito cita — inclui a armadilha
    citadas = {(d["coluna"], float(d["valor"])) for d in manifesto["defects"]
               if isinstance(d.get("valor"), (int, float)) and not isinstance(d["valor"], bool)}
    print("\nMARGEM DO TESTE DE GAP  (todo candidato avaliado, aceito ou nao)")
    print(f"  {'coluna':<26}{'valor':>10}{'massa':>8}{'rota':>10}{'razao':>13}"
          f"{'limiar':>9}  veredito")
    # Sempre: os aceitos e qualquer candidato numa coluna citada no gabarito — a
    # armadilha e a linha mais importante da tabela e nao pode ser cortada por empate.
    # Depois, os rejeitados de maior razao, que sao os que chegaram mais perto.
    mostrar = [d for d in diag if d["aceito"] or (d["coluna"], d["valor"]) in citadas]
    vistos = {id(d) for d in mostrar}
    for rota in ("discreta", "continua"):
        rej = sorted((d for d in diag if d["rota"] == rota and id(d) not in vistos),
                     key=lambda d: -d["razao"])
        mostrar += rej[:3]
        if len(rej) > 3:
            print(f"  ... e mais {len(rej) - 3} candidatos rejeitados na rota {rota}")
    armadilhas = {(d["coluna"], float(d["valor"])) for d in manifesto["defects"]
                  if not d["esperado"] and isinstance(d.get("valor"), (int, float))}
    for d in sorted(mostrar, key=lambda d: (d["rota"], -d["razao"])):
        k = (d["coluna"], d["valor"])
        if k in sentinelas:
            verd = "SENTINELA"
        elif k in tipo_de:
            verd = tipo_de[k].replace("_", " ")
        else:
            verd = "rejeitado"
        arm = "  <- ARMADILHA" if k in armadilhas else ""
        print(f"  {d['coluna']:<26}{d['valor']:>10.4g}{d['massa']:>8.2%}{d['rota']:>10}"
              f"{d['razao']:>13,.4g}{d['limiar']:>9g}  {verd}{arm}")

    for nome in ("discreta", "continua"):
        grupo = [d for d in diag if d["rota"] == nome]
        pos = [d["razao"] for d in grupo if (d["coluna"], d["valor"]) in sentinelas]
        neg = [d["razao"] for d in grupo if (d["coluna"], d["valor"]) not in sentinelas]
        if not pos:
            continue
        lim = grupo[0]["limiar"]
        linha = (f"  rota {nome}: positivo mais fraco {min(pos):,.4g}, "
                 f"limiar {lim:g} (folga {min(pos) / lim:,.1f}x)")
        if neg:
            linha += (f", negativo mais forte {max(neg):,.4g} "
                      f"-> separacao de {min(pos) / max(neg):,.0f}x")
        print(linha)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "rows": manifesto["rows"],
        "por_detector": r["por_detector"],
        "tp": [{"defeito": d["id"], "achado": f.to_dict()} for d, f in r["tp"]],
        "fn": [d["id"] for d in r["fn"]],
        "fp_armadilha": [f.to_dict() for _, f in r["fp_armadilha"]],
        "fp_limpa": [f.to_dict() for f in r["fp_limpa"]],
        "outros": [f.to_dict() for f in r["outros"]],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
