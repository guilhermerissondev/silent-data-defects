"""Quanto custa a aproximacao — erro dos sketches contra o valor exato.

Velocidade sem erro medido nao vale nada: o sdd troca exatidao por memoria constante
em tres pontos (cardinalidade via HyperLogLog, quantis via t-digest, valores frequentes
via Space-Saving) e precisa dizer quanto isso custa. Aqui o mesmo DuckDB calcula a
versao exata — cara, mas viavel nesta escala — e comparamos.

Nulos, min, max, media e desvio NAO sao aproximados: sao exatos por construcao.

    python bench/accuracy.py data/contas.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sdd.scan import duckdb_pass as dp
from sdd.scan import metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/contas.parquet")
    ap.add_argument("--out", default="reports/accuracy.json")
    args = ap.parse_args()

    paths = metadata.resolve(args.path)
    # exact_ndv=False de proposito: aqui queremos medir o erro do HyperLogLog puro,
    # que e o que sobra nas colunas largas demais para o count(DISTINCT) exato.
    res = dp.scan(paths, exact_ndv=False)
    con = dp.connect()
    src = dp.source_expr(paths)

    print(f"{args.path}  {res.rows:,} linhas | perfil aproximado em {res.elapsed_s:.2f}s\n")

    # --- cardinalidade: HyperLogLog x count(DISTINCT) exato --------------------
    cols = [c for c in res.columns.values() if c.ndv_approx is not None and c.kind != "boolean"]
    t = time.perf_counter()
    exprs = [f"count(DISTINCT {dp.q(c.name)})" for c in cols]
    exact_ndv = []
    for s in range(0, len(exprs), 4):
        exact_ndv += list(con.execute(f"SELECT {', '.join(exprs[s:s+4])} FROM {src}").fetchone())
    t_exact_ndv = time.perf_counter() - t

    print(f"CARDINALIDADE  HyperLogLog x count(DISTINCT) exato ({t_exact_ndv:.1f}s para o exato)")
    print(f"  {'coluna':<28}{'aproximado':>14}{'exato':>14}{'erro rel.':>12}")
    ndv_rows, worst = [], 0.0
    for c, ex in zip(cols, exact_ndv):
        err = abs(c.ndv_approx - ex) / ex if ex else 0.0
        worst = max(worst, err)
        ndv_rows.append({"coluna": c.name, "aprox": c.ndv_approx, "exato": ex, "erro": err})
        if err > 0.005:
            print(f"  {c.name:<28}{c.ndv_approx:>14,}{ex:>14,}{err:>11.2%}")
    print(f"  erro relativo maximo: {worst:.2%} | mediano: "
          f"{sorted(r['erro'] for r in ndv_rows)[len(ndv_rows)//2]:.3%}")

    # --- quantis: t-digest x quantile_cont exato ------------------------------
    num = [c for c in res.columns.values() if c.kind == "numeric" and c.quantiles]
    probs = [float(k[1:]) for k in num[0].quantiles] if num else []
    t = time.perf_counter()
    exact_q = {}
    for c in num:
        qs = ", ".join(str(p) for p in probs)
        exact_q[c.name] = con.execute(
            f"SELECT quantile_cont({dp.q(c.name)}, [{qs}]) FROM {src}").fetchone()[0]
    t_exact_q = time.perf_counter() - t

    print(f"\nQUANTIS  t-digest x quantile_cont exato ({t_exact_q:.1f}s para o exato)")
    print(f"  {'coluna':<28}{'quantil':>10}{'aproximado':>16}{'exato':>16}{'erro rel.':>12}")
    q_rows, worst_q = [], (0.0, "", "")
    for c in num:
        span = (c.max - c.min) or 1
        for (k, av), ev in zip(c.quantiles.items(), exact_q[c.name]):
            av, ev = float(av), float(ev)
            # erro normalizado pela amplitude da coluna: comparar em unidades da
            # propria variavel esconde que 100 de erro em renda e ruido e em idade e tudo
            err = abs(av - ev) / span
            q_rows.append({"coluna": c.name, "q": k, "aprox": av, "exato": ev, "erro_amp": err})
            if err > worst_q[0]:
                worst_q = (err, c.name, k)
            if err > 0.01:
                print(f"  {c.name:<28}{k:>10}{av:>16,.4g}{ev:>16,.4g}{err:>11.2%}")
    print(f"  erro maximo (fracao da amplitude): {worst_q[0]:.3%} em {worst_q[1]}.{worst_q[2]}")
    print(f"  mediano: {sorted(r['erro_amp'] for r in q_rows)[len(q_rows)//2]:.4%}")

    # --- contagem de valores: exata onde importa ------------------------------
    hist = [c for c in res.columns.values() if c.counts_source == "histogram"]
    topk = [c for c in res.columns.values() if c.counts_source == "top_k"]
    print(f"\nCONTAGEM DE VALORES  {len(hist)} colunas com histogram() exato e completo, "
          f"{len(topk)} com top-k")
    print("  Nas colunas de top-k a contagem de cada candidato tambem e exata (count FILTER);")
    print("  o que e aproximado e QUAIS valores viraram candidatos, nao a massa deles.")

    print(f"\nCUSTO DO EXATO  ndv {t_exact_ndv:.1f}s + quantis {t_exact_q:.1f}s = "
          f"{t_exact_ndv + t_exact_q:.1f}s, contra {res.elapsed_s:.1f}s do perfil inteiro "
          f"({(t_exact_ndv + t_exact_q) / res.elapsed_s:.1f}x)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": res.rows, "ndv": ndv_rows, "quantis": q_rows,
         "t_exact_ndv_s": t_exact_ndv, "t_exact_quantile_s": t_exact_q,
         "t_sdd_s": res.elapsed_s}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
