"""Calibra o lote por classe de expressao — a medicao que derrubou o desenho inicial.

O instinto e "uma query com tudo, uma passada so". Isso e certo para agregado barato
e errado para agregado pesado com estado. Este script mede as duas familias variando
quantas expressoes vao por query, e imprime a tabela que justifica BATCH_CPU.

    python bench/batching.py data/contas.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sdd.scan import duckdb_pass as dp
from sdd.scan import metadata


def bench(con, src: str, label: str, exprs: list[str], sizes: list[int]) -> None:
    print(f"\n{label}  ({len(exprs)} expressoes)")
    print(f"  {'expr/query':>10}{'queries':>10}{'tempo':>10}")
    best = None
    for per in sizes:
        if per > len(exprs):
            continue
        t = time.perf_counter()
        nq = 0
        for s in range(0, len(exprs), per):
            con.execute(f"SELECT {', '.join(exprs[s:s + per])} FROM {src}").fetchone()
            nq += 1
        el = time.perf_counter() - t
        best = min(best, el) if best else el
        print(f"  {per:>10}{nq:>10}{el:>9.2f}s")
    print(f"  melhor: {best:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/contas.parquet")
    ap.add_argument("--filters", type=int, default=400)
    args = ap.parse_args()

    paths = metadata.resolve(args.path)
    src = dp.source_expr(paths)
    con = dp.connect()
    cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    size_mb = sum(Path(p).stat().st_size for p in paths) / 1e6
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    print(f"{args.path}  {n:,} linhas x {len(cols)} colunas  {size_mb:,.1f} MB")

    kinds = {name: dp.classify(ty) for name, ty in cols}
    numeric = [n_ for n_, k in kinds.items() if k == "numeric"]
    topk_cols = [n_ for n_, k in kinds.items() if k in ("numeric", "string", "temporal")]

    cheap = ([f"count({dp.q(c)})" for c, _ in cols]
             + [f"approx_count_distinct({dp.q(c)})" for c, _ in cols]
             + [f"min({dp.q(c)})" for c, _ in cols]
             + [f"max({dp.q(c)})" for c, _ in cols])
    bench(con, src, "cheap  (count / ndv / min / max)", cheap, [16, 32, 64, 128, 512])

    qs = ", ".join(str(x) for x in dp.DEFAULT_QUANTILES)
    bench(con, src, "quantile  (approx_quantile / t-digest)",
          [f"approx_quantile({dp.q(c)}, [{qs}])" for c in numeric], [1, 2, 4, 8, 32])

    bench(con, src, "topk  (approx_top_k / Space-Saving)",
          [f"approx_top_k({dp.q(c)}, {dp.TOP_K})" for c in topk_cols], [1, 2, 4, 8, 32])

    col = numeric[0]
    bench(con, src, f"filter  (count(*) FILTER sobre {col})",
          [f"count(*) FILTER (WHERE {dp.q(col)} = {-999.0 - i})" for i in range(args.filters)],
          [16, 32, 64, 128, 400])

    print("\nLeitura: agregado barato divide a varredura e deve ser empacotado ao maximo.")
    print("Agregado com estado (Space-Saving, t-digest, cadeia de FILTER) degrada de forma")
    print("super-linear na mesma query e pede lote pequeno. 'Numero de passadas' nao e a")
    print("metrica de custo — classe de expressao e.")


if __name__ == "__main__":
    main()
