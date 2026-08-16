"""Compara sdd e ydata-profiling na mesma base, medindo tempo e pico de memoria.

Os dois rodam em subprocesso (venvs diferentes: ydata-profiling nao instala em
Python 3.14), com o pico de RSS amostrado pelo pai. Nenhum dos dois renderiza HTML —
so o calculo do perfil — e o ydata roda em `minimal=True`, que desliga a matriz de
correlacao. E a configuracao que favorece o ydata: sem isso o custo e O(n*k^2).

    python bench/compare_ydata.py --sizes 100000 1000000 5000000

Requer psutil no venv que executa este script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY_SDD = ROOT / ".venv" / "Scripts" / "python.exe"
PY_YDATA = ROOT / ".venv-ydata" / "Scripts" / "python.exe"


def run(py: Path, script: str, path: str, rows: int, mode: str, timeout: int) -> dict:
    out = HERE / "_result.json"
    out.unlink(missing_ok=True)
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")}
    t0 = time.perf_counter()
    proc = subprocess.Popen([str(py), str(HERE / script), path, str(rows), mode, str(out)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    peak = 0
    try:
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
                for ch in ps.children(recursive=True):
                    rss += ch.memory_info().rss
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.perf_counter() - t0 > timeout:
                proc.kill()
                return {"ok": False, "erro": f"timeout > {timeout}s",
                        "total_s": None, "peak_mb": peak / 1e6}
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()

    wall = time.perf_counter() - t0
    if proc.returncode != 0 or not out.exists():
        err = proc.stderr.read().decode("utf-8", "replace").strip().splitlines()
        motivo = err[-1] if err else f"exit {proc.returncode}"
        return {"ok": False, "erro": motivo[:120], "total_s": round(wall, 2),
                "peak_mb": peak / 1e6}
    res = json.loads(out.read_text())
    res["wall_s"] = round(wall, 2)
    res["peak_mb"] = round(peak / 1e6, 1)
    out.unlink(missing_ok=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/contas.parquet")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100_000, 500_000, 1_000_000, 5_000_000])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="reports/bench_ydata.json")
    args = ap.parse_args()

    rows = []
    for n in args.sizes:
        print(f"\n=== {n:,} linhas " + "=" * 40)
        for label, py, script, mode in [
            ("sdd camada 0", PY_SDD, "_run_sdd.py", "metadata"),
            ("sdd completo", PY_SDD, "_run_sdd.py", "full"),
            ("ydata minimal", PY_YDATA, "_run_ydata.py", "minimal"),
        ]:
            r = run(py, script, args.path, n, mode, args.timeout)
            r.update(ferramenta=label, linhas=n)
            rows.append(r)
            if r.get("ok"):
                print(f"  {label:<16}{r['wall_s']:>8.2f}s   pico {r['peak_mb']:>8,.0f} MB")
            else:
                print(f"  {label:<16}{'FALHOU':>9}   pico {r['peak_mb']:>8,.0f} MB   {r['erro']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'linhas':>10}  {'sdd cam.0':>10}{'sdd full':>11}{'ydata min':>12}{'sdd pico':>11}{'ydata pico':>12}")
    for n in args.sizes:
        g = {r["ferramenta"]: r for r in rows if r["linhas"] == n}
        def f(k, field, unit=""):
            r = g.get(k)
            if not r or not r.get("ok"):
                return f"{'--':>10}"
            return f"{r[field]:>9,.2f}{unit}" if field == "wall_s" else f"{r[field]:>9,.0f}{unit}"
        print(f"{n:>10,}  {f('sdd camada 0','wall_s','s')}{f('sdd completo','wall_s','s')}"
              f"{f('ydata minimal','wall_s','s')}{f('sdd completo','peak_mb','M')}"
              f"{f('ydata minimal','peak_mb','M')}")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
