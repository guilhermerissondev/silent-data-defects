"""Roda ydata-profiling sobre um Parquet. Executado em subprocesso pelo comparador."""

import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

path, rows, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

t0 = time.perf_counter()
import pandas as pd

df = pd.read_parquet(path)
if rows and rows < len(df):
    df = df.head(rows)
t_read = time.perf_counter() - t0

from ydata_profiling import ProfileReport

t1 = time.perf_counter()
report = ProfileReport(df, minimal=(mode == "minimal"), progress_bar=False,
                       title="bench", explorative=False)
report.get_description()          # forca o calculo, sem custo de render HTML
t_profile = time.perf_counter() - t1

json.dump({"ok": True, "read_s": round(t_read, 2), "profile_s": round(t_profile, 2),
           "total_s": round(t_read + t_profile, 2)}, open(out, "w"))
