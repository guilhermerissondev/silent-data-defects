"""Roda o sdd sobre um Parquet. Executado em subprocesso pelo comparador."""

import json
import sys
import time

path, rows, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

t0 = time.perf_counter()
from sdd.scan import duckdb_pass, metadata

paths = metadata.resolve(path)
meta = metadata.scan_metadata(paths)
t_meta = time.perf_counter() - t0

t1 = time.perf_counter()
if mode != "metadata":
    res = duckdb_pass.scan(paths, value_counts=(mode != "minimal"),
                           limit=rows or None)
    queries = res.queries
else:
    queries = 0
t_profile = time.perf_counter() - t1

json.dump({"ok": True, "read_s": round(t_meta, 3), "profile_s": round(t_profile, 2),
           "total_s": round(t_meta + t_profile, 2), "queries": queries}, open(out, "w"))
