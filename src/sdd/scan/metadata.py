"""Camada 0 — perfil a partir do footer do Parquet, sem ler uma linha de dado.

O footer ja carrega, por row group e por coluna: contagem de linhas, null_count,
min/max estatisticos, distinct_count (quando o writer gravou) e tamanho.
Num arquivo de dezenas de GB isso sai em milissegundos, e ja responde:

  - taxa de nulos exata por coluna
  - faixa de valores, logo sentinela obvia no extremo (min = -999 numa renda)
  - colunas constantes e colunas 100% nulas
  - schema desalinhado entre arquivos do mesmo diretorio

CSV nao tem camada 0. O scan degrada para a camada 1 e o relatorio diz isso.
"""

from __future__ import annotations

import glob as _glob
import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from ..types import ColumnMeta, MetadataScan


def resolve(pattern: str | Path) -> list[str]:
    """Expande glob/diretorio para uma lista de arquivos, ordenada e estavel."""
    p = str(pattern)
    if os.path.isdir(p):
        hits = sorted(_glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
    elif any(ch in p for ch in "*?["):
        hits = sorted(_glob.glob(p, recursive=True))
    else:
        hits = [p] if os.path.exists(p) else []
    return hits


def is_parquet(paths: list[str]) -> bool:
    return bool(paths) and all(p.lower().endswith((".parquet", ".pq")) for p in paths)


def _merge(a, b, op):
    if a is None:
        return b
    if b is None:
        return a
    try:
        return op(a, b)
    except TypeError:  # tipos incomparaveis entre row groups
        return a


def scan_metadata(paths: list[str]) -> MetadataScan:
    t0 = time.perf_counter()
    out = MetadataScan(paths=list(paths), rows=0, row_groups=0)
    first_schema: list[tuple[str, str]] | None = None

    for path in paths:
        pf = pq.ParquetFile(path)
        md = pf.metadata
        out.rows += md.num_rows
        out.row_groups += md.num_row_groups
        out.file_bytes += os.path.getsize(path)
        # o footer e a unica parte do arquivo efetivamente lida
        out.footer_bytes += md.serialized_size

        schema = [(md.schema.column(i).name, str(md.schema.column(i).physical_type))
                  for i in range(md.num_columns)]
        if first_schema is None:
            first_schema = schema
        elif schema != first_schema:
            only_here = {n for n, _ in schema} - {n for n, _ in first_schema}
            only_there = {n for n, _ in first_schema} - {n for n, _ in schema}
            detail = []
            if only_here:
                detail.append(f"so em {Path(path).name}: {sorted(only_here)}")
            if only_there:
                detail.append(f"ausente em {Path(path).name}: {sorted(only_there)}")
            out.schema_conflicts.append("; ".join(detail) or f"ordem/tipo divergem em {Path(path).name}")

        for i in range(md.num_columns):
            col = md.schema.column(i)
            name = col.name
            cm = out.columns.get(name)
            if cm is None:
                cm = ColumnMeta(
                    name=name,
                    physical_type=str(col.physical_type),
                    logical_type=str(col.logical_type) if col.logical_type is not None else None,
                )
                out.columns[name] = cm

            for rg in range(md.num_row_groups):
                cc = md.row_group(rg).column(i)
                cm.rows += cc.num_values
                cm.compressed_bytes += cc.total_compressed_size
                cm.uncompressed_bytes += cc.total_uncompressed_size

                st = cc.statistics
                if st is None:
                    cm.stats_complete = False
                    continue
                if st.has_null_count:
                    cm.null_count = (cm.null_count or 0) + st.null_count
                else:
                    cm.stats_complete = False
                if st.has_min_max:
                    cm.min = _merge(cm.min, st.min, min)
                    cm.max = _merge(cm.max, st.max, max)
                else:
                    cm.stats_complete = False
                if getattr(st, "has_distinct_count", False):
                    cm.distinct_count = (cm.distinct_count or 0) + st.distinct_count

    out.elapsed_s = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Achados que ja saem da camada 0, sem custo de leitura.
# ---------------------------------------------------------------------------

# Constantes que quase nunca sao valor legitimo quando aparecem no extremo.
EXTREME_SENTINELS = {-1, -9, -99, -999, -9999, -99999, -999999, -9999999,
                     -999999999, 9999, 99999, 999999, 9999999, 99999999,
                     999999999, 2147483647, -2147483648}


def metadata_findings(scan: MetadataScan) -> list[dict]:
    """Achados de custo zero. Sao pistas, nao vereditos — a confirmacao exige massa,
    que so a camada 1/2 da. Cada um vira candidato prioritario para o detector D1."""
    out: list[dict] = []

    for c in scan.schema_conflicts:
        out.append({"coluna": "*", "achado": "schema_divergente", "detalhe": c,
                    "camada": 0, "confianca": 1.0})

    for name, cm in scan.columns.items():
        if cm.is_all_null:
            out.append({"coluna": name, "achado": "coluna_100pct_nula",
                        "detalhe": f"{cm.rows:,} linhas, todas nulas",
                        "camada": 0, "confianca": 1.0})
            continue
        if cm.is_constant:
            out.append({"coluna": name, "achado": "coluna_constante",
                        "detalhe": f"min = max = {cm.min!r}",
                        "camada": 0, "confianca": 1.0})
            continue

        nr = cm.null_rate
        if nr is not None and nr > 0.5:
            out.append({"coluna": name, "achado": "nulo_predominante",
                        "detalhe": f"{nr:.1%} de nulos",
                        "camada": 0, "confianca": 1.0})

        for bound, v in (("min", cm.min), ("max", cm.max)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v in EXTREME_SENTINELS or (float(v).is_integer() and int(v) in EXTREME_SENTINELS):
                out.append({
                    "coluna": name, "achado": "sentinela_no_extremo",
                    "detalhe": f"{bound} = {v!r} sem ler linha; massa a confirmar na camada 1",
                    "camada": 0, "confianca": 0.6, "candidato": v,
                })
    return out
