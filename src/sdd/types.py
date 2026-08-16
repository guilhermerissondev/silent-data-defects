"""Estruturas de perfil. Tudo aqui e serializavel e, quando faz sentido, mergeavel."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def _plain(v: Any) -> Any:
    """Converte Decimal/date/bytes em algo que o json aceita."""
    import datetime
    import decimal

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return str(v)


@dataclass
class ColumnMeta:
    """Camada 0: sai do footer do Parquet, sem ler uma linha de dado."""

    name: str
    physical_type: str
    logical_type: str | None = None
    rows: int = 0
    null_count: int | None = None
    min: Any = None
    max: Any = None
    distinct_count: int | None = None
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    stats_complete: bool = True

    @property
    def null_rate(self) -> float | None:
        if self.null_count is None or not self.rows:
            return None
        return self.null_count / self.rows

    @property
    def is_constant(self) -> bool:
        return (
            self.min is not None
            and self.max is not None
            and self.min == self.max
            and self.stats_complete
        )

    @property
    def is_all_null(self) -> bool:
        return self.null_count is not None and self.rows > 0 and self.null_count == self.rows


@dataclass
class MetadataScan:
    paths: list[str]
    rows: int
    row_groups: int
    columns: dict[str, ColumnMeta] = field(default_factory=dict)
    file_bytes: int = 0
    footer_bytes: int = 0
    elapsed_s: float = 0.0
    schema_conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["columns"] = {k: _plain(asdict(v)) for k, v in self.columns.items()}
        return _plain(d)


@dataclass
class ColumnStats:
    """Camada 1: uma passada de dados."""

    name: str
    kind: str  # numeric | string | temporal | boolean | other
    sql_type: str
    n: int = 0
    non_null: int = 0
    ndv_approx: int | None = None
    # "exact_histogram" | "exact_distinct" | "hll"  — ver ColumnStats.ndv_exact
    ndv_source: str = "hll"
    min: Any = None
    max: Any = None
    mean: float | None = None
    stddev: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    avg_len: float | None = None
    top_values: list[Any] = field(default_factory=list)
    value_counts: dict[str, int] = field(default_factory=dict)
    # "histogram"  -> value_counts cobre TODOS os valores da coluna (exato e completo)
    # "top_k"      -> cobre so os candidatos do Space-Saving (exato, mas parcial)
    # "none"       -> nao apurado
    counts_source: str = "none"

    @property
    def counts_complete(self) -> bool:
        return self.counts_source == "histogram"

    @property
    def ndv_exact(self) -> bool:
        return self.ndv_source in ("exact_histogram", "exact_distinct")

    @property
    def null_count(self) -> int:
        return self.n - self.non_null

    @property
    def null_rate(self) -> float:
        return (self.n - self.non_null) / self.n if self.n else 0.0

    @property
    def ndv_ratio(self) -> float | None:
        if self.ndv_approx is None or not self.non_null:
            return None
        return self.ndv_approx / self.non_null

    def mass(self, value: Any) -> float:
        """Fracao das linhas nao-nulas ocupada por um valor exato."""
        c = self.value_counts.get(str(value))
        if c is None or not self.non_null:
            return 0.0
        return c / self.non_null


@dataclass
class ScanResult:
    source: str
    rows: int
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    metadata: MetadataScan | None = None
    elapsed_s: float = 0.0
    queries: int = 0
    engine: str = ""
    timing_by_class: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "rows": self.rows,
            "elapsed_s": round(self.elapsed_s, 3),
            "queries": self.queries,
            "timing_by_class": self.timing_by_class,
            "engine": self.engine,
            "columns": {},
        }
        for k, v in self.columns.items():
            cd = asdict(v)
            cd["null_rate"] = v.null_rate
            cd["ndv_ratio"] = v.ndv_ratio
            d["columns"][k] = _plain(cd)
        if self.metadata is not None:
            d["metadata_scan"] = self.metadata.to_dict()
        return d
