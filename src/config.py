"""Typed configuration loaded from config/*.yaml.

Everything the manifest needs to reproduce a run lives here. Nothing reads YAML
directly outside this module.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- run


class Seeds(_Frozen):
    query_subsample: int
    warmup_holdout: int
    plan_shuffle: int
    muvera: int


class QueryConfig(_Frozen):
    # null = no cap; every judged query runs. Settable so smoke tests
    # and quick debugging runs can limit the query count.
    cap: int | None
    warmup_count: int


class ExecutionConfig(_Frozen):
    repeats: int
    final_limit: int
    max_stored_results: int
    flush_every: int
    depths: list[int]
    # colbert_rescore and muvera_colbert issue one extra, untimed Qdrant call per query
    # to measure how many relevant documents survived their prefetch stage, independent
    # of the final rescored ranking. Real added wall-clock cost (roughly doubles those
    # two arms' own query-side work), no effect on any recorded latency.
    capture_pool_recall: bool = True

    @model_validator(mode="after")
    def _check(self) -> ExecutionConfig:
        # Qdrant requires prefetch.limit >= limit + offset. Offset is always
        # 0 here, so the weakest depth must still cover the final limit.
        if min(self.depths) < self.final_limit:
            raise ValueError(
                f"smallest depth {min(self.depths)} < final_limit {self.final_limit}; "
                "prefetch.limit must be >= limit + offset"
            )
        return self


class HnswConfig(_Frozen):
    m: int
    ef_construct: int
    # Segment size (KB) below which Qdrant full-scans instead of using the HNSW graph.
    # Qdrant's default is 10000, which would full-scan NFCorpus and SciFact entirely
    # (~5-8 MB of dense vectors) while the larger datasets use ANN — making cross-dataset
    # latency a comparison of two different algorithms. 10 is the smallest value Qdrant
    # accepts (0 is rejected with "must be 10 or larger") and is small enough that every
    # segment here clears it, so the index is always used.
    full_scan_threshold_kb: int = 10

    @model_validator(mode="after")
    def _threshold_is_accepted_by_qdrant(self) -> "HnswConfig":
        if self.full_scan_threshold_kb < 10:
            raise ValueError(
                f"full_scan_threshold_kb={self.full_scan_threshold_kb} is rejected by "
                "Qdrant, which requires 10 or larger. Use 10 to always use the index."
            )
        return self


class QdrantConfig(_Frozen):
    url: str
    grpc_port: int
    prefer_grpc: bool
    timeout_s: int
    image: str
    container_name: str
    # Docker named volume, not a host bind mount. A bind mount from macOS reaches the
    # container through the VM's virtiofs passthrough, which makes Qdrant refuse
    # multi-mmap ("you may see reduced performance") and warn about corruption risk —
    # both of which land directly on the latency and memory numbers this measures. A
    # named volume lives on the VM's own filesystem and avoids the passthrough.
    storage_volume: str
    hnsw: HnswConfig
    upsert_batch_size: int
    # Segment size (KB) above which Qdrant builds the HNSW index. Indexing is disabled
    # (threshold 0) for the duration of the upload and restored to this afterwards, which
    # is what triggers one clean index build instead of repeatedly indexing partial
    # segments. Qdrant's own default is 20000; a collection left at 0 stays unindexed.
    indexing_threshold_kb: int = 20000
    # Capture Qdrant's self-reported server time. Only the REST transport exposes it
    # (qdrant-client drops it on the gRPC path), so enabling this forces HTTP for the
    # measured queries.
    capture_server_time: bool = True


class IngestionConfig(_Frozen):
    # Documents embedded and upserted together before moving to the next chunk. Bounds
    # peak embedding memory to one chunk regardless of corpus size, independently of
    # embedding/upload batch sizes (model throughput and Qdrant's payload limit).
    chunk_size: int


class ArmsConfig(_Frozen):
    enabled: list[str]
    gated: dict[str, str]


class PathsConfig(_Frozen):
    results: str
    raw: str
    agg: str
    queries: str
    cache: str
    manifest: str

    def resolve(self, root: Path) -> dict[str, Path]:
        return {k: root / v for k, v in self.model_dump().items()}


class RunConfig(_Frozen):
    seeds: Seeds
    queries: QueryConfig
    execution: ExecutionConfig
    qdrant: QdrantConfig
    ingestion: IngestionConfig
    arms: ArmsConfig
    paths: PathsConfig


# ------------------------------------------------------------------------ models


class DenseReference(_Frozen):
    dataset: str
    metric: str
    value: float
    tolerance: float
    source: str
    query_set: str
    verified: bool
    # What this pipeline actually measured against `value`, recorded so the gap between
    # published and reproduced is visible rather than folded into `tolerance`.
    observed: float | None = None


class DenseModel(_Frozen):
    name: str
    dim: int
    distance: Literal["cosine", "dot", "euclid"]
    max_length: int
    reference: DenseReference


class ColbertModel(_Frozen):
    name: str
    dim: int
    distance: Literal["cosine", "dot", "euclid"]
    doc_max_tokens: int
    query_max_tokens: int
    hnsw_m: int
    comparator: Literal["max_sim"]

    @model_validator(mode="after")
    def _m_zero(self) -> ColbertModel:
        # max_sim is not symmetric, so HNSW is unreliable on it.
        if self.hnsw_m != 0:
            raise ValueError("colbert.hnsw_m must be 0; multivectors never prefetch")
        return self


class MuveraModel(_Frozen):
    """Parameters for fastembed.postprocess.Muvera.

    Field names keep this project's own convention (k_sim/n_reps/d_proj); they map
    directly onto fastembed's Muvera(k_sim=, r_reps=, dim_proj=).
    """

    source_tokens: str
    k_sim: int
    n_reps: int
    d_proj: int
    distance: Literal["cosine", "dot"]
    # Set true only after tests/test_muvera_calibration.py passes on real embeddings.
    calibrated: bool = False
    calibration_note: str | None = None

    @property
    def n_partitions(self) -> int:
        return 2**self.k_sim

    @property
    def dim(self) -> int:
        return self.n_reps * self.n_partitions * self.d_proj


class CrossEncoderModel(_Frozen):
    name: str
    size_gb: float
    license: str


class ModelsConfig(_Frozen):
    dense: DenseModel
    colbert: ColbertModel
    muvera: MuveraModel
    cross_encoder: CrossEncoderModel


# ---------------------------------------------------------------------- datasets


class DatasetSpec(_Frozen):
    name: str
    hf_corpus: str | None
    hf_corpus_revision: str | None = None
    hf_qrels: str | None
    hf_qrels_revision: str | None = None
    corpus_size: int | None
    n_queries: int | None
    expected_grades: list[int] | None
    representations: list[str]
    enabled: bool
    text_template: str
    qrels_split: str
    colbert_on_disk_threshold: int
    measure_memory: bool
    # Per-dataset candidate-depth override; null falls back to execution.depths.
    depths: list[int] | None = None
    excluded_arms: dict[str, str] = Field(default_factory=dict)
    disabled_reason: str | None = None
    notes: str | None = None

    @property
    def colbert_on_disk(self) -> bool:
        if self.corpus_size is None:
            return False
        return self.corpus_size >= self.colbert_on_disk_threshold


class DatasetsConfig(_Frozen):
    datasets: dict[str, DatasetSpec]

    def get(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            known = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset {name!r}; known: {known}")
        return self.datasets[name]

    @property
    def enabled_ordered(self) -> list[DatasetSpec]:
        """Enabled datasets, smallest corpus first."""
        specs = [d for d in self.datasets.values() if d.enabled]
        return sorted(specs, key=lambda d: d.corpus_size or 0)


# ------------------------------------------------------------------------ bundle


class Config(_Frozen):
    run: RunConfig
    models: ModelsConfig
    datasets: DatasetsConfig
    root: Path

    def depths_for(self, dataset: str) -> list[int]:
        """Candidate depths for a dataset: its override, else the global sweep."""
        spec = self.datasets.get(dataset)
        return list(spec.depths) if spec.depths else list(self.run.execution.depths)

    def with_depths_override(self, depths: list[int]) -> "Config":
        """Force `depths` everywhere, including over every dataset's own override.

        A plain global-depths update is not enough: `depths_for()` checks a dataset's
        own override FIRST, so a dataset that has one (NFCorpus) would silently ignore
        the global value while every other dataset appeared to honour it.
        """
        cleared = {
            name: spec.model_copy(update={"depths": None}) for name, spec in self.datasets.datasets.items()
        }
        return self.model_copy(
            update={
                "run": self.run.model_copy(
                    update={"execution": self.run.execution.model_copy(update={"depths": list(depths)})}
                ),
                "datasets": self.datasets.model_copy(update={"datasets": cleared}),
            }
        )

    @property
    def paths(self) -> dict[str, Path]:
        return self.run.paths.resolve(self.root)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _build_datasets(raw: dict[str, Any]) -> DatasetsConfig:
    defaults = raw.get("defaults", {})
    out: dict[str, DatasetSpec] = {}
    for name, entry in raw["datasets"].items():
        merged = {**defaults, **entry, "name": name}
        out[name] = DatasetSpec(**merged)
    return DatasetsConfig(datasets=out)


@functools.lru_cache(maxsize=4)
def load_config(config_dir: Path | None = None) -> Config:
    """Load and validate all configuration. Cached; call once per process."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    return Config(
        run=RunConfig(**_read_yaml(cdir / "run.yaml")),
        models=ModelsConfig(**_read_yaml(cdir / "models.yaml")),
        datasets=_build_datasets(_read_yaml(cdir / "datasets.yaml")),
        root=cdir.parent,
    )
