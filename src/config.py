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
    # late_interaction_rescore and muvera_late_interaction_rescore issue one extra, untimed Qdrant call per query
    # to measure how many relevant documents survived their prefetch stage, independent
    # of the final rescored ranking. Real added wall-clock cost (roughly doubles those
    # two arms' own query-side work), no effect on any recorded latency.
    capture_pool_recall: bool = True
    # A cross-encoder scores each (query, document) pair independently of the pool, so
    # within one query the depth sweep and the two cross-encoder arms keep re-meeting
    # the same pairs. With sharing on, a pair is scored once per query and its measured
    # cost is attributed to every unit that uses it; rankings are identical and
    # `t_rerank_ms` stays a measured per-pair figure (throughput is depth-independent:
    # 28.6-29.1 pairs/s at depths 10-500). False restores fully isolated executions.
    share_pointwise_scores: bool = True
    # Per-arm ceiling on the depth sweep, by arm name. An arm listed here runs the
    # dataset's ladder truncated at its cap rather than a ladder of its own, so every
    # depth it reports is a depth the other arms also report and the paired
    # significance tests stay aligned. Exists because a client-side cross-encoder costs
    # ~500x per candidate what a server-side rescore does, so one shared ladder deep
    # enough for MUVERA is unaffordable for it.
    arm_depth_caps: dict[str, int] = {}

    @model_validator(mode="after")
    def _check(self) -> ExecutionConfig:
        # Qdrant requires prefetch.limit >= limit + offset. Offset is always
        # 0 here, so the weakest depth must still cover the final limit.
        if min(self.depths) < self.final_limit:
            raise ValueError(
                f"smallest depth {min(self.depths)} < final_limit {self.final_limit}; "
                "prefetch.limit must be >= limit + offset"
            )
        for arm, cap in self.arm_depth_caps.items():
            if cap < self.final_limit:
                raise ValueError(
                    f"arm_depth_caps[{arm}] = {cap} < final_limit {self.final_limit}; "
                    "a capped arm must still be able to return the final ranking"
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


class IngestionConfig(_Frozen):
    # Documents embedded and upserted together before moving to the next chunk. Bounds
    # peak embedding memory to one chunk regardless of corpus size, independently of
    # embedding/upload batch sizes (model throughput and Qdrant's payload limit).
    chunk_size: int
    # Worker processes for MUVERA document encoding, which is NumPy-light per document
    # and otherwise serial. 0 = one per CPU, scaled down for small corpora.
    muvera_workers: int = 0


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


class QualityReference(_Frozen):
    """A published score a baseline must reproduce before any number downstream counts."""

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


class OnnxSource(_Frozen):
    """An ONNX file FastEmbed does not ship, registered with FastEmbed as a custom model."""

    hf: str
    file: str
    pooling: Literal["cls", "mean"]
    normalize: bool


class DenseModel(_Frozen):
    name: str
    dim: int
    distance: Literal["cosine", "dot", "euclid"]
    max_length: int
    reference: QualityReference
    onnx: OnnxSource | None = None

    @property
    def model_id(self) -> str:
        """The model's FastEmbed name, and its identity in cache keys and manifests.

        A custom ONNX source gets its own name, since FastEmbed refuses to register a
        second file under a model it already ships, and a different file must never
        share cached embeddings with it.
        """
        if self.onnx is None:
            return self.name
        return f"{self.name} ({self.onnx.hf}:{self.onnx.file})"


class Bm25Model(_Frozen):
    """BM25 scored by Qdrant on a full-text payload index (Qdrant 1.20+).

    Qdrant tokenizes documents and queries with the index's own settings and derives
    every corpus statistic (document count, IDF, average length) from the indexed data,
    so nothing is computed client-side.
    """

    # The payload key every point stores its document text under (see indexing._point).
    field: Literal["text"]
    tokenizer: Literal["word"]
    lowercase: bool
    stemmer: Literal["english"]
    stopwords: Literal["english"]
    k: float
    b: float
    reference: QualityReference


class LateInteractionModel(_Frozen):
    name: str
    dim: int
    distance: Literal["cosine", "dot", "euclid"]
    hnsw_m: int
    comparator: Literal["max_sim"]

    @model_validator(mode="after")
    def _m_zero(self) -> LateInteractionModel:
        # max_sim is not symmetric, so HNSW is unreliable on it.
        if self.hnsw_m != 0:
            raise ValueError("late_interaction.hnsw_m must be 0; multivectors never prefetch")
        return self


class MuveraTuningGrid(_Frozen):
    """Settings `tune-muvera` sweeps, declared before any result is seen."""

    k_sim: list[int]
    d_proj: list[int]
    n_reps: list[int]
    datasets: list[str]


class MuveraModel(_Frozen):
    """Parameters for fastembed.postprocess.Muvera.

    Field names keep this project's own convention (k_sim/n_reps/d_proj); they map
    directly onto fastembed's Muvera(k_sim=, r_reps=, dim_proj=).
    """

    source_tokens: str
    # Subtract the corpus mean token vector before FDE encoding, on both sides.
    # Late-interaction token embeddings are anisotropic -- they collapse into a narrow
    # cone -- and SimHash partitions a cone poorly, since almost every random hyperplane
    # falls to one side of it. Centering moves the cloud onto the origin, where the
    # partitions actually separate documents. Affects the `muvera` vector only: the
    # stored multivectors the rescoring stage reads are never centered.
    center: bool = True
    k_sim: int
    n_reps: int
    d_proj: int
    distance: Literal["cosine", "dot"]
    # Set true only after tests/test_muvera_calibration.py passes on real embeddings.
    calibrated: bool = False
    # The late-interaction model whose tokens the calibration used. The FDE's quality
    # depends on the token distribution, so a calibration never carries over to another
    # model.
    calibrated_model: str | None = None
    calibration_note: str | None = None
    # Fraction of the exact max_sim top-k the FDE prefetch must recover at
    # `calibration_depth`. A project convention, not a published standard.
    min_recall: float = 0.8
    # The candidate depth min_recall is read at. Fixed rather than "the deepest depth
    # swept", so the bar stays where the FDE setting was selected: extending the depth
    # ladder must not move the threshold a recorded pass/fail verdict was measured
    # against. Must be a depth the calibration artifact actually contains.
    calibration_depth: int = 500
    # False runs muvera_late_interaction_rescore wherever its calibration fails, instead
    # of excluding the arm, and records the verdict per dataset in the manifest. The FDE
    # setting is chosen once on the tuning datasets, so a per-dataset failure is a result
    # to report, not a reason to leave a gap in the comparison.
    require_calibration: bool = True
    tuning_grid: MuveraTuningGrid | None = None
    # Set only by `run --allow-uncalibrated-muvera`, for measuring the arm end to end
    # before a setting is chosen. Recorded in the manifest with the rest of this block.
    allow_uncalibrated: bool = False

    @property
    def n_partitions(self) -> int:
        return 2**self.k_sim

    @property
    def dim(self) -> int:
        return self.n_reps * self.n_partitions * self.d_proj


class RerankerOnnxSource(_Frozen):
    """An ONNX cross-encoder FastEmbed does not ship, registered as a custom model.

    The file must emit a single score per (query, document) pair; a backbone-only export
    that emits hidden states cannot be scored and is rejected when the model loads.
    """

    hf: str
    file: str


class CrossEncoderModel(_Frozen):
    name: str
    size_gb: float
    license: str
    onnx: RerankerOnnxSource | None = None

    @property
    def model_id(self) -> str:
        """The model's FastEmbed name, and its identity in digests and manifests.

        A custom ONNX source gets its own name: FastEmbed refuses to register a second
        file under a model it already ships, and two files must never be confused in a
        recorded result.
        """
        if self.onnx is None:
            return self.name
        return f"{self.name} ({self.onnx.hf}:{self.onnx.file})"


class ModelsConfig(_Frozen):
    dense: DenseModel
    bm25: Bm25Model
    late_interaction: LateInteractionModel
    muvera: MuveraModel
    cross_encoder: CrossEncoderModel

    @model_validator(mode="after")
    def _calibration_matches_model(self) -> ModelsConfig:
        if self.muvera.calibrated and self.muvera.calibrated_model != self.late_interaction.name:
            raise ValueError(
                f"muvera.calibrated is true for {self.muvera.calibrated_model!r}, but the "
                f"late-interaction model is {self.late_interaction.name!r}. Recalibrate with "
                "`calibrate-muvera` and set calibrated_model, or set calibrated: false."
            )
        return self

    @property
    def muvera_calibrated(self) -> bool:
        """The config-level override, valid only for the model it was calibrated on."""
        return self.muvera.calibrated and self.muvera.calibrated_model == self.late_interaction.name


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
    late_interaction_on_disk_threshold: int
    measure_memory: bool
    # Per-dataset candidate-depth override; null falls back to execution.depths.
    depths: list[int] | None = None
    # Per-dataset query cap; null runs every judged query. Takes precedence over the
    # global queries.cap, which exists only so smoke tests can limit every dataset.
    query_cap: int | None = None
    excluded_arms: dict[str, str] = Field(default_factory=dict)
    disabled_reason: str | None = None
    notes: str | None = None

    @property
    def late_interaction_on_disk(self) -> bool:
        if self.corpus_size is None:
            return False
        return self.corpus_size >= self.late_interaction_on_disk_threshold


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

    def depths_for_arm(self, dataset: str | None, arm: str) -> list[int]:
        """The dataset's ladder, truncated at this arm's cap when it has one.

        Truncation rather than a separate ladder: a capped arm's depths stay a subset
        of every other arm's, so arms remain comparable at every depth they share.
        """
        depths = self.depths_for(dataset) if dataset else list(self.run.execution.depths)
        cap = self.run.execution.arm_depth_caps.get(arm)
        return depths if cap is None else [d for d in depths if d <= cap]

    def with_depths_override(self, depths: list[int]) -> "Config":
        """Force `depths` everywhere, including over every dataset's own override.

        A plain global-depths update is not enough: `depths_for()` checks a dataset's
        own override FIRST, so a dataset that has one would silently ignore the global
        value while every other dataset appeared to honour it. Per-arm caps are dropped
        for the same reason — an explicit --depths asks for exactly these depths on
        every arm, which is what makes a deep diagnostic run on the capped arms possible.
        """
        cleared = {
            name: spec.model_copy(update={"depths": None}) for name, spec in self.datasets.datasets.items()
        }
        return self.model_copy(
            update={
                "run": self.run.model_copy(
                    update={
                        "execution": self.run.execution.model_copy(
                            update={"depths": list(depths), "arm_depth_caps": {}}
                        )
                    }
                ),
                "datasets": self.datasets.model_copy(update={"datasets": cleared}),
            }
        )

    def with_results_dir(self, results: str) -> "Config":
        """Write every run output (raw, agg, queries, manifest) under `results` instead.

        Keeps a new run from touching a recorded one. The embedding cache stays where it
        is: it is keyed by model, corpus and platform, so sharing it is always safe.
        """
        current = self.run.paths
        base = Path(current.results)

        def moved(path: str) -> str:
            return str(Path(results) / Path(path).relative_to(base))

        paths = current.model_copy(
            update={
                "results": results,
                **{key: moved(getattr(current, key)) for key in ("raw", "agg", "queries", "manifest")},
            }
        )
        return self.model_copy(update={"run": self.run.model_copy(update={"paths": paths})})

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
