"""Arm registry: name -> callable with a uniform signature.

Two first stages, each a baseline arm of its own: `dense` (bge-small HNSW search) and
`bm25` (sparse BM25 search). Every other arm reranks the candidates of one of them, is
named after it (`late_interaction_rescore` vs `bm25_late_interaction_rescore`), and
declares it as its `baseline`, the arm its quality gain is measured against.
`muvera_late_interaction_rescore` generates its own candidates and is measured against
`dense`.

"Late interaction" names the mechanism, not a model: any multivector model scored with
MaxSim (ColBERT-family or otherwise) is configured under `late_interaction` in
config/models.yaml.

Adding an arm requires only a new `@register` block; the runner reads the registry and
never names an arm directly. All arms query the dataset's single collection, so every
prefetch reads the same index.

Out of scope: hybrid-search/fusion arms (RRF, DBSF, weighted RRF), formula/boost
rescoring, and sharding. Nothing here fuses rank lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client import grpc as qgrpc
from qdrant_client.conversions.conversion import GrpcToRest, RestToGrpc

from .config import Config
from .datasets import IdMap
from .embedding import Encoders
from .timing import QueryTimers

Retrieved = list[tuple[str, float]]


class ArmError(RuntimeError):
    pass


@dataclass
class ArmContext:
    client: QdrantClient
    cfg: Config
    encoders: Encoders
    id_map: IdMap
    collection: str
    dataset: str


@dataclass
class ArmOutput:
    retrieved: Retrieved
    timers: QueryTimers
    # Candidate doc ids the arm considered before its final cutoff, when that pool is
    # distinct from `retrieved` and cheap to expose. Recall against this pool bounds
    # what the rescoring stage could possibly have recovered, independent of how well
    # it ranked what it got. None where an arm has no separate pool (`dense`).
    pool_doc_ids: list[str] | None = None


@dataclass(frozen=True)
class ArmSpec:
    name: str
    fn: Callable[[ArmContext, str, int | None], ArmOutput]
    sweeps_depth: bool
    requires: tuple[str, ...]
    group: str
    # The baseline arm this arm's quality gain is measured against; itself for a baseline.
    baseline: str
    client_side_rerank: bool = False
    license_note: str | None = None

    def __call__(self, ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
        return self.fn(ctx, query, depth)


REGISTRY: dict[str, ArmSpec] = {}


def register(
    name: str,
    *,
    sweeps_depth: bool,
    requires: Sequence[str],
    group: str,
    baseline: str,
    client_side_rerank: bool = False,
    license_note: str | None = None,
):
    def decorator(fn):
        REGISTRY[name] = ArmSpec(
            name=name,
            fn=fn,
            sweeps_depth=sweeps_depth,
            requires=tuple(requires),
            group=group,
            baseline=baseline,
            client_side_rerank=client_side_rerank,
            license_note=license_note,
        )
        return fn

    return decorator


# ----------------------------------------------------------------------- helpers


def _final_limit(ctx: ArmContext, depth: int | None) -> int:
    """Stored result count: min(max_stored_results, depth).

    Records keep more than the top-10 the quality metrics read, so a later analysis can
    look deeper without re-running; an arm can never return more than its own depth.
    """
    cap = ctx.cfg.run.execution.max_stored_results
    return cap if depth is None else min(cap, depth)


def _guard_prefetch(depth: int, limit: int) -> None:
    # Qdrant requires prefetch.limit >= limit + offset. Offset is always 0.
    if depth < limit:
        raise ArmError(f"prefetch limit {depth} < final limit {limit}; would truncate results")


FIRST_STAGES = ("dense", "bm25")


def first_stage_query(ctx: ArmContext, timers: QueryTimers, first_stage: str, query: str):
    """The (query, using) pair Qdrant searches a first stage with.

    `dense` embeds the query client-side, timed into its own embed bucket, and searches
    the `dense` vector. `bm25` sends the raw query text to the BM25-scored full-text
    index on the document-text payload field; Qdrant tokenizes it server-side, so there
    is nothing to time client-side. Either way the pair is identical to what that
    stage's baseline arm sends, so an arm's difference from its baseline is its
    reranking alone.
    """
    if first_stage == "dense":
        with timers.embed("dense"):
            return ctx.encoders.embed_query_dense(query).tolist(), "dense"
    if first_stage == "bm25":
        bm25 = ctx.cfg.models.bm25
        text_query = models.TextQuery(
            text=models.TextQueryInput(
                query=query,
                scoring=models.Bm25Scoring(bm25=models.Bm25Params(k=bm25.k, b=bm25.b)),
            )
        )
        return text_query, bm25.field
    raise ArmError(f"unknown first stage {first_stage!r}; expected one of {FIRST_STAGES}")


def _pool_doc_ids(ctx: ArmContext, *, query, using: str, depth: int) -> list[str] | None:
    """Re-run a prefetch alone, to see what it handed the reranker.

    Qdrant's single-request prefetch-then-rescore never returns the prefetch's own
    ranking — only the final, rescored one — so measuring it costs a second request.
    That request is real Qdrant work and takes real wall-clock time, but is timed into
    a throwaway `QueryTimers`, never the query's own, so it cannot inflate any recorded
    latency. Controlled by `execution.capture_pool_recall`, since it roughly doubles
    the calling arm's own query-side work.
    """
    if not ctx.cfg.run.execution.capture_pool_recall:
        return None
    scratch = QueryTimers()
    points = run_query(
        ctx, scratch, map_ids=False,
        query=query, using=using, limit=depth, with_payload=["doc_id"],
    )
    return [ctx.id_map.doc(p.id) for p in points]


def _to_retrieved(points, id_map: IdMap) -> Retrieved:
    out: Retrieved = []
    for point in points:
        doc_id = (point.payload or {}).get("doc_id")
        # Fall back to the ID map rather than trusting the payload alone: a mismatch
        # between the two is exactly the silent-wrong-metrics failure to guard against.
        mapped = id_map.doc(point.id)
        if doc_id is not None and doc_id != mapped:
            raise ArmError(
                f"payload doc_id {doc_id!r} disagrees with id map {mapped!r} for point {point.id}"
            )
        out.append((mapped, float(point.score)))
    return out


def _query_message(
    ctx: ArmContext,
    *,
    query,
    using: str,
    limit: int,
    with_payload: Sequence[str],
    prefetch: models.Prefetch | Sequence[models.Prefetch] | None = None,
    params: models.SearchParams | None = None,
) -> qgrpc.QueryPoints:
    """Build the gRPC request from an arm's REST-flavoured arguments.

    Arms express intent in the documented REST vocabulary (`models.Prefetch`, plain
    vectors, a list of payload keys); this is the single place that translates it. Every
    supported argument is named explicitly rather than forwarded as **kwargs, so an
    argument this function does not understand is a TypeError at the call site instead of
    a silently dropped search parameter.

    A bare vector has to be wrapped in `NearestQuery` before conversion — the converter
    rejects a raw list. A `TextQuery` (BM25) is already a complete query, and `Prefetch`
    resolves its own inner query, so neither needs wrapping.
    """
    if isinstance(query, models.TextQuery):
        top_query = query
    else:
        top_query = models.NearestQuery(nearest=query)
    prefetches = None
    if prefetch is not None:
        listed = prefetch if isinstance(prefetch, (list, tuple)) else [prefetch]
        prefetches = [RestToGrpc.convert_prefetch_query(p) for p in listed]
    return qgrpc.QueryPoints(
        collection_name=ctx.collection,
        query=RestToGrpc.convert_query(top_query),
        prefetch=prefetches,
        using=using,
        limit=limit,
        with_payload=RestToGrpc.convert_with_payload_interface(list(with_payload)),
        params=RestToGrpc.convert_search_params(params) if params is not None else None,
    )


def run_query(ctx: ArmContext, timers: QueryTimers, *, map_ids: bool = True, **kwargs):
    """Issue one Qdrant query over gRPC, timing the round trip and Qdrant's own time.

    Every arm goes through here so the two timers can never drift apart. gRPC carries
    Qdrant's self-reported processing time in the response (`QueryResponse.time`), which
    is what makes the server/transport split measurable without paying JSON's cost —
    `QdrantClient.query_points()` drops that field, so the stub is called directly.

    `map_ids=True` (the default) converts the response to `Retrieved` (doc_id, score)
    INSIDE the same timer block as the round trip, so response deserialisation and
    ID-map lookup are charged to `t_qdrant_ms` rather than falling into an untimed gap
    between this call returning and the caller processing it. Arms that need the raw
    points (`cross_encoder`, for the `text` payload) pass `map_ids=False` and time their
    own post-processing separately.
    """
    request = _query_message(ctx, **kwargs)
    with timers.qdrant():
        response = ctx.client.grpc_points.Query(
            request, timeout=ctx.cfg.run.qdrant.timeout_s
        )
        timers.add_server_time(response.time)
        points = [GrpcToRest.convert_scored_point(hit) for hit in response.result]
        return _to_retrieved(points, ctx.id_map) if map_ids else points


# ---------------------------------------------------------------------- baselines


def _first_stage_search(ctx: ArmContext, query: str, depth: int | None, *, first_stage: str) -> ArmOutput:
    """A first stage on its own, with no reranking: the baseline for its family."""
    timers = QueryTimers()
    limit = _final_limit(ctx, depth)
    first_stage_vec, using = first_stage_query(ctx, timers, first_stage, query)
    retrieved = run_query(
        ctx,
        timers,
        query=first_stage_vec,
        using=using,
        limit=limit,
        with_payload=["doc_id"],
    )
    return ArmOutput(retrieved, timers)


@register("dense", sweeps_depth=False, requires=["dense"], group="baseline", baseline="dense")
def dense_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """Dense baseline: a single full-precision HNSW search, no reranking stage."""
    return _first_stage_search(ctx, query, depth, first_stage="dense")


@register("bm25", sweeps_depth=False, requires=["bm25"], group="baseline", baseline="bm25")
def bm25_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """Sparse baseline: a single BM25 search, scored by Qdrant on the full-text index
    over the document text, no reranking stage."""
    return _first_stage_search(ctx, query, depth, first_stage="bm25")


# --------------------------------------------------------------------- rescoring
#
# Every arm below reranks the candidates of one first stage, or
# (`muvera_late_interaction_rescore`) generates its own. Fusion arms are out of scope because they combine retrievers rather
# than rescore a candidate set; so are sharding and formula-rescore machinery.


def _late_interaction_rescore(ctx: ArmContext, query: str, depth: int | None, *, first_stage: str) -> ArmOutput:
    """`first_stage` prefetch, then late-interaction max_sim rescoring inside Qdrant."""
    if depth is None:
        raise ArmError("late-interaction rescoring sweeps depth; depth must be set")
    timers = QueryTimers()
    limit = _final_limit(ctx, depth)
    _guard_prefetch(depth, limit)

    first_stage_vec, using = first_stage_query(ctx, timers, first_stage, query)
    with timers.embed("late_interaction"):
        late_interaction_tokens_list = ctx.encoders.embed_query_late_interaction(query).tolist()
    retrieved = run_query(
        ctx,
        timers,
        prefetch=[models.Prefetch(query=first_stage_vec, using=using, limit=depth)],
        query=late_interaction_tokens_list,
        using="late_interaction",
        limit=limit,
        with_payload=["doc_id"],
    )
    pool = _pool_doc_ids(ctx, query=first_stage_vec, using=using, depth=depth)
    return ArmOutput(retrieved, timers, pool_doc_ids=pool)


@register(
    "late_interaction_rescore",
    sweeps_depth=True,
    requires=["dense", "late_interaction"],
    group="rescoring",
    baseline="dense",
)
def late_interaction_rescore_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """Full-precision dense prefetch, then late-interaction max_sim rescoring inside Qdrant."""
    return _late_interaction_rescore(ctx, query, depth, first_stage="dense")


@register(
    "bm25_late_interaction_rescore",
    sweeps_depth=True,
    requires=["bm25", "late_interaction"],
    group="rescoring",
    baseline="bm25",
)
def bm25_late_interaction_rescore_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """BM25 prefetch, then late-interaction max_sim rescoring inside Qdrant."""
    return _late_interaction_rescore(ctx, query, depth, first_stage="bm25")


@register(
    "muvera_late_interaction_rescore",
    sweeps_depth=True,
    requires=["muvera", "late_interaction"],
    group="rescoring",
    baseline="dense",
)
def muvera_late_interaction_rescore_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """MUVERA FDE prefetch, then exact late-interaction max_sim rescoring inside Qdrant."""
    if depth is None:
        raise ArmError("muvera_late_interaction_rescore sweeps depth; depth must be set")
    timers = QueryTimers()
    limit = _final_limit(ctx, depth)
    _guard_prefetch(depth, limit)

    with timers.embed("late_interaction"):
        late_interaction_tokens = ctx.encoders.embed_query_late_interaction(query)
        late_interaction_tokens_list = late_interaction_tokens.tolist()
    with timers.embed("muvera"):
        muvera_vec_list = ctx.encoders.muvera.process_query(late_interaction_tokens).tolist()
    retrieved = run_query(
        ctx,
        timers,
        prefetch=[models.Prefetch(query=muvera_vec_list, using="muvera", limit=depth)],
        query=late_interaction_tokens_list,
        using="late_interaction",
        limit=limit,
        with_payload=["doc_id"],
    )
    pool = _pool_doc_ids(ctx, query=muvera_vec_list, using="muvera", depth=depth)
    return ArmOutput(retrieved, timers, pool_doc_ids=pool)


def _cross_encoder(ctx: ArmContext, query: str, depth: int | None, *, first_stage: str) -> ArmOutput:
    """`first_stage` candidate generation, then scoring in the Python process.

    Architecturally different from the server-side arms: Qdrant returns candidates and
    the sort happens locally, so Qdrant time is small and client time is large. That
    contrast is the point.

    The candidate query is the top-level query rather than a prefetch, since nothing is
    rescored server-side — but it is the same retrieval as the first stage's baseline
    and the server-side arms' prefetch, so all of them start from the same candidates.
    """
    if depth is None:
        raise ArmError("cross-encoder reranking sweeps depth; depth must be set")
    timers = QueryTimers()
    limit = _final_limit(ctx, depth)
    _guard_prefetch(depth, limit)
    first_stage_vec, using = first_stage_query(ctx, timers, first_stage, query)
    candidates = run_query(
        ctx, timers, map_ids=False, query=first_stage_vec, using=using,
        limit=depth, with_payload=["doc_id", "text"],
    )

    with timers.rerank():
        documents = [(p.payload or {}).get("text", "") for p in candidates]
        scores = list(ctx.encoders.cross.rerank(query, documents)) if documents else []
        order = np.argsort(-np.asarray(scores, dtype=np.float64)) if scores else []
        retrieved = [
            (ctx.id_map.doc(candidates[i].id), float(scores[i])) for i in list(order)[:limit]
        ]

    # The candidates already ARE the pool this arm reranks — no extra query needed.
    pool = [ctx.id_map.doc(p.id) for p in candidates]
    return ArmOutput(retrieved, timers, pool_doc_ids=pool)


@register(
    "cross_encoder",
    sweeps_depth=True,
    requires=["dense"],
    group="rescoring",
    baseline="dense",
    client_side_rerank=True,
)
def cross_encoder_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """Full-precision dense candidates, then a client-side cross-encoder."""
    return _cross_encoder(ctx, query, depth, first_stage="dense")


@register(
    "bm25_cross_encoder",
    sweeps_depth=True,
    requires=["bm25"],
    group="rescoring",
    baseline="bm25",
    client_side_rerank=True,
)
def bm25_cross_encoder_arm(ctx: ArmContext, query: str, depth: int | None) -> ArmOutput:
    """BM25 candidates, then a client-side cross-encoder."""
    return _cross_encoder(ctx, query, depth, first_stage="bm25")


# ------------------------------------------------------------------- resolution


def muvera_calibration_status(cfg: Config, dataset: str) -> tuple[bool, str]:
    """Is `muvera_late_interaction_rescore` allowed to run on this dataset?

    `muvera.calibrated: true` in config is a manual override for someone who has
    verified the model themselves; absent that, a passing calibration artifact for
    THIS dataset must exist on disk. Either one counts only for the late-interaction
    model it was calibrated on. Neither being present is a hard exclusion, not a
    warning — an uncalibrated FDE prefetch can silently measure its own recall ceiling
    rather than reranking quality.
    """
    model = cfg.models.late_interaction.name
    if cfg.models.muvera_calibrated:
        return True, f"models.yaml muvera.calibrated: true for {model}"

    path = cfg.paths["agg"] / f"{dataset}__muvera_calibration.json"
    if not path.exists():
        return False, (
            f"not calibrated: no artifact at {path}. Run "
            f"`calibrate-muvera --dataset {dataset}` first, or set "
            "models.yaml muvera.calibrated: true if you have verified it yourself."
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"not calibrated: {path} is unreadable ({exc})"

    calibrated_on = payload.get("late_interaction_model")
    if calibrated_on != model:
        return False, (
            f"not calibrated: {path} was calibrated on {calibrated_on!r}, not {model!r}. "
            f"Run `calibrate-muvera --dataset {dataset}` with this model."
        )

    if not payload.get("beats_bag_of_tokens"):
        return False, f"not calibrated: {path} shows the FDE does not beat bag-of-tokens"

    recall_at = payload.get("recall_at") or {}
    if not recall_at:
        return False, f"not calibrated: {path} has no recall_at data"
    deepest = max(int(k) for k in recall_at)
    recall = recall_at[str(deepest)]
    if recall < 0.8:
        return False, (
            f"not calibrated: {path} shows recall@{deepest}={recall:.2f}, below the "
            "0.80 threshold"
        )
    return True, f"{path}: recall@{deepest}={recall:.2f}, beats bag-of-tokens"


def resolve_arms(cfg: Config, dataset: str, *, include: Sequence[str] | None = None) -> tuple[list[ArmSpec], dict[str, str]]:
    """Return (runnable arms, exclusions) for a dataset.

    Exclusions are returned, not swallowed: reporting which
    arms did not run and why is required.
    """
    spec = cfg.datasets.get(dataset)
    requested = list(include) if include is not None else list(cfg.run.arms.enabled)
    available = set(spec.representations)

    arms: list[ArmSpec] = []
    exclusions: dict[str, str] = dict(spec.excluded_arms)

    for name in requested:
        if name not in REGISTRY:
            raise KeyError(f"unknown arm {name!r}; known: {', '.join(sorted(REGISTRY))}")
        if name in exclusions:
            continue
        arm = REGISTRY[name]
        missing = [r for r in arm.requires if r not in available]
        if missing:
            exclusions[name] = (
                f"dataset {dataset} does not index representation(s): {', '.join(missing)}"
            )
            continue
        if name == "muvera_late_interaction_rescore":
            ok, reason = muvera_calibration_status(cfg, dataset)
            if not ok:
                exclusions[name] = reason
                continue
        arms.append(arm)

    for name, flag in cfg.run.arms.gated.items():
        if name not in requested and name not in exclusions:
            exclusions[name] = f"gated: enable with {flag}"

    return arms, exclusions
