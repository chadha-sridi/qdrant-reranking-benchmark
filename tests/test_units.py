"""Unit tests for the parts that must be right before any live run.

These run without Docker or model downloads.
"""

from __future__ import annotations

import inspect
import json

import pytest

from src.config import ExecutionConfig, load_config
from src.datasets import IdMap, IdMapError, _hold_out_warmup, _subsample
from src.metrics import (
    LATENCY_COMPONENTS,
    _metric_supported,
    kendall_tau_b,
    latency_percentiles,
    rbo,
    top_k_overlap,
)
from src.storage import RawRecord, RawWriter, completed_keys, repair_truncated, unit_slug
from src.timing import EMBED_STAGES, QueryTimers


# ------------------------------------------------------------------------ config


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg.models.late_interaction.hnsw_m == 0, "multivector HNSW must be disabled"
    muvera = cfg.models.muvera
    assert muvera.dim == muvera.n_reps * muvera.n_partitions * muvera.d_proj
    assert cfg.run.execution.final_limit <= min(cfg.run.execution.depths)


def test_results_dir_moves_every_run_output_but_shares_the_embedding_cache():
    cfg = load_config()
    moved = cfg.with_results_dir("results_new").paths
    root = cfg.root
    assert moved["results"] == root / "results_new"
    assert moved["raw"] == root / "results_new" / "raw"
    assert moved["agg"] == root / "results_new" / "agg"
    assert moved["queries"] == root / "results_new" / "queries"
    assert moved["manifest"] == root / "results_new" / "manifest.json"
    assert moved["cache"] == cfg.paths["cache"]


def test_muvera_calibration_flag_has_a_reason_recorded():
    """Whatever `calibrated` is set to, the note must say why."""
    cfg = load_config()
    assert cfg.models.muvera.calibration_note, "calibration_note must explain the flag's basis"
    if cfg.models.muvera.calibrated:
        assert "calibrat" in cfg.models.muvera.calibration_note.lower()


def test_muvera_calibration_gate_blocks_without_an_artifact_or_override():
    """An uncalibrated MUVERA must not be silently runnable."""
    from src.arms import muvera_calibration_status

    cfg = load_config()
    uncalibrated = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"muvera": cfg.models.muvera.model_copy(update={"calibrated": False})}
            )
        }
    )
    ok, reason = muvera_calibration_status(uncalibrated, "__no_such_dataset__")
    assert not ok
    assert "calibrate-muvera" in reason


def _with_late_interaction_model(cfg, name):
    late_interaction = cfg.models.late_interaction.model_copy(update={"name": name})
    return cfg.model_copy(
        update={"models": cfg.models.model_copy(update={"late_interaction": late_interaction})}
    )


def _with_muvera_calibrated_for(cfg, model):
    muvera = cfg.models.muvera.model_copy(update={"calibrated": True, "calibrated_model": model})
    return cfg.model_copy(update={"models": cfg.models.model_copy(update={"muvera": muvera})})


def test_muvera_calibration_is_refused_at_load_for_another_model():
    """Switching models.yaml to a new late-interaction model must not inherit the old
    model's MUVERA calibration."""
    from src.config import ModelsConfig

    raw = load_config().models.model_dump()
    raw["muvera"]["calibrated"] = True
    raw["muvera"]["calibrated_model"] = "some/other-model"
    with pytest.raises(ValueError, match="Recalibrate"):
        ModelsConfig.model_validate(raw)


def test_muvera_calibration_counts_only_for_the_model_it_was_calibrated_on(tmp_path):
    from src.arms import muvera_calibration_status

    cfg = load_config().with_results_dir(str(tmp_path))
    model = cfg.models.late_interaction.name
    assert muvera_calibration_status(_with_muvera_calibrated_for(cfg, model), "nfcorpus")[0]
    switched = _with_late_interaction_model(_with_muvera_calibrated_for(cfg, model), "some/other-model")
    ok, reason = muvera_calibration_status(switched, "nfcorpus")
    assert not ok and "calibrate-muvera" in reason

    artifact = cfg.paths["agg"] / "nfcorpus__muvera_calibration.json"
    artifact.parent.mkdir(parents=True)
    muvera = cfg.models.muvera
    settings = {"k_sim": muvera.k_sim, "n_reps": muvera.n_reps, "d_proj": muvera.d_proj}
    passing = {"beats_bag_of_tokens": True, "recall_at": {"100": 0.9}, "config": settings}
    artifact.write_text(json.dumps(passing | {"late_interaction_model": "some/other-model"}))
    ok, reason = muvera_calibration_status(cfg, "nfcorpus")
    assert not ok and "some/other-model" in reason
    artifact.write_text(json.dumps(passing | {"late_interaction_model": model}))
    assert muvera_calibration_status(cfg, "nfcorpus")[0]

    retuned = passing | {"late_interaction_model": model, "config": settings | {"k_sim": 6}}
    artifact.write_text(json.dumps(retuned))
    ok, reason = muvera_calibration_status(cfg, "nfcorpus")
    assert not ok and "calibrated with" in reason

    below = passing | {"late_interaction_model": model, "recall_at": {"100": muvera.min_recall - 0.01}}
    artifact.write_text(json.dumps(below))
    assert not muvera_calibration_status(cfg, "nfcorpus")[0]


def test_dense_onnx_source_gets_its_own_identity_and_registers_once():
    """A custom file must never share a name, and so cached embeddings, with the build
    FastEmbed ships under the same model."""
    from fastembed import TextEmbedding

    from src.embedding import _register_onnx_source

    dense = load_config().models.dense
    assert dense.onnx is not None and dense.model_id != dense.name
    assert dense.model_copy(update={"onnx": None}).model_id == dense.name
    _register_onnx_source(dense)
    _register_onnx_source(dense)
    names = [entry["model"] for entry in TextEmbedding.list_supported_models()]
    assert names.count(dense.model_id) == 1


def test_pointwise_scores_are_shared_within_a_query_and_costed_per_pair():
    """Depths and both cross-encoder arms share one forward pass per (query, doc) pair;
    each unit's t_rerank is the attributed measured cost of its own pairs."""
    from types import SimpleNamespace

    from src.arms import _pointwise_scores
    from src.timing import QueryTimers

    calls: list[list[str]] = []

    class Cross:
        def rerank(self, query, texts):
            calls.append(list(texts))
            return [float(len(text)) for text in texts]

    cross = Cross()
    ctx = SimpleNamespace(pointwise_scores={}, encoders=SimpleNamespace(cross=cross))
    ids = [f"d{i}" for i in range(6)]
    texts = ["x" * (i + 1) for i in range(6)]

    deep = QueryTimers()
    scores_deep = _pointwise_scores(ctx, deep, "q", ids, texts)
    shallow = QueryTimers()
    scores_shallow = _pointwise_scores(ctx, shallow, "q", ids[:3], texts[:3])
    assert calls == [texts], "a shallower unit must not re-score cached pairs"
    assert scores_shallow == scores_deep[:3]
    per_pair_ms = ctx.pointwise_scores[ids[0]][1] * 1000.0
    assert deep.t_rerank_ms == pytest.approx(6 * per_pair_ms)
    assert shallow.t_rerank_ms == pytest.approx(3 * per_pair_ms)

    # The other first stage's arm pays only for pairs it alone introduces.
    _pointwise_scores(ctx, QueryTimers(), "q", [ids[0], "other"], [texts[0], "zz"])
    assert calls[1] == ["zz"]

    # Sharing off (cache None): every call is a full, independently timed scoring.
    off = SimpleNamespace(pointwise_scores=None, encoders=SimpleNamespace(cross=cross))
    assert _pointwise_scores(off, QueryTimers(), "q", ids[:2], texts[:2]) == scores_deep[:2]
    assert calls[2] == texts[:2]


def test_muvera_parallel_encoding_matches_serial():
    import numpy as np

    from src.muvera import encode_documents

    cfg = load_config().models.muvera.model_copy(update={"k_sim": 2, "d_proj": 4, "n_reps": 2})
    rng = np.random.default_rng(0)
    docs = [
        rng.standard_normal((int(rng.integers(3, 30)), 16)).astype(np.float32)
        for _ in range(40)
    ]
    serial = encode_documents(cfg, 16, 123, docs, workers=1)
    parallel = encode_documents(cfg, 16, 123, docs, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert serial.dtype == np.float32 and serial.shape == (40, cfg.dim)


def test_vectorized_chamfer_matches_the_per_document_loop():
    import numpy as np

    from src.muvera import chamfer_blocks, chamfer_similarity, exact_chamfer_scores

    rng = np.random.default_rng(1)
    docs = [
        rng.standard_normal((int(rng.integers(1, 9)), 8)).astype(np.float32)
        for _ in range(23)
    ]
    query = rng.standard_normal((5, 8)).astype(np.float32)
    blocks = chamfer_blocks(docs, max_block_tokens=40)  # force several blocks
    assert len(blocks) > 1
    want = np.array([chamfer_similarity(query, doc) for doc in docs])
    np.testing.assert_allclose(exact_chamfer_scores(query, blocks), want, rtol=1e-5)
    with pytest.raises(ValueError, match="empty document"):
        chamfer_blocks([np.zeros((0, 8), dtype=np.float32)])


def test_only_required_representations_are_built():
    from src.arms import REGISTRY, required_representations

    assert required_representations([REGISTRY["dense"], REGISTRY["cross_encoder"]]) == {"dense"}
    assert required_representations([REGISTRY["bm25_cross_encoder"]]) == {"bm25"}
    assert required_representations(list(REGISTRY.values())) == {
        "dense", "bm25", "late_interaction", "muvera",
    }


def test_execution_config_rejects_depth_below_final_limit():
    with pytest.raises(ValueError, match="prefetch.limit"):
        ExecutionConfig(
            repeats=1,
            final_limit=10,
            max_stored_results=100,
            flush_every=5,
            depths=[5, 10],
        )


def test_late_interaction_config_rejects_nonzero_m():
    from src.config import LateInteractionModel

    with pytest.raises(ValueError, match="hnsw_m must be 0"):
        LateInteractionModel(
            name="x", dim=128, distance="cosine", hnsw_m=16, comparator="max_sim",
        )


def test_late_interaction_tokens_are_kept_to_the_model_own_window():
    """The encoder's output is stored as is, however long the model allows it to be."""
    import numpy as np

    from src.embedding import Encoders

    class LongModel:
        def embed(self, texts, batch_size):
            return (np.ones((600, 4)) for _ in texts)

        def query_embed(self, text):
            return iter([np.ones((40, 4))])

    cfg = load_config()
    enc = Encoders(models=cfg.models, muvera_seed=0, _late_interaction=LongModel())
    assert [m.shape[0] for m in enc.embed_docs_late_interaction(["a", "b"])] == [600, 600]
    assert enc.embed_query_late_interaction("q").shape[0] == 40


def test_dataset_set_is_the_agreed_five_with_quora_out_of_the_default_set():
    """Quora is configured but out of `--dataset all`: it is ~80% of the runtime."""
    cfg = load_config()
    assert set(cfg.datasets.datasets) == {"nfcorpus", "scifact", "scidocs", "fiqa", "quora"}
    assert [s.name for s in cfg.datasets.enabled_ordered] == [
        "nfcorpus", "scifact", "scidocs", "fiqa",
    ]
    quora = cfg.datasets.get("quora")
    assert not quora.enabled
    assert quora.disabled_reason, "a dataset out of the default set must say why"


def test_only_quora_caps_its_queries():
    """Every dataset in the default set runs every judged query it has."""
    cfg = load_config()
    assert cfg.run.queries.cap is None, (
        "the global cap must stay unset; it exists only so smoke tests can limit every "
        "dataset at once"
    )
    for spec in cfg.datasets.enabled_ordered:
        assert spec.query_cap is None, f"{spec.name} must run every judged query"
    assert cfg.datasets.get("quora").query_cap == 500


def test_per_dataset_cap_takes_precedence_over_the_global_cap():
    """A dataset's own cap wins; the global one only fills in where there is none."""
    from src.datasets import _subsample

    ids = [f"q{i}" for i in range(1000)]
    cfg = load_config()
    quora, nfcorpus = cfg.datasets.get("quora"), cfg.datasets.get("nfcorpus")

    def effective(spec, global_cap):
        return spec.query_cap if spec.query_cap is not None else global_cap

    assert len(_subsample(ids, effective(quora, None), seed=1)) == 500
    assert len(_subsample(ids, effective(nfcorpus, None), seed=1)) == 1000
    # A global cap (smoke tests) still applies where a dataset sets none.
    assert len(_subsample(ids, effective(nfcorpus, 20), seed=1)) == 20


def test_uncapped_subsample_returns_every_query_sorted():
    ids = [f"q{i}" for i in range(1000)]
    assert _subsample(ids, None, seed=1) == sorted(ids)


def test_only_nfcorpus_sweeps_to_depth_500():
    """NFCorpus is the only dataset still gaining quality past depth 200."""
    cfg = load_config()
    assert cfg.depths_for("nfcorpus") == [10, 25, 50, 100, 200, 500]
    for name in ("scifact", "scidocs", "fiqa", "quora"):
        assert cfg.depths_for(name) == [10, 25, 50, 100, 200], name
        assert 500 not in cfg.depths_for(name)


def test_enabled_datasets_are_ordered_smallest_first():
    cfg = load_config()
    sizes = [spec.corpus_size for spec in cfg.datasets.enabled_ordered]
    assert sizes == sorted(sizes)


# -------------------------------------------------------------------------- arms


def test_registry_has_two_baselines_their_rerankers_and_no_fusion_arms():
    """Two first stages, each with its own rerankers.

    Fusion arms (rrf/rrf_weighted/dbsf) and formula_rrf are out of scope: they combine
    retrievers rather than rescore a candidate set, so they are not reranking under
    this benchmark's scope.
    """
    from src.arms import REGISTRY

    assert set(REGISTRY) == {
        "dense", "late_interaction_rescore", "muvera_late_interaction_rescore", "cross_encoder",
        "bm25", "bm25_late_interaction_rescore", "bm25_cross_encoder",
    }
    assert not {"rrf", "rrf_weighted", "dbsf", "formula_rrf"} & set(REGISTRY)
    # Every arm except the baselines sweeps candidate depth.
    baselines = {a.name for a in REGISTRY.values() if a.group == "baseline"}
    assert baselines == {"dense", "bm25"}
    assert {a.name for a in REGISTRY.values() if not a.sweeps_depth} == baselines


def test_every_arm_is_measured_against_a_baseline_it_shares_a_first_stage_with():
    """An arm's `baseline` must be a baseline arm, and a reranker must read that
    baseline's vector: its gain is attributable to reranking only if both start from
    the same candidates. `muvera_late_interaction_rescore` makes its own candidates and is
    the one arm compared across first stages."""
    from src.arms import REGISTRY

    for arm in REGISTRY.values():
        baseline = REGISTRY[arm.baseline]
        assert baseline.group == "baseline", arm.name
        if arm.group == "baseline":
            assert arm.baseline == arm.name
        elif arm.name != "muvera_late_interaction_rescore":
            assert set(baseline.requires) <= set(arm.requires), arm.name
            prefix = "" if arm.baseline == "dense" else f"{arm.baseline}_"
            assert arm.name.startswith(prefix), arm.name


def test_pool_doc_ids_declared_only_where_cheap_or_meaningful():
    """dense has no separate pool. The other three expose one."""
    from src.arms import REGISTRY

    exposes_pool = {
        "late_interaction_rescore", "muvera_late_interaction_rescore", "cross_encoder",
        "bm25_late_interaction_rescore", "bm25_cross_encoder",
    }
    no_pool = {"dense", "bm25"}
    assert exposes_pool | no_pool == set(REGISTRY)


def test_pool_doc_ids_capture_is_untimed_and_config_gated():
    """The pool probe must be structurally unable to touch the arm's own timers, and
    must issue no call at all when disabled."""
    import inspect

    from qdrant_client import grpc as qgrpc

    from src.arms import ArmContext, _pool_doc_ids
    from src.datasets import IdMap

    cfg = load_config()
    id_map = IdMap.build("d", ["docA", "docB"])
    calls = []

    class FakeGrpcPoints:
        def Query(self, request, timeout=None):
            calls.append(request)
            return qgrpc.QueryResponse(
                result=[
                    qgrpc.ScoredPoint(id=qgrpc.PointId(num=0), score=0.9),
                    qgrpc.ScoredPoint(id=qgrpc.PointId(num=1), score=0.8),
                ],
                time=0.001,
            )

    class FakeClient:
        grpc_points = FakeGrpcPoints()

    ctx = ArmContext(
        client=FakeClient(), cfg=cfg, encoders=None, id_map=id_map,
        collection="c", dataset="d",
    )
    assert _pool_doc_ids(ctx, query=[0.1], using="muvera", depth=10) == ["docA", "docB"]
    assert len(calls) == 1

    # It takes no timers argument, so it cannot write into the arm's measurement.
    assert "timers" not in inspect.signature(_pool_doc_ids).parameters

    # Disabled via config: no call at all, not merely an untimed one.
    disabled = cfg.model_copy(update={"run": cfg.run.model_copy(update={
        "execution": cfg.run.execution.model_copy(update={"capture_pool_recall": False})
    })})
    ctx_disabled = ArmContext(
        client=FakeClient(), cfg=disabled, encoders=None, id_map=id_map,
        collection="c", dataset="d",
    )
    assert _pool_doc_ids(ctx_disabled, query=[0.1], using="muvera", depth=10) is None
    assert len(calls) == 1, "disabled must not issue the call at all"


def test_resolve_arms_matches_the_default_set(tmp_path):
    """With MUVERA calibrated for the configured model, nothing is excluded."""
    from src.arms import REGISTRY, resolve_arms

    cfg = load_config().with_results_dir(str(tmp_path))
    cfg = _with_muvera_calibrated_for(cfg, cfg.models.late_interaction.name)
    arms, exclusions = resolve_arms(cfg, "nfcorpus")
    assert not exclusions, f"unexpected exclusions: {exclusions}"
    assert {a.name for a in arms} == set(cfg.run.arms.enabled)
    assert set(cfg.run.arms.enabled) == set(REGISTRY)


def test_per_dataset_depths_fall_back_to_the_global_sweep():
    """Depths are per dataset, with a global default to fall back on."""
    cfg = load_config()
    # scifact sets no override, so it inherits the global sweep.
    assert cfg.datasets.get("scifact").depths is None
    assert cfg.depths_for("scifact") == cfg.run.execution.depths
    # nfcorpus overrides, so it must NOT equal the global sweep.
    assert cfg.datasets.get("nfcorpus").depths is not None
    assert cfg.depths_for("nfcorpus") != cfg.run.execution.depths
    # Whatever a dataset resolves to, no depth may fall below the reported limit.
    for name in cfg.datasets.datasets:
        assert min(cfg.depths_for(name)) >= cfg.run.execution.final_limit, name


def test_depths_override_wins_over_every_dataset_own_override():
    """`--depths` must win everywhere, including over a dataset's own `depths`.

    A dataset that sets its own sweep (NFCorpus does) would otherwise silently ignore a
    global-only override."""
    cfg = load_config()
    assert cfg.datasets.get("nfcorpus").depths is not None, "test assumes nfcorpus has its own override"
    overridden = cfg.with_depths_override([10, 50])
    assert overridden.depths_for("nfcorpus") == [10, 50]
    assert overridden.depths_for("scifact") == [10, 50]


def test_resolve_arms_excludes_by_missing_representation():
    """A dataset that does not index a representation excludes the arms needing it."""
    cfg = load_config()
    from src.arms import resolve_arms

    spec = cfg.datasets.get("nfcorpus")
    dense_only = spec.model_copy(update={"representations": ["dense"]})
    trimmed = cfg.model_copy(
        update={
            "datasets": cfg.datasets.model_copy(
                update={"datasets": {**cfg.datasets.datasets, "nfcorpus": dense_only}}
            )
        }
    )
    _, exclusions = resolve_arms(trimmed, "nfcorpus", include=["dense", "late_interaction_rescore"])
    assert "late_interaction_rescore" in exclusions


# ------------------------------------------------------------------ query transport


def test_query_message_accepts_arm_vocabulary_and_rejects_unknown_arguments():
    """Arms speak REST vocabulary; one place translates it to gRPC.

    The explicit signature matters: a search parameter this function does not understand
    must be a TypeError at the call site, never a silently dropped argument that changes
    what was measured.
    """
    from qdrant_client import models

    from src.arms import ArmContext, _query_message
    from src.datasets import IdMap

    cfg = load_config()
    ctx = ArmContext(
        client=None, cfg=cfg, encoders=None,
        id_map=IdMap.build("d", ["docA"]), collection="c", dataset="d",
    )

    msg = _query_message(
        ctx,
        prefetch=[models.Prefetch(query=[0.1, 0.2], using="dense", limit=50)],
        query=[[0.1, 0.2], [0.3, 0.4]],           # multivector, as late_interaction arms send
        using="late_interaction", limit=10, with_payload=["doc_id"],
        params=models.SearchParams(hnsw_ef=128),
    )
    assert msg.collection_name == "c"
    assert msg.using == "late_interaction"
    assert msg.limit == 10
    assert len(msg.prefetch) == 1
    assert msg.params.hnsw_ef == 128

    # A bare (non-multivector) query is accepted too.
    assert _query_message(
        ctx, query=[0.1, 0.2], using="dense", limit=5, with_payload=["doc_id"]
    ).limit == 5

    # An argument it does not model must fail loudly, not vanish.
    with pytest.raises(TypeError):
        _query_message(
            ctx, query=[0.1], using="dense", limit=5, with_payload=["doc_id"],
            score_threshold=0.9,
        )


def test_full_scan_threshold_is_a_value_qdrant_accepts():
    """Qdrant rejects full_scan_threshold=0 with "must be 10 or larger".

    Set to 0, every create_collection call fails at runtime with a 422 — and no fast
    test would notice, because they do not build collections.
    """
    from src.config import HnswConfig, load_config

    cfg = load_config()
    assert cfg.run.qdrant.hnsw.full_scan_threshold_kb >= 10

    with pytest.raises(ValueError, match="10 or larger"):
        HnswConfig(m=16, ef_construct=100, full_scan_threshold_kb=0)


def test_the_index_is_always_used_not_full_scanned():
    """Qdrant's 10000 KB default would full-scan the two smallest datasets.

    Latency would then compare ANN on the big corpora against brute force on the small
    ones, which is not a corpus-size curve.
    """
    from src.config import load_config

    cfg = load_config()
    assert cfg.run.qdrant.hnsw.full_scan_threshold_kb < 10_000


# ------------------------------------------------------------- ingestion integrity


def test_verify_ingestion_rejects_a_short_collection():
    """Silent under-ingestion inflates latency and deflates recall; it must not pass."""
    from src.indexing import IngestionError, verify_ingestion

    class ShortClient:
        def count(self, collection_name, exact):
            return type("C", (), {"count": 1098})()

    with pytest.raises(IngestionError, match="1098 points present, expected 3633"):
        verify_ingestion(ShortClient(), "c", expected_points=3633)


def test_verify_ingestion_rejects_an_unindexed_collection():
    """indexing_threshold=0 reports green with no HNSW built: brute-force latency."""
    from qdrant_client import models

    from src.indexing import IngestionError, verify_ingestion

    class UnindexedClient:
        def count(self, collection_name, exact):
            return type("C", (), {"count": 10})()

        def get_collection(self, name):
            return type(
                "I",
                (),
                {
                    "status": models.CollectionStatus.GREEN,
                    "config": type(
                        "Cfg", (), {"optimizer_config": type("O", (), {"indexing_threshold": 0})()}
                    )(),
                },
            )()

    with pytest.raises(IngestionError, match="indexing_threshold is 0"):
        verify_ingestion(UnindexedClient(), "c", expected_points=10)

    # Tolerated where there are no vectors to index (the payload-only baseline).
    verify_ingestion(UnindexedClient(), "c", expected_points=10, require_indexed=False)


def test_ingestion_resume_targets_only_the_missing_points():
    """Resume must re-send exactly the gap — deterministic ids make that non-duplicating."""
    from src.indexing import _missing_point_indices

    class PartialClient:
        """Holds points 0..99; 100..149 never landed."""

        def retrieve(self, collection_name, ids, with_payload, with_vectors):
            return [type("P", (), {"id": i})() for i in ids if i < 100]

    missing = _missing_point_indices(PartialClient(), "c", list(range(150)))
    assert missing == list(range(100, 150))


def test_missing_point_indices_scopes_to_a_chunk_not_the_whole_corpus():
    """The chunked ingestion path checks one chunk's ids, not a full-corpus scan."""
    from src.indexing import _missing_point_indices

    class PartialClient:
        """Holds points 0..99 only."""

        def __init__(self):
            self.seen_ids: list[int] = []

        def retrieve(self, collection_name, ids, with_payload, with_vectors):
            self.seen_ids.extend(ids)
            return [type("P", (), {"id": i})() for i in ids if i < 100]

    client = PartialClient()
    # A later chunk, entirely missing: only its own ids should ever be checked.
    missing = _missing_point_indices(client, "c", list(range(200, 210)))
    assert missing == list(range(200, 210))
    assert set(client.seen_ids) == set(range(200, 210))


def test_point_from_chunk_uses_global_index_for_the_document_and_local_index_for_the_vector():
    """A chunk starting mid-corpus must not silently overwrite an earlier chunk's points.

    `chunk_vectors` is indexed from 0 within the chunk; `bundle.documents` and
    `bundle.id_map` are indexed globally. Conflating the two makes every chunk after
    the first fetch the wrong document while attaching a point id that collides with
    an earlier chunk's, silently overwriting it instead of adding a new point.
    """
    import numpy as np

    from src.datasets import Document, IdMap
    from src.indexing import DocumentVectors, _point_from_chunk

    n_docs = 10
    doc_ids = [f"doc{i}" for i in range(n_docs)]
    documents = [Document(doc_id=d, text=f"text-{d}") for d in doc_ids]
    id_map = IdMap.build("fake", doc_ids)

    class FakeBundle:
        pass

    bundle = FakeBundle()
    bundle.documents = documents
    bundle.id_map = id_map

    chunk_start, chunk_end = 5, 10  # second chunk of a 5-doc-per-chunk split
    # Distinguishable per-position values so a local/global index mix-up is visible.
    chunk_vectors = DocumentVectors(dense=np.arange(chunk_start, chunk_end).reshape(-1, 1).astype("float32"))

    global_i = 7  # doc7, third position in this chunk (local_i = 2)
    point = _point_from_chunk(bundle, chunk_vectors, ["dense"], global_i, chunk_start)

    assert point.payload["doc_id"] == "doc7"
    assert point.id == id_map.point("doc7")
    assert point.vector["dense"] == [7.0]  # chunk_vectors[local_i=2] == arange(5,10)[2] == 7


def test_bm25_index_scores_with_bm25_and_matches_standard_english_processing():
    """Qdrant's default text index only lowercases; without stemming and stopwords the
    baseline would be weaker than published English BM25. It must also stay out of the
    dense HNSW graph, which every dense arm searches."""
    from qdrant_client import models

    from src.indexing import bm25_index_params

    params = bm25_index_params(load_config())
    assert params.scoring == models.TextScoringParams(type=models.TextScoringType.BM25)
    assert params.stemmer == models.SnowballParams(
        type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
    )
    assert params.stopwords == models.Language.ENGLISH
    assert params.lowercase is True
    assert params.enable_hnsw is False


def test_bm25_indexes_the_payload_key_documents_are_stored_under():
    from src.datasets import Document, IdMap
    from src.indexing import DocumentVectors, _point_from_chunk

    class FakeBundle:
        documents = [Document(doc_id="d0", text="some text")]
        id_map = IdMap.build("fake", ["d0"])

    point = _point_from_chunk(FakeBundle(), DocumentVectors(), [], 0, 0)
    assert point.payload[load_config().models.bm25.field] == "some text"


def test_collection_without_the_bm25_index_is_rebuilt_not_resumed():
    """Resuming into a collection with no BM25 index, or one with different text
    processing, would serve BM25 from the wrong index or none."""
    from types import SimpleNamespace

    from qdrant_client import models

    from src.indexing import (
        _config_matches,
        bm25_index_params,
        collection_metadata,
        vectors_config,
    )

    cfg = load_config()
    spec = cfg.datasets.get("nfcorpus")
    reps = spec.representations
    assert "bm25" in reps
    field = cfg.models.bm25.field

    def client_with(payload_schema):
        params = SimpleNamespace(vectors=vectors_config(cfg, spec, reps))
        config = SimpleNamespace(params=params, metadata=collection_metadata(cfg, reps))
        info = SimpleNamespace(config=config, payload_schema=payload_schema)
        return SimpleNamespace(get_collection=lambda name: info)

    def index(params):
        return {field: models.PayloadIndexInfo(data_type=models.PayloadSchemaType.TEXT, params=params, points=1)}

    assert _config_matches(client_with(index(bm25_index_params(cfg))), cfg, spec, "c", reps)
    assert not _config_matches(client_with({}), cfg, spec, "c", reps)
    unscored = bm25_index_params(cfg).model_copy(update={"scoring": None})
    assert not _config_matches(client_with(index(unscored)), cfg, spec, "c", reps)
    unstemmed = bm25_index_params(cfg).model_copy(update={"stemmer": None})
    assert not _config_matches(client_with(index(unstemmed)), cfg, spec, "c", reps)


def test_collection_built_by_another_model_of_the_same_dim_is_rebuilt_not_resumed():
    """ColBERTv2, LateOn and mLateOn are all 128-dim, so vector sizes alone would let a
    resume mix two models' vectors in one collection."""
    from types import SimpleNamespace

    from src.indexing import _config_matches, collection_metadata, vectors_config

    cfg = load_config()
    spec = cfg.datasets.get("scifact")
    reps = [r for r in spec.representations if r != "bm25"]

    def client_built_by(build_cfg):
        params = SimpleNamespace(vectors=vectors_config(build_cfg, spec, reps))
        config = SimpleNamespace(params=params, metadata=collection_metadata(build_cfg, reps))
        return SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=config))

    other = _with_late_interaction_model(cfg, "colbert-ir/colbertv2.0")
    assert _config_matches(client_built_by(cfg), cfg, spec, "c", reps)
    assert not _config_matches(client_built_by(cfg), other, spec, "c", reps)
    unrecorded = SimpleNamespace(
        get_collection=lambda name: SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=vectors_config(cfg, spec, reps)), metadata=None
            )
        )
    )
    assert not _config_matches(unrecorded, cfg, spec, "c", reps)


def test_bm25_first_stage_is_a_server_side_text_query_with_configured_k_and_b():
    """BM25 sends raw text, so it must charge nothing to client-side embedding."""
    from types import SimpleNamespace

    from qdrant_client import models

    from src.arms import _query_message, first_stage_query

    cfg = load_config()
    timers = QueryTimers()
    query, using = first_stage_query(SimpleNamespace(cfg=cfg), timers, "bm25", "vitamin d")
    assert using == cfg.models.bm25.field
    assert isinstance(query, models.TextQuery)
    assert query.text.query == "vitamin d"
    assert query.text.scoring.bm25 == models.Bm25Params(k=cfg.models.bm25.k, b=cfg.models.bm25.b)
    assert timers.t_embed_ms == 0.0

    message = _query_message(
        SimpleNamespace(collection="c"), query=query, using=using, limit=5, with_payload=["doc_id"]
    )
    assert message.query.text.query == "vitamin d"
    assert message.using == using


def test_payload_index_memory_is_read_per_field_and_mapped_to_bm25():
    from src.memory import (
        CollectionMemory,
        Component,
        MemoryReadError,
        representation_memory,
        verify_expected_representations,
    )

    cfg = load_config()
    memory = CollectionMemory(
        dataset="d", collection="c", n_documents=1, n_token_vectors=None,
        payload_indexes={cfg.models.bm25.field: Component(disk_bytes=7, ram_bytes=3)},
    )
    reported = representation_memory(cfg, memory)
    assert reported["bm25"]["total"]["disk_bytes"] == 7
    verify_expected_representations(cfg, memory, ["bm25"])
    with pytest.raises(MemoryReadError, match="no memory for"):
        verify_expected_representations(cfg, memory, ["bm25", "dense"])


def test_payload_index_entry_without_usage_is_an_error_not_zero(monkeypatch):
    """Qdrant 1.20 nests each payload index's bytes under `usage`; an entry without it
    means the format moved again, and reading it as zero would understate memory."""
    from src import memory as memory_module

    cfg = load_config()
    result = {"vectors": [], "payload_index": [{"name": "text", "disk_bytes": 5}]}
    monkeypatch.setattr(memory_module, "_fetch", lambda *a, **k: result)
    with pytest.raises(memory_module.MemoryReadError, match="unexpected payload_index entry"):
        memory_module.read_collection_memory(cfg, "c", dataset="d", n_documents=1)


def test_first_stage_query_rejects_an_unknown_first_stage():
    from src.arms import ArmError, first_stage_query

    with pytest.raises(ArmError, match="unknown first stage"):
        first_stage_query(None, QueryTimers(), "splade", "q")


def test_chunk_bounds_covers_the_corpus_exactly_once():
    from src.indexing import _chunk_bounds

    bounds = _chunk_bounds(25, 10)
    assert bounds == [(0, 10), (10, 20), (20, 25)]
    covered = [i for start, end in bounds for i in range(start, end)]
    assert covered == list(range(25))

    # A corpus smaller than one chunk is a single chunk, not a special case.
    assert _chunk_bounds(5, 10) == [(0, 5)]


# --------------------------------------------------------------- embedding cache


def test_cache_save_is_atomic_no_torn_file_is_ever_visible(tmp_path, monkeypatch):
    """A crash mid-write must leave either the old complete file or nothing — never a
    truncated one that a resume would load as valid."""
    import numpy as np

    from src.embedding import EmbeddingCache

    cache = EmbeddingCache(tmp_path)
    key = "testkey"
    vectors = np.ones((10, 4), dtype=np.float32)

    original_replace = __import__("os").replace
    calls = []

    def _boom(*a, **k):
        calls.append(a)
        raise OSError("simulated crash before publish")

    import os as os_module

    monkeypatch.setattr(os_module, "replace", _boom)
    with pytest.raises(OSError):
        cache.save_dense(key, vectors, {"model": "m", "dataset": "d"})
    monkeypatch.setattr(os_module, "replace", original_replace)

    # The publish step never ran, so nothing resume-visible exists — not a half file.
    assert cache.load_dense(key) is None
    data_path = cache._data_path(key, ".dense.npy")
    assert not data_path.exists()
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files, "expected the temp file to exist, orphaned by the simulated crash"

    # A subsequent successful save is unaffected by the earlier failed attempt.
    cache.save_dense(key, vectors, {"model": "m", "dataset": "d"})
    loaded = cache.load_dense(key)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, vectors)


def test_cache_key_differs_per_chunk_bounds():
    """Two chunks of the same dataset/model/digest must not collide in the cache."""
    from src.embedding import EmbeddingCache

    cache = EmbeddingCache.__new__(EmbeddingCache)  # key() touches no disk
    common = dict(representation="dense", model_name="m", dataset="d", corpus_digest="x")
    key_a = cache.key(**common, extra={"chunk_start": 0, "chunk_end": 100})
    key_b = cache.key(**common, extra={"chunk_start": 100, "chunk_end": 200})
    assert key_a != key_b


# ----------------------------------------------------------------------- indexing


def _with_grpc(cfg, prefer_grpc: bool):
    qdrant = cfg.run.qdrant.model_copy(update={"prefer_grpc": prefer_grpc})
    return cfg.model_copy(update={"run": cfg.run.model_copy(update={"qdrant": qdrant})})


def test_upsert_batch_is_capped_below_qdrant_payload_limit():
    """A fixed batch of wide MUVERA vectors exceeds Qdrant's 32 MiB request ceiling.

    256 points x 10,240 float32 dims serialises to ~48 MB as JSON and the upsert fails
    with a 400. The batch must shrink with the vector width.
    """
    from src.indexing import DocumentVectors, safe_batch_size

    cfg = load_config()
    reps = ["dense", "late_interaction", "muvera"]
    vectors = DocumentVectors(n_token_vectors=12 * 200)

    http = safe_batch_size(_with_grpc(cfg, False), vectors, reps, 200)
    grpc = safe_batch_size(_with_grpc(cfg, True), vectors, reps, 200)

    assert 1 <= http <= cfg.run.qdrant.upsert_batch_size
    assert http < cfg.run.qdrant.upsert_batch_size, "JSON transport must shrink the batch"
    assert grpc >= http, "binary transport fits more points per request"

    # The whole point: the resulting request stays under the limit.
    from src.indexing import estimate_point_floats

    floats = estimate_point_floats(cfg, vectors, reps, 200)
    assert http * floats * 14 < 32 * 1024**2


def test_upsert_batch_respects_configured_ceiling_for_narrow_vectors():
    from src.indexing import DocumentVectors, safe_batch_size

    cfg = load_config()
    batch = safe_batch_size(_with_grpc(cfg, True), DocumentVectors(), ["dense"], 100)
    assert batch == cfg.run.qdrant.upsert_batch_size


# ---------------------------------------------------------------------- id mapping


def test_id_map_roundtrips():
    id_map = IdMap.build("test", ["MED-1", "MED-2", "MED-3"])
    assert id_map.point("MED-2") == 1
    assert id_map.doc(1) == "MED-2"
    assert len(id_map) == 3


def test_id_map_rejects_duplicates():
    with pytest.raises(IdMapError, match="duplicate doc_ids"):
        IdMap.build("test", ["a", "b", "a"])


def test_id_map_rejects_unknown_ids():
    id_map = IdMap.build("test", ["a"])
    with pytest.raises(IdMapError, match="not in corpus"):
        id_map.point("zzz")
    with pytest.raises(IdMapError, match="not in map"):
        id_map.doc(99)


def test_id_map_persists_and_detects_tampering(tmp_path):
    id_map = IdMap.build("test", [f"doc-{i}" for i in range(10)])
    path = tmp_path / "map.json"
    id_map.save(path)
    assert IdMap.load(path).digest == id_map.digest

    payload = json.loads(path.read_text())
    payload["doc_ids"][0] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(IdMapError, match="digest mismatch"):
        IdMap.load(path)


# ------------------------------------------------------------- sampling and splits


def test_subsample_is_seed_stable_and_capped():
    ids = [f"q{i}" for i in range(1000)]
    a = _subsample(ids, 500, seed=42)
    b = _subsample(ids, 500, seed=42)
    c = _subsample(ids, 500, seed=43)
    assert a == b and len(a) == 500
    assert a != c


def test_subsample_returns_everything_below_the_cap():
    ids = [f"q{i}" for i in range(300)]
    assert _subsample(ids, 500, seed=1) == sorted(ids)


def test_warmup_holdout_is_disjoint_and_stable():
    """No training happens — warmup is a cache holdout, not a train split."""
    ids = [f"q{i}" for i in range(400)]
    warm_a, eval_a = _hold_out_warmup(ids, 50, seed=9)
    warm_b, eval_b = _hold_out_warmup(ids, 50, seed=9)
    assert (warm_a, eval_a) == (warm_b, eval_b)
    assert not set(warm_a) & set(eval_a), "warmup and evaluated sets must not overlap"
    assert len(warm_a) + len(eval_a) == len(ids)
    assert len(warm_a) == 50
    # The evaluated set keeps almost everything.
    assert len(eval_a) == 350


def test_warmup_shrinks_rather_than_starving_a_small_query_set():
    """A 50-query dataset must not lose its whole evaluation set to warmup."""
    ids = [f"q{i}" for i in range(50)]
    warm, evaluated = _hold_out_warmup(ids, 50, seed=1)
    assert len(warm) <= len(ids) // 3
    assert len(evaluated) >= 34, "most of a small query set must still be evaluated"


# ------------------------------------------------------------------------ timing


def test_timers_accumulate_independently():
    timers = QueryTimers()
    with timers.embed("dense"):
        pass
    with timers.qdrant():
        pass
    with timers.qdrant():
        pass
    payload = timers.as_dict()
    assert payload["t_rerank_ms"] == 0.0
    assert payload["t_total_ms"] == pytest.approx(
        payload["t_embed_ms"] + payload["t_qdrant_ms"] + payload["t_rerank_ms"], abs=1e-3
    )


def test_embed_time_is_bucketed_per_encoder_and_summed():
    """Each encoder's query-side cost is recorded on its own; t_embed_ms is their sum."""
    timers = QueryTimers(t_embed_dense_ms=1.0, t_embed_late_interaction_ms=2.0, t_embed_muvera_ms=4.0)
    assert timers.t_embed_ms == pytest.approx(7.0)
    payload = timers.as_dict()
    assert payload["t_embed_dense_ms"] == 1.0
    assert payload["t_embed_late_interaction_ms"] == 2.0
    assert payload["t_embed_muvera_ms"] == 4.0
    assert payload["t_embed_ms"] == pytest.approx(7.0)
    assert payload["t_total_ms"] == pytest.approx(7.0)


def test_embed_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="unknown embed stage"):
        with QueryTimers().embed("sparse"):
            pass


def test_every_arm_buckets_its_query_embedding():
    """No arm may use an unbucketed embed timer: every encoder call names its stage."""
    import re
    from src import arms as arms_module

    source = inspect.getsource(arms_module)
    assert "timers.embed()" not in source
    stages = set(re.findall(r'timers\.embed\("(\w+)"\)', source))
    assert stages == set(EMBED_STAGES), stages


def test_server_time_is_tracked_separately_from_the_round_trip():
    """40-60% of a round trip is transport + deserialisation, not search."""
    timers = QueryTimers()
    with timers.qdrant():
        pass
    timers.t_qdrant_ms = 10.0
    timers.add_server_time(0.004)  # 4 ms reported by Qdrant
    assert timers.t_qdrant_server_ms == pytest.approx(4.0)
    assert timers.t_qdrant_overhead_ms == pytest.approx(6.0)
    payload = timers.as_dict()
    assert payload["t_qdrant_server_ms"] == pytest.approx(4.0)
    assert payload["t_qdrant_overhead_ms"] == pytest.approx(6.0)
    # Server time is not double counted into the total.
    assert payload["t_total_ms"] == pytest.approx(10.0)


def test_server_time_is_none_when_transport_does_not_report_it():
    timers = QueryTimers()
    timers.add_server_time(None)
    assert timers.t_qdrant_server_ms is None
    assert timers.t_qdrant_overhead_ms is None
    assert timers.as_dict()["t_qdrant_server_ms"] is None


def test_server_time_accumulates_over_multiple_round_trips():
    timers = QueryTimers()
    timers.add_server_time(0.002)
    timers.add_server_time(0.003)
    assert timers.t_qdrant_server_ms == pytest.approx(5.0)


def test_query_client_always_uses_grpc():
    """Server-reported time is only readable off the gRPC response, so queries require it."""
    from src.indexing import build_client

    cfg = load_config()
    assert build_client(cfg, for_queries=True)._client._prefer_grpc is True
    # Even if the configured transport were REST, a query client still needs gRPC.
    rest_cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={
        "qdrant": cfg.run.qdrant.model_copy(update={"prefer_grpc": False})})})
    assert build_client(rest_cfg, for_queries=True)._client._prefer_grpc is True
    assert build_client(rest_cfg, for_queries=False)._client._prefer_grpc is False


def test_latency_components_include_the_server_split():

    assert "t_qdrant_server_ms" in LATENCY_COMPONENTS
    assert "t_qdrant_overhead_ms" in LATENCY_COMPONENTS
    assert {f"t_embed_{stage}_ms" for stage in EMBED_STAGES} <= set(LATENCY_COMPONENTS)


def test_timers_record_time_even_on_exception():
    """A failed call still cost wall-clock time; dropping it would understate latency."""
    timers = QueryTimers()
    with pytest.raises(RuntimeError):
        with timers.qdrant():
            raise RuntimeError("boom")
    assert timers.t_qdrant_ms > 0


# ------------------------------------------------------------------------ storage


def test_unit_slug_uses_none_not_zero_for_non_sweeping_arms():
    assert unit_slug("nfcorpus", "dense", None) == "nfcorpus__dense__depth-none"
    assert unit_slug("nfcorpus", "late_interaction_rescore", 100) == "nfcorpus__late_interaction_rescore__depth-100"


def test_raw_writer_appends_and_resume_reads_back(tmp_path):
    with RawWriter(tmp_path, flush_every=1) as writer:
        for run_idx in range(2):
            writer.write(
                RawRecord(
                    query_id="PLAIN-2",
                    arm="dense",
                    dataset="nfcorpus",
                    depth=None,
                    run_idx=run_idx,
                    retrieved=[("MED-1", 0.9)],
                    timers=QueryTimers(),
                )
            )
    done = completed_keys(tmp_path, "nfcorpus")
    assert ("dense", None, "PLAIN-2", 0) in done
    assert ("dense", None, "PLAIN-2", 1) in done
    assert len(done) == 2


def test_truncated_final_line_is_survivable(tmp_path):
    path = tmp_path / "nfcorpus__dense__depth-none.jsonl"
    good = json.dumps({"query_id": "q1", "arm": "dense", "depth": None, "run_idx": 0})
    path.write_text(good + "\n" + '{"query_id": "q2", "arm": "de')

    done = completed_keys(tmp_path, "nfcorpus")
    assert done == {("dense", None, "q1", 0)}

    touched = repair_truncated(tmp_path, "nfcorpus")
    assert touched == [path]
    assert path.read_text().strip() == good


def test_raw_record_json_carries_every_timer():
    payload = json.loads(
        RawRecord(
            query_id="q", arm="dense", dataset="nfcorpus", depth=100, run_idx=0,
            retrieved=[("d1", 1.0)], timers=QueryTimers(t_embed_dense_ms=1.0, t_qdrant_ms=2.0),
        ).to_json()
    )
    assert set(LATENCY_COMPONENTS) <= payload.keys()
    assert payload["retrieved"] == [["d1", 1.0]]
    assert payload["depth"] == 100


# ------------------------------------------------------------------------ metrics


def test_count_relevant_counts_graded_matches_only():
    from src.metrics import count_relevant

    qrels = {"docA": 2, "docB": 1, "docC": 0}  # grade 0 = judged not relevant
    assert count_relevant(qrels, ["docA", "docB", "docC", "docZ"]) == 2
    assert count_relevant(qrels, []) == 0
    assert count_relevant({}, ["docA"]) == 0


def test_pool_recall_averages_per_query_ratios():
    from src.metrics import pool_recall

    qrels = {
        "q1": {"docA": 1, "docB": 1, "docC": 0},  # 2 relevant
        "q2": {"docX": 1},  # 1 relevant
        "q3": {},  # 0 relevant -> excluded, not a divide-by-zero
    }
    records = [
        {"query_id": "q1", "run_idx": 0, "n_relevant_in_pool": 1},  # 1/2
        {"query_id": "q2", "run_idx": 0, "n_relevant_in_pool": 1},  # 1/1
        {"query_id": "q3", "run_idx": 0, "n_relevant_in_pool": 0},  # excluded
        {"query_id": "q1", "run_idx": 1, "n_relevant_in_pool": 0},  # wrong run_idx, ignored
    ]
    assert pool_recall(records, qrels) == pytest.approx((0.5 + 1.0) / 2)


def test_pool_recall_is_none_when_nothing_exposes_a_pool():
    from src.metrics import pool_recall

    dense_records = [{"query_id": "q1", "run_idx": 0}]  # no n_relevant_in_pool key
    assert pool_recall(dense_records, {"q1": {"docA": 1}}) is None


def test_recall_at_100_is_unsupported_below_depth_100():
    assert _metric_supported("recall@100", 100) is True
    assert _metric_supported("recall@100", 50) is False
    assert _metric_supported("ndcg@10", 10) is True
    assert _metric_supported("recall@100", None) is True


def test_latency_percentiles_cover_every_component():
    records = [
        {"t_embed_ms": float(i), "t_qdrant_ms": float(i * 2), "t_rerank_ms": 0.0, "t_total_ms": float(i * 3)}
        for i in range(1, 101)
    ]
    from src.metrics import PERCENTILES

    out = latency_percentiles(records)
    for component in ("t_embed_ms", "t_qdrant_ms", "t_rerank_ms", "t_total_ms"):
        for pct in PERCENTILES:
            assert f"{component}_p{pct}" in out
    assert out["t_embed_ms_p50"] == pytest.approx(50.5)
    assert out["n_latency_samples"] == 100
    # p99 is deliberately not reported at one pass per query.
    assert "t_total_ms_p99" not in out


def test_overlap_tau_and_rbo_on_identical_and_disjoint_lists():
    a = [f"d{i}" for i in range(10)]
    assert top_k_overlap(a, a) == 1.0
    assert rbo(a, a, p=0.9) == pytest.approx(1.0, abs=1e-6)
    assert kendall_tau_b(a, a) == pytest.approx(1.0)

    b = [f"x{i}" for i in range(10)]
    assert top_k_overlap(a, b) == 0.0
    assert rbo(a, b, p=0.9) < 0.05


def test_rbo_is_top_weighted():
    a = ["d1", "d2", "d3", "d4"]
    swap_top = ["d2", "d1", "d3", "d4"]
    swap_bottom = ["d1", "d2", "d4", "d3"]
    assert rbo(a, swap_top, p=0.9) < rbo(a, swap_bottom, p=0.9)


def test_significance_p_values_are_actually_extracted():
    """`report.comparisons` is a FrozensetDict, not a plain dict, so an
    `isinstance(table, dict)` gate on it is always False and every p-value comes back
    None with no error at all. Uses real ranx, no network/Qdrant needed.
    """
    from src.metrics import compare_arms

    rng = __import__("numpy").random.default_rng(0)
    qrels = {f"q{i}": {f"d{j}": int(j < 3) for j in range(10)} for i in range(30)}
    run_a = {q: {f"d{j}": float(rng.random()) for j in range(10)} for q in qrels}
    run_b = {q: {f"d{j}": float(rng.random()) for j in range(10)} for q in qrels}

    result = compare_arms(qrels, {"a": run_a, "b": run_b})
    row = result.pairs.iloc[0]
    assert row["p_value"] is not None
    assert 0.0 <= row["p_value"] <= 1.0
    assert row["significant_raw"] is not None


def test_kendall_tau_b_detects_a_reversal():
    a = [f"d{i}" for i in range(10)]
    assert kendall_tau_b(a, list(reversed(a))) == pytest.approx(-1.0)


# ------------------------------------------------------------------ server memory fit


def _collection_with_working_set(gib: float):
    from src.memory import CollectionMemory, Component

    return CollectionMemory(
        dataset="fiqa",
        collection="fiqa",
        n_documents=1,
        n_token_vectors=None,
        total=Component(expected_cache_bytes=int(gib * 1024**3)),
    )


def test_collection_that_fits_in_server_memory_passes():
    from src.memory import check_fits_in_memory

    check_fits_in_memory(_collection_with_working_set(6.4), server_memory_bytes=10 * 1024**3)


def test_collection_larger_than_server_memory_is_refused():
    """FiQA's 6.4 GiB working set in a 1.9 GiB VM made the large-vector arms disk-bound."""
    from src.memory import MemoryReadError, check_fits_in_memory

    with pytest.raises(MemoryReadError, match="disk-bound"):
        check_fits_in_memory(_collection_with_working_set(6.4), server_memory_bytes=int(1.9 * 1024**3))


def test_fit_check_leaves_headroom_for_qdrant_itself():
    from src.memory import MAX_WORKING_SET_FRACTION, MemoryReadError, check_fits_in_memory

    server = 10 * 1024**3
    just_over = server * MAX_WORKING_SET_FRACTION / 1024**3 + 0.01
    with pytest.raises(MemoryReadError):
        check_fits_in_memory(_collection_with_working_set(just_over), server_memory_bytes=server)
