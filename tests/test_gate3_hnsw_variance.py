"""Gate 3 — HNSW build variance is negligible.

Index `dense` twice into two collections, run the same queries, compare NDCG@10. Expect
a gap under ~0.005. This quantifies HNSW graph-construction nondeterminism, which is the
floor below which no arm-to-arm difference is interpretable.
"""

from __future__ import annotations

import pytest

from src.arms import ArmContext, REGISTRY
from src.indexing import build_client, create_collection, embed_corpus, upsert_documents
from src.metrics import PRIMARY_METRIC, build_run_dict, evaluate_quality

pytestmark = [pytest.mark.gate, pytest.mark.slow]

THRESHOLD = 0.005
N_QUERIES = 100


def _score_dense(cfg, bundle, encoders, collection: str) -> float:
    client = build_client(cfg)
    ctx = ArmContext(
        client=client,
        cfg=cfg,
        encoders=encoders,
        id_map=bundle.id_map,
        collection=collection,
        dataset=bundle.spec.name,
    )
    arm = REGISTRY["dense"]
    query_ids = sorted(bundle.eval_queries())[:N_QUERIES]
    records = [
        {
            "query_id": qid,
            "retrieved": arm(ctx, bundle.queries[qid], None).retrieved,
            "run_idx": 0,
        }
        for qid in query_ids
    ]
    client.close()
    score = evaluate_quality(
        bundle.eval_qrels(), build_run_dict(records), metrics=(PRIMARY_METRIC,)
    )[PRIMARY_METRIC]
    assert score is not None
    return score


def test_gate3_hnsw_build_variance(cfg, nfcorpus, encoders, controller):
    """Two independent builds of the same vectors must agree to within ~0.005 NDCG@10."""
    client = build_client(cfg)
    vectors = embed_corpus(cfg, nfcorpus, encoders, ["dense"])
    names = ["nfcorpus__gate3_a", "nfcorpus__gate3_b"]
    for name in names:
        create_collection(client, cfg, nfcorpus.spec, name, ["dense"])
        upsert_documents(client, cfg, nfcorpus, name, ["dense"], vectors)
    client.close()

    scores = [_score_dense(cfg, nfcorpus, encoders, name) for name in names]
    gap = abs(scores[0] - scores[1])

    cleanup = build_client(cfg)
    for name in names:
        cleanup.delete_collection(name)
    cleanup.close()

    print(f"HNSW build variance: {scores[0]:.5f} vs {scores[1]:.5f} -> gap {gap:.5f}")
    assert gap < THRESHOLD, (
        f"Gate 3 FAILED: two independent builds of the same dense vectors differ by "
        f"{gap:.5f} NDCG@10 (threshold {THRESHOLD}). Arm-to-arm differences smaller than "
        "this are not interpretable."
    )
