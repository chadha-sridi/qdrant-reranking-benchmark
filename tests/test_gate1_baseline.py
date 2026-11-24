"""Gate 1 — each baseline reproduces its published NDCG.

`dense` and `bm25` on NFCorpus must each match a published NDCG@10 within the tolerance
pinned in config/models.yaml. If either does not, something is wrong in qrels parsing,
ID mapping, document text, the encoder or text-index setup, or the metric call, and every number
measured against that baseline is invalid. Nothing else matters until this passes.

Run against the FULL 323-query NFCorpus test set, not the subsample: the published figure
is a full-set number and a 250-query subsample is not comparable to it.
This is the only place a metric is computed outside the validation split.
"""

from __future__ import annotations

import pytest

from src.arms import REGISTRY, ArmContext
from src.indexing import build_client
from src.metrics import build_run_dict, evaluate_quality

pytestmark = [pytest.mark.gate, pytest.mark.slow]


BASELINES = ["dense", "bm25"]


@pytest.mark.parametrize("baseline", BASELINES)
def test_gate1_reference_is_verified(cfg, baseline):
    """A reference number nobody checked makes the gate either vacuous or falsely failing."""
    ref = getattr(cfg.models, baseline).reference
    if not ref.verified:
        pytest.fail(
            f"config/models.yaml pins {baseline}.reference.{ref.metric}={ref.value} on "
            f"{ref.dataset} with verified: false. Confirm it against {ref.source} and set "
            "verified: true before trusting Gate 1 or publishing anything downstream."
        )


@pytest.mark.parametrize("baseline", BASELINES)
def test_gate1_baseline_matches_published_ndcg(
    cfg, nfcorpus, encoders, nfcorpus_collection, baseline
):
    ref = getattr(cfg.models, baseline).reference
    assert ref.dataset == "nfcorpus", "Gate 1 is defined against nfcorpus"

    client = build_client(cfg, for_queries=True)
    ctx = ArmContext(
        client=client,
        cfg=cfg,
        encoders=encoders,
        id_map=nfcorpus.id_map,
        collection=nfcorpus_collection,
        dataset="nfcorpus",
    )
    arm = REGISTRY[baseline]

    # Full query set, judged queries only.
    qrels = _all_qrels(nfcorpus)
    judged = {qid: text for qid, text in nfcorpus.all_queries.items() if qid in qrels}
    assert len(judged) > 300, f"expected ~323 judged nfcorpus queries, got {len(judged)}"

    records = []
    for query_id, text in judged.items():
        output = arm(ctx, text, None)
        records.append({"query_id": query_id, "retrieved": output.retrieved, "run_idx": 0})
    client.close()

    run = build_run_dict(records)
    scores = evaluate_quality(qrels, run, metrics=(ref.metric,))
    observed = scores[ref.metric]

    assert observed is not None, "metric did not evaluate"
    delta = abs(observed - ref.value)
    assert delta <= ref.tolerance, (
        f"Gate 1 FAILED for {baseline}: {ref.metric}={observed:.4f} on {len(run)} nfcorpus queries, "
        f"published {ref.value:.4f} (delta {delta:.4f} > tolerance {ref.tolerance}). "
        "Suspect qrels parsing, ID mapping, document text construction, or the metric "
        "call — not the model. Every downstream number is invalid until this passes."
    )


def _all_qrels(bundle) -> dict[str, dict[str, int]]:
    """Reload the unsubsampled qrels for the full-set comparison."""
    from datasets import load_dataset as hf_load

    spec = bundle.spec
    out: dict[str, dict[str, int]] = {}
    for row in hf_load(spec.hf_qrels, split=spec.qrels_split):
        out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    # Keep only judged docs that exist in the indexed corpus.
    known = set(bundle.id_map.to_point)
    return {
        qid: {did: rel for did, rel in judgements.items() if did in known}
        for qid, judgements in out.items()
        if any(did in known for did in judgements)
    }
