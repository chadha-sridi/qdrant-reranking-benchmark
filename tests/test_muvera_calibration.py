"""MUVERA calibration on real late-interaction embeddings.

MUVERA is a prefetch, so the question that matters is whether the exact max_sim top-k
survives into the candidate set at each sweep depth. This cannot be answered on synthetic
vectors: on isotropic random tokens the FDE cannot beat a plain summed-token dot product
at any parameter setting, so a synthetic test would either pass vacuously or fail
misleadingly.

This gate must pass before any `muvera_late_interaction_rescore` number is published. It
is marked slow because it needs the real late-interaction model.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.slow]

@pytest.fixture(scope="module")
def calibration(cfg, nfcorpus, encoders):
    from src.muvera import calibrate, calibration_inputs

    doc_tokens, query_tokens = calibration_inputs(cfg, nfcorpus, encoders)
    result = calibrate(
        encoders.muvera,
        doc_tokens,
        query_tokens,
        depths=tuple(cfg.depths_for("nfcorpus")),  # nfcorpus's own (deeper) sweep, not the global default
        truth_k=cfg.run.execution.final_limit,
    )
    payload = result.as_dict() | {
        "config": cfg.models.muvera.model_dump(),
        "late_interaction_model": cfg.models.late_interaction.name,
        "corpus": f"nfcorpus, all {len(doc_tokens)} documents",
        "queries": f"{len(query_tokens)} held-out warmup queries",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return result


def test_muvera_beats_bag_of_tokens(calibration):
    """The floor. Thousands of dimensions must buy something over a summed-token dot."""
    assert calibration.beats_bag_of_tokens, (
        "MUVERA calibration FAILED: the FDE does not recover the exact max_sim top-k any "
        f"better than a plain summed-token dot product "
        f"(FDE {calibration.recall_at} vs bag {calibration.bag_recall_at}). The muvera arm "
        f"would be storing {calibration.dim} dimensions per document for nothing. Retune "
        "k_sim / n_reps / d_proj in config/models.yaml, or exclude the arm explicitly."
    )


def test_muvera_prefetch_recall_is_usable_at_sweep_depths(cfg, calibration):
    """A prefetch that loses the true top-10 makes the whole depth curve meaningless."""
    deepest = max(cfg.depths_for("nfcorpus"))
    recall = calibration.recall_at[deepest]
    assert recall >= cfg.models.muvera.min_recall, (
        f"MUVERA calibration FAILED: at the deepest sweep depth ({deepest}) the FDE "
        f"prefetch recovers only {recall:.2%} of the exact max_sim top-"
        f"{cfg.run.execution.final_limit}. `muvera_late_interaction_rescore` would measure the FDE's "
        "recall ceiling rather than the value of late-interaction rescoring, and its curve would "
        "be uninterpretable. Retune the FDE or exclude the arm."
    )
