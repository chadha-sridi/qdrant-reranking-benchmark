"""Gate 2 — arms are distinct.

An arm that silently falls back to its prefetch ranking looks exactly like "reranking
didn't help", which is the wrong conclusion drawn from a bug.

The gate takes no exemptions. Every arm reranks the dense baseline (or a prefetch
derived from it) with a genuinely different mechanism, so if any pair ever needs an
exemption that is itself worth investigating rather than suppressing.

Note `quantized_rescore` vs `dense`: at low bit widths TurboQuant plus full-precision
rescoring can legitimately reproduce the baseline ranking on easy queries. That is the
arm working, not failing — but reproducing it on EVERY probe query would mean the
quantized index is not actually being consulted, which is what this gate catches.
"""

from __future__ import annotations

import itertools

import pytest

from src.arms import REGISTRY, resolve_arms

pytestmark = [pytest.mark.gate, pytest.mark.slow]

N_PROBE_QUERIES = 5


def _top_ids(output) -> tuple[str, ...]:
    return tuple(doc_id for doc_id, _ in output.retrieved)


def test_gate2_arms_return_different_results(cfg, nfcorpus, arm_context):
    arms, exclusions = resolve_arms(cfg, "nfcorpus")
    query_ids = sorted(nfcorpus.eval_queries())[:N_PROBE_QUERIES]
    assert query_ids, "no evaluation queries available"

    results: dict[str, list[tuple[str, ...]]] = {}
    for arm in arms:
        depth = cfg.depths_for("nfcorpus")[-1] if arm.sweeps_depth else None
        results[arm.name] = [
            _top_ids(arm(arm_context, nfcorpus.queries[qid], depth)) for qid in query_ids
        ]

    identical = [
        f"{a} == {b}"
        for a, b in itertools.combinations(sorted(results), 2)
        if results[a] == results[b]
    ]

    print("Arms excluded from this dataset:")
    for arm, reason in sorted(exclusions.items()):
        print(f"  {arm}: {reason}")

    assert not identical, (
        "Gate 2 FAILED: these arm pairs returned identical result sets on all "
        f"{len(query_ids)} probe queries, which usually means an arm silently fell back "
        f"to its prefetch ranking: {identical}"
    )


def test_gate2_every_declared_arm_is_accounted_for(cfg):
    """Every registered arm is either runnable by default or excluded with a stated reason."""
    arms, exclusions = resolve_arms(cfg, "nfcorpus")
    accounted = {arm.name for arm in arms} | set(exclusions)
    missing = set(REGISTRY) - accounted
    assert not missing, f"arms neither run nor excluded: {sorted(missing)}"
    assert len(REGISTRY) == 5, f"expected 5 registered arms, found {len(REGISTRY)}"
