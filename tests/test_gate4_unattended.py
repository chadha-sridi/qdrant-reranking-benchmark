"""Gate 4 — unattended run.

The pipeline runs end to end without intervention, resumes idempotently, and aggregates
into the published artefacts.

Deliberately scoped to a small subset — a handful of queries at two depths, writing to a
temporary results directory. What this gate checks is *mechanism*: that the plan executes,
that resume skips exactly what is already on disk, and that aggregation produces its
outputs. None of that needs the full sweep, and running the full sweep here would take
hours and be a large slice of the benchmark rather than a check on it.
"""

from __future__ import annotations

import dataclasses

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.slow]

N_QUERIES = 5
DEPTHS = [10, 25]


@pytest.fixture
def scoped(cfg, nfcorpus, tmp_path):
    """cfg + bundle trimmed to a few queries and two depths, isolated on disk."""
    scoped_cfg = cfg.model_copy(
        update={
            "run": cfg.run.model_copy(
                update={
                    "paths": cfg.run.paths.model_copy(
                        update={
                            "raw": "gate4/raw",
                            "agg": "gate4/agg",
                            "queries": "gate4/queries",
                            # Absolute, so redirecting `root` below moves the run outputs
                            # without moving the embedding cache with them. The cache is
                            # keyed by model, corpus and platform, so sharing it is safe,
                            # and re-embedding the corpus per test would dominate this
                            # gate's runtime while checking nothing it exists to check.
                            "cache": str(cfg.paths["cache"]),
                        }
                    ),
                    # One repeat is enough to exercise the loop; resume is what is
                    # under test, not percentile stability.
                    "execution": cfg.run.execution.model_copy(update={"repeats": 1}),
                }
            ),
            "root": tmp_path,
        }
    )
    # `with_depths_override` rather than setting execution.depths: NFCorpus carries its
    # own depth override, which a plain global update would silently lose to.
    scoped_cfg = scoped_cfg.with_depths_override(DEPTHS)
    for sub in ("raw", "agg", "queries"):
        (tmp_path / "gate4" / sub).mkdir(parents=True, exist_ok=True)

    bundle = dataclasses.replace(nfcorpus, eval_ids=list(nfcorpus.eval_ids)[:N_QUERIES])
    return scoped_cfg, bundle


@pytest.mark.timeout(1800)
def test_gate4_pipeline_runs_and_aggregates(scoped, encoders, nfcorpus_collection):
    from src.analysis import analyse_dataset
    from src.runner import run_dataset

    cfg, bundle = scoped
    summary = run_dataset(
        cfg, bundle, encoders, nfcorpus_collection, resume=True, log=lambda msg: None
    )

    expected = len(summary.units) * summary.n_eval_queries * cfg.run.execution.repeats
    accounted = summary.n_records_written + summary.n_records_skipped
    assert accounted == expected, (
        f"Gate 4 FAILED: expected {expected} records ({len(summary.units)} units x "
        f"{summary.n_eval_queries} queries x {cfg.run.execution.repeats} runs) but "
        f"accounted for {accounted}"
    )

    written = analyse_dataset(cfg, bundle)
    assert "aggregate" in written and written["aggregate"].exists()
    assert "gain_vs_depth" in written, "the depth curve is the primary deliverable"


@pytest.mark.timeout(1800)
def test_gate4_resume_is_idempotent(scoped, encoders, nfcorpus_collection):
    """A second run must write nothing new — otherwise a resumed run double-counts."""
    from src.runner import run_dataset

    cfg, bundle = scoped
    first = run_dataset(cfg, bundle, encoders, nfcorpus_collection, resume=True, log=lambda m: None)
    second = run_dataset(
        cfg, bundle, encoders, nfcorpus_collection, resume=True, log=lambda m: None
    )

    assert second.n_records_written == 0, (
        f"Gate 4 FAILED: a resumed run wrote {second.n_records_written} additional records "
        "for work already on disk. Aggregates would double-count."
    )
    assert second.n_records_skipped == first.n_records_written + first.n_records_skipped
