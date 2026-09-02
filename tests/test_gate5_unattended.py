"""Gate 5 — unattended run.

The full pipeline completes on NFCorpus, all arms, all depths, with no intervention: run,
resume cleanly, and aggregate into the published artefacts.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.slow]


@pytest.mark.timeout(7200)
def test_gate5_full_pipeline_on_nfcorpus(cfg, nfcorpus, encoders, nfcorpus_collection, tmp_path):
    from src.analysis import analyse_dataset
    from src.runner import run_dataset

    summary = run_dataset(
        cfg, nfcorpus, encoders, nfcorpus_collection, resume=True, log=lambda msg: print(msg)
    )

    assert summary.n_records_written + summary.n_records_skipped > 0
    expected = (
        len(summary.units) * summary.n_eval_queries * cfg.run.execution.repeats
    )
    assert summary.n_records_written + summary.n_records_skipped == expected, (
        f"Gate 5 FAILED: expected {expected} records "
        f"({len(summary.units)} units x {summary.n_eval_queries} queries x "
        f"{cfg.run.execution.repeats} runs) but accounted for "
        f"{summary.n_records_written + summary.n_records_skipped}"
    )

    written = analyse_dataset(cfg, nfcorpus)
    assert "aggregate" in written and written["aggregate"].exists()
    assert "gain_vs_depth" in written, "the depth curve is the primary deliverable"


@pytest.mark.timeout(3600)
def test_gate5_resume_is_idempotent(cfg, nfcorpus, encoders, nfcorpus_collection):
    """A second run must write nothing new — otherwise a resumed run double-counts."""
    from src.runner import run_dataset

    first = run_dataset(cfg, nfcorpus, encoders, nfcorpus_collection, resume=True, log=lambda m: None)
    second = run_dataset(cfg, nfcorpus, encoders, nfcorpus_collection, resume=True, log=lambda m: None)

    assert second.n_records_written == 0, (
        f"Gate 5 FAILED: a resumed run wrote {second.n_records_written} additional records "
        "for work already on disk. Aggregates would double-count."
    )
    assert second.n_records_skipped == first.n_records_written + first.n_records_skipped
