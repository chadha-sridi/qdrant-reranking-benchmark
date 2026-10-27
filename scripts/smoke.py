"""Quick end-to-end smoke test: nfcorpus, dense + late_interaction_rescore.

Trims query count, depth sweep, and repeats so the whole pipeline finishes in a couple
minutes. Writes to results_smoke/ so it can never collide with a real run's output.

Usage:
    .venv/bin/python scripts/smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import analyse_dataset
from src.config import load_config
from src.datasets import load_dataset_bundle
from src.embedding import build_encoders
from src.indexing import build_collection
from src.qdrant_control import QdrantController
from src.runner import run_dataset


def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}")

    cfg = load_config()
    smoke_run = cfg.run.model_copy(
        update={
            "queries": cfg.run.queries.model_copy(update={"cap": 20, "warmup_count": 2}),
            "execution": cfg.run.execution.model_copy(update={"repeats": 1, "depths": [10, 50]}),
            "paths": cfg.run.paths.model_copy(
                update={
                    "results": "results_smoke",
                    "raw": "results_smoke/raw",
                    "agg": "results_smoke/agg",
                    "queries": "results_smoke/queries",
                    "manifest": "results_smoke/manifest.json",
                }
            ),
        }
    )
    cfg = cfg.model_copy(update={"run": smoke_run})
    log(
        f"smoke config: cap={cfg.run.queries.cap} depths={cfg.run.execution.depths} "
        f"repeats={cfg.run.execution.repeats}"
    )

    controller = QdrantController(cfg)
    controller.up()
    log(f"qdrant up (launcher={controller._launcher})")

    encoders = build_encoders(cfg)
    log("encoders built (models load lazily on first use)")

    bundle = load_dataset_bundle(cfg, "nfcorpus")
    log(
        f"nfcorpus loaded: {len(bundle.documents)} docs, {len(bundle.queries)} sampled "
        f"queries ({len(bundle.eval_ids)} evaluated / {len(bundle.warmup_ids)} warmup)"
    )

    # Only dense + late_interaction vectors are needed for these arms — skips muvera
    # embedding entirely, which is most of the time cost.
    collection = build_collection(
        cfg, bundle, encoders, controller, representations=["dense", "late_interaction"]
    )
    log(f"collection built: {collection}")

    summary = run_dataset(
        cfg, bundle, encoders, collection,
        include_arms=["dense", "late_interaction_rescore"],
        resume=True,
        log=log,
    )
    log(f"RUN DONE: wrote={summary.n_records_written} skipped={summary.n_records_skipped}")
    print("exclusions:", summary.exclusions)

    written = analyse_dataset(cfg, bundle)
    log("ANALYSIS DONE")
    for label, path in written.items():
        print(f"  [{label}] {path}")

    log("TOTAL")


if __name__ == "__main__":
    main()
