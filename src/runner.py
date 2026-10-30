"""Execution plan, interleaving, warmup, and resume.

Queries form the OUTER loop and units (arm, depth pairs) the INNER loop, so
environmental drift spreads across all arms instead of concentrating in whichever arm
ran last. Both loops are reshuffled with independent seeds — unit order fresh per
query, query order fresh per repeat — so no arm or query is systematically favored or
disadvantaged by a fixed position in the run.

Records append per query rather than per unit, because under this ordering a unit
does not complete until the final query.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .arms import REGISTRY, ArmContext, ArmSpec, resolve_arms
from .config import Config
from .datasets import LoadedDataset
from .embedding import Encoders
from .indexing import build_client
from .metrics import count_relevant
from .platform_info import stable_digest
from .storage import RawRecord, RawWriter, completed_keys, repair_truncated


@dataclass(frozen=True)
class Unit:
    arm: ArmSpec
    depth: int | None

    def __str__(self) -> str:
        return f"{self.arm.name}@{'-' if self.depth is None else self.depth}"


@dataclass
class RunSummary:
    dataset: str
    units: list[str]
    exclusions: dict[str, str]
    n_eval_queries: int
    n_records_written: int
    n_records_skipped: int
    plan_digest: str
    wall_seconds: float


def build_plan(cfg: Config, arms: Sequence[ArmSpec], dataset: str | None = None) -> list[Unit]:
    """All (arm, depth) units, shuffled with a fixed seed.

    Depths come from the dataset's own override when it has one, else the global sweep
    — the useful depth range depends on how many relevant documents a dataset
    actually has per query. An arm named in `execution.arm_depth_caps` runs that same
    ladder truncated at its cap.
    """
    unknown = sorted(set(cfg.run.execution.arm_depth_caps) - set(REGISTRY))
    if unknown:
        raise ValueError(
            f"arm_depth_caps names unknown arm(s): {', '.join(unknown)}. "
            f"Known arms: {', '.join(sorted(REGISTRY))}"
        )

    units: list[Unit] = []
    for arm in arms:
        if not arm.sweeps_depth:
            units.append(Unit(arm=arm, depth=None))
            continue
        depths = cfg.depths_for_arm(dataset, arm.name)
        if not depths:
            raise ValueError(
                f"arm {arm.name} has no depths left: cap "
                f"{cfg.run.execution.arm_depth_caps[arm.name]} excludes every depth in "
                f"{cfg.depths_for(dataset) if dataset else list(cfg.run.execution.depths)}"
            )
        units.extend(Unit(arm=arm, depth=depth) for depth in depths)

    rng = np.random.default_rng(cfg.run.seeds.plan_shuffle)
    order = rng.permutation(len(units))
    return [units[i] for i in order]


def plan_digest(cfg: Config, bundle: LoadedDataset, units: Sequence[Unit]) -> str:
    """Fingerprint of everything that must not change across a resume."""
    return stable_digest(
        {
            "dataset": bundle.spec.name,
            "corpus_digest": bundle.corpus_digest,
            "id_map_digest": bundle.id_map.digest,
            "eval_ids": bundle.eval_ids,
            "units": sorted(f"{u.arm.name}@{u.depth}" for u in units),
            "seeds": cfg.run.seeds.model_dump(),
            "execution": cfg.run.execution.model_dump(),
            "models": {
                "dense": cfg.models.dense.model_id,
                "bm25": cfg.models.bm25.model_dump(exclude={"reference"}),
                "late_interaction": cfg.models.late_interaction.name,
                "cross_encoder": cfg.models.cross_encoder.model_id,
                "muvera": cfg.models.muvera.model_dump(),
            },
            "qdrant_image": cfg.run.qdrant.image,
        }
    )


def _assert_resumable(path: Path, digest: str) -> None:
    """Refuse to mix results from two different configurations."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"plan_digest": digest}, indent=2))
        return
    existing = json.loads(path.read_text()).get("plan_digest")
    if existing != digest:
        raise RuntimeError(
            f"cannot resume: the existing results in {path.parent} were produced by a "
            f"different plan (digest {existing[:12]}... vs {digest[:12]}...). A config, "
            "seed, model, or dataset changed. Start a new results directory rather than "
            "mixing two configurations into one output."
        )


# ---------------------------------------------------------------------- execution


def warmup(
    bundle: LoadedDataset,
    ctx: ArmContext,
    units: Sequence[Unit],
    count: int,
    *,
    log: Callable[[str], None] = print,
) -> int:
    """Discarded warmup pass over the held-out warmup queries.

    These queries are never evaluated, so no reported query is served from a cache this
    pass warmed. Nothing is trained on them.
    """
    query_ids = bundle.warmup_ids[:count]
    if not query_ids:
        log("  warmup skipped: no warmup queries held out")
        return 0
    done = 0
    for query_id in query_ids:
        text = bundle.queries[query_id]
        ctx.pointwise_scores = {} if ctx.cfg.run.execution.share_pointwise_scores else None
        for unit in units:
            try:
                unit.arm(ctx, text, unit.depth)
                done += 1
            except Exception as exc:  # a warmup failure is a real failure; surface it
                raise RuntimeError(f"warmup failed on {unit}: {exc}") from exc
    log(f"  warmup: {done} discarded calls over {len(query_ids)} queries")
    return done


def run_dataset(
    cfg: Config,
    bundle: LoadedDataset,
    encoders: Encoders,
    collection: str,
    *,
    include_arms: Sequence[str] | None = None,
    resume: bool = True,
    log: Callable[[str], None] = print,
) -> RunSummary:
    started = time.monotonic()
    dataset = bundle.spec.name
    arms, exclusions = resolve_arms(cfg, dataset, include=include_arms)
    if not arms:
        raise RuntimeError(f"{dataset}: no runnable arms; exclusions={exclusions}")

    units = build_plan(cfg, arms, dataset)
    digest = plan_digest(cfg, bundle, units)
    raw_root = cfg.paths["raw"]
    _assert_resumable(raw_root / f"{dataset}__plan.json", digest)

    client = build_client(cfg, for_queries=True)
    try:
        # The collection's MUVERA vectors were centered on this corpus's mean, so the
        # query side must use the identical vector. Cached by the build that wrote them.
        muvera_mean = None
        if any("muvera" in arm.requires for arm in arms):
            from .indexing import muvera_token_mean

            muvera_mean = muvera_token_mean(cfg, bundle, encoders)
        ctx = ArmContext(
            client=client,
            cfg=cfg,
            encoders=encoders,
            id_map=bundle.id_map,
            collection=collection,
            dataset=dataset,
            muvera_mean=muvera_mean,
        )

        if resume:
            repaired = repair_truncated(raw_root, dataset)
            if repaired:
                log(f"  repaired {len(repaired)} truncated raw file(s) from a previous crash")
        done = completed_keys(raw_root, dataset) if resume else set()
        if done:
            log(f"  resume: {len(done)} records already on disk")

        warmup(bundle, ctx, units, cfg.run.queries.warmup_count, log=log)

        eval_queries = bundle.eval_queries()
        query_ids = sorted(eval_queries)  # canonical order; only used as the base to permute
        repeats = cfg.run.execution.repeats
        written = skipped = 0

        log(f"  executing {len(units)} units x {len(query_ids)} queries x {repeats} runs")
        # Two independent reshuffles protect two different things:
        #  - unit order, per (query, repeat): stops whichever arm goes first from always
        #    paying a query's cold index-region cost while later arms benefit from pages
        #    it just warmed.
        #  - query order, per repeat: without this, whichever queries sort last are
        #    ALWAYS measured under the same point in the run's drift (thermal, memory
        #    pressure) on every repeat, biasing per-query and percentile stats even
        #    though arm-vs-arm comparisons are already protected by the unit reshuffle.
        # A large constant offset keeps the two seed derivations from ever colliding.
        with RawWriter(raw_root, flush_every=cfg.run.execution.flush_every) as writer:
            # Queries outer, units inner: environmental drift spreads across all arms.
            for run_idx in range(repeats):
                query_order_seed = cfg.run.seeds.plan_shuffle + 1_000_000 + run_idx
                query_order = np.random.default_rng(query_order_seed).permutation(len(query_ids))
                for q_index, qi in enumerate(query_order):
                    query_id = query_ids[qi]
                    text = eval_queries[query_id]
                    ctx.pointwise_scores = {} if cfg.run.execution.share_pointwise_scores else None
                    order_seed = cfg.run.seeds.plan_shuffle + run_idx * len(query_ids) + q_index
                    order = np.random.default_rng(order_seed).permutation(len(units))
                    for unit_index in order:
                        unit = units[unit_index]
                        key = (unit.arm.name, unit.depth, query_id, run_idx)
                        if key in done:
                            skipped += 1
                            continue
                        output = unit.arm(ctx, text, unit.depth)
                        extra: dict = {}
                        if unit.arm.license_note:
                            extra["license_note"] = unit.arm.license_note
                        if output.pool_doc_ids is not None:
                            extra["n_relevant_in_pool"] = count_relevant(
                                bundle.qrels.get(query_id, {}), output.pool_doc_ids
                            )
                        writer.write(
                            RawRecord(
                                query_id=query_id,
                                arm=unit.arm.name,
                                dataset=dataset,
                                depth=unit.depth,
                                run_idx=run_idx,
                                retrieved=output.retrieved,
                                timers=output.timers,
                                extra=extra,
                            )
                        )
                        written += 1
    finally:
        client.close()

    return RunSummary(
        dataset=dataset,
        units=[str(u) for u in units],
        exclusions=exclusions,
        n_eval_queries=len(query_ids),
        n_records_written=written,
        n_records_skipped=skipped,
        plan_digest=digest,
        wall_seconds=time.monotonic() - started,
    )
