"""Aggregation from raw records to the published CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .arms import REGISTRY
from .config import Config
from .datasets import LoadedDataset
from .metrics import PRIMARY_METRIC, aggregate, build_run_dict, compare_arms, gain_versus_depth
from .storage import read_raw


def _arm_metadata() -> dict[str, dict]:
    return {
        name: {
            "requires": arm.requires,
            "group": arm.group,
            "baseline": arm.baseline,
            "client_side_rerank": arm.client_side_rerank,
            "license_note": arm.license_note,
            "sweeps_depth": arm.sweeps_depth,
        }
        for name, arm in REGISTRY.items()
    }


def _memory_lookup(cfg: Config, dataset: str) -> dict[str, dict]:
    """Qdrant's accounting for this dataset's collection, keyed by representation.

    The record on disk describes the whole collection; the aggregate needs it per
    representation, so an arm's row can carry the cost of what that arm actually reads.
    Records from runs before BM25 have only `vectors`, which is the same mapping for
    every representation they had.
    """
    path = cfg.paths["agg"] / f"{dataset}__memory.json"
    if not path.exists():
        return {}
    memory = json.loads(path.read_text())
    return dict(memory.get("representations") or memory.get("vectors") or {})


class IncompleteRunError(RuntimeError):
    pass


def check_completeness(cfg: Config, bundle: LoadedDataset, records: list[dict]) -> dict[str, dict]:
    """For every (arm, depth) unit present on disk, is every (query, repeat) there?

    Unchecked, an interrupted, corrupted, or partially-resumed run aggregates into
    normal-looking scores and reads as a finished benchmark result with no signal that
    anything was missing. This does not need to know which arms were
    *intended* — an arm nobody ran simply has no file, which is a scoping choice, not
    incompleteness. It flags a unit that has SOME records but not all of them, which
    can only mean a crash, a corrupt line, or an interrupted resume.
    """
    eval_ids = set(bundle.eval_ids)
    repeats = cfg.run.execution.repeats
    expected_per_unit = len(eval_ids) * repeats

    seen: dict[tuple[str, object], set[tuple[str, int]]] = {}
    for record in records:
        key = (record["arm"], record.get("depth"))
        qid = record["query_id"]
        if qid not in eval_ids:
            continue  # a stale record from a since-changed query sample; not this check's job
        seen.setdefault(key, set()).add((qid, int(record.get("run_idx", 0))))

    incomplete: dict[str, dict] = {}
    for (arm, depth), keys in seen.items():
        if len(keys) < expected_per_unit:
            incomplete[f"{arm}@{depth}"] = {
                "expected": expected_per_unit,
                "found": len(keys),
                "missing": expected_per_unit - len(keys),
            }
    return incomplete


def analyse_dataset(cfg: Config, bundle: LoadedDataset, *, allow_partial: bool = False) -> dict[str, Path]:
    """Write the aggregate CSV, the depth curve, and the significance table.

    Refuses on incomplete data unless `allow_partial=True` — pass it explicitly to
    inspect an in-progress run; never pass it for a result you intend to trust or
    publish.
    """
    dataset = bundle.spec.name
    records = read_raw(cfg.paths["raw"], dataset)
    if not records:
        raise RuntimeError(f"no raw records found for {dataset} in {cfg.paths['raw']}")

    incomplete = check_completeness(cfg, bundle, records)
    if incomplete and not allow_partial:
        lines = "\n".join(
            f"  {unit}: {info['found']}/{info['expected']} records "
            f"({info['missing']} missing)"
            for unit, info in sorted(incomplete.items())
        )
        raise IncompleteRunError(
            f"{dataset}: {len(incomplete)} unit(s) have partial data, so this cannot be "
            f"aggregated as a finished result:\n{lines}\n"
            "Resume the run to fill the gaps, or pass allow_partial=True "
            "(--allow-partial on the CLI) to inspect it anyway."
        )

    qrels = bundle.eval_qrels()
    agg_dir = cfg.paths["agg"]
    agg_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    agg = aggregate(
        records,
        qrels,
        memory_by_representation=_memory_lookup(cfg, dataset),
        arm_metadata=_arm_metadata(),
    )
    agg_path = agg_dir / f"{dataset}__aggregate.csv"
    agg.to_csv(agg_path, index=False)
    written["aggregate"] = agg_path

    curve = gain_versus_depth(agg)
    if not curve.empty:
        curve_path = agg_dir / f"{dataset}__gain_vs_depth.csv"
        curve.to_csv(curve_path, index=False)
        written["gain_vs_depth"] = curve_path

    sig_path = _write_significance(cfg, dataset, records, qrels, agg_dir)
    if sig_path is not None:
        written["significance"] = sig_path

    return written


def _write_significance(
    cfg: Config,
    dataset: str,
    records: list[dict],
    qrels: dict[str, dict[str, int]],
    agg_dir: Path,
) -> Path | None:
    """Pairwise significance at EVERY depth actually swept — never at a chosen "best" one.

    Picking each arm's best-scoring depth by NDCG@10 on the eval set and then
    significance-testing those runs on that SAME eval set is a textbook
    selection-bias/winner's-curse setup: it inflates apparent winners and invalidates
    the reported p-values. There is no separate tuning split to select a depth on
    instead: nothing here trains, so holding queries out would cost statistical power
    for no gain.

    So there is no depth selection at all: every depth actually swept gets its own
    comparison, non-sweeping arms (constant across depth) participate in all of them,
    and the Bonferroni correction is applied ONCE across every (arm_a, arm_b, depth)
    comparison made — not just across arm pairs at one depth — so the correction
    reflects the real number of hypotheses tested.
    """
    frame = pd.DataFrame(records)
    if frame.empty:
        return None

    runs_by_arm_depth: dict[tuple[str, object], dict[str, dict[str, float]]] = {}
    for (arm, depth), group in frame.groupby(["arm", "depth"], dropna=False):
        run = build_run_dict(group.to_dict("records"), run_idx=0)
        if run:
            depth_value = None if pd.isna(depth) else int(depth)
            runs_by_arm_depth[(arm, depth_value)] = run

    depths = sorted({d for _, d in runs_by_arm_depth if d is not None})
    non_sweeping = {arm: run for (arm, d), run in runs_by_arm_depth.items() if d is None}

    all_rows: list[pd.DataFrame] = []
    per_depth_queries: dict[int, int] = {}
    for depth in depths:
        runs = dict(non_sweeping)
        for (arm, d), run in runs_by_arm_depth.items():
            if d == depth:
                runs[arm] = run
        if len(runs) < 2:
            continue
        # A paired test needs one shared query set at this depth.
        common = set.intersection(*(set(r) for r in runs.values()))
        if len(common) < 2:
            continue
        runs = {name: {qid: r[qid] for qid in common} for name, r in runs.items()}
        result = compare_arms(qrels, runs, metric=PRIMARY_METRIC)
        pairs = result.pairs.drop(columns=["p_bonferroni", "significant_bonferroni"])
        pairs.insert(2, "depth", depth)
        all_rows.append(pairs)
        per_depth_queries[depth] = len(common)

    if not all_rows:
        return None

    combined = pd.concat(all_rows, ignore_index=True)
    n_total = len(combined)
    combined["p_bonferroni"] = (combined["p_value"] * n_total).clip(upper=1.0)
    combined["significant_bonferroni"] = combined["p_bonferroni"] < 0.05

    path = agg_dir / f"{dataset}__significance.csv"
    combined.to_csv(path, index=False)

    (agg_dir / f"{dataset}__significance_meta.json").write_text(
        json.dumps(
            {
                "metric": PRIMARY_METRIC,
                "alpha": 0.05,
                "n_comparisons": n_total,
                "bonferroni_factor": n_total,
                "depths_compared": depths,
                "n_queries_per_depth": per_depth_queries,
                "note": (
                    "Every depth actually swept is compared separately — no depth is "
                    "selected by its own observed score. The Bonferroni "
                    "factor is the total number of (arm_a, arm_b, depth) rows in this "
                    "file, not just the arm-pair count at one depth."
                ),
            },
            indent=2,
        )
    )
    return path


def combine_datasets(cfg: Config) -> Path | None:
    """Concatenate every per-dataset aggregate into one published table."""
    agg_dir = cfg.paths["agg"]
    parts = sorted(agg_dir.glob("*__aggregate.csv"))
    if not parts:
        return None
    frames = [pd.read_csv(path) for path in parts]
    combined = pd.concat(frames, ignore_index=True)
    out = agg_dir / "all_datasets__aggregate.csv"
    combined.to_csv(out, index=False)
    return out
