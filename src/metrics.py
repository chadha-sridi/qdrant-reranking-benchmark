"""Quality metrics via ranx, latency percentiles, and aggregation.

Aggregation happens only here, at analysis time, from the raw per-query records.

Two rules that shape everything below:
  * Quality comes from run_idx 0 only. Retrieval is deterministic given a fixed index,
    so repeated passes return identical lists; counting each one would inflate a
    significance test's sample size.
  * QUALITY_METRICS is ndcg@10 and recall@10 only; both are always well-defined at
    every configured depth (depths never go below final_limit).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIMARY_METRIC = "ndcg@10"
QUALITY_METRICS = ("ndcg@10", "recall@10")
LATENCY_COMPONENTS = (
    "t_embed_ms",
    "t_embed_dense_ms",
    "t_embed_late_interaction_ms",
    "t_embed_muvera_ms",
    "t_qdrant_ms",
    "t_qdrant_server_ms",
    "t_qdrant_overhead_ms",
    "t_rerank_ms",
    "t_total_ms",
)
# p99 is not reported: with one pass there is one latency sample per evaluated query,
# so a p99 would rest on a handful of points and read as false precision.
PERCENTILES = (50, 95)


class MetricsError(RuntimeError):
    pass


# ------------------------------------------------------------------------ quality


def count_relevant(qrels_for_query: dict[str, int], doc_ids: Sequence[str]) -> int:
    """How many of `doc_ids` are relevant per this query's qrels (grade > 0)."""
    return sum(1 for doc_id in doc_ids if qrels_for_query.get(doc_id, 0) > 0)


def pool_recall(
    records: Sequence[dict], qrels: dict[str, dict[str, int]], *, run_idx: int = 0
) -> float | None:
    """Mean, across queries, of (relevant docs in the prefetch pool / relevant docs total).

    The ceiling on `recall@10`: what fraction of a query's relevant documents were even
    available to the reranker at this depth, before rescoring ever ran. None where no
    record in `records` exposes a pool (`dense` has none); a query with zero relevant
    qrels is excluded, the same convention `recall@10` uses.
    """
    ratios = []
    for r in records:
        if int(r.get("run_idx", 0)) != run_idx:
            continue
        pool_count = r.get("n_relevant_in_pool")
        if pool_count is None:
            continue
        judged = qrels.get(r["query_id"], {})
        total_relevant = count_relevant(judged, judged.keys())
        if total_relevant == 0:
            continue
        ratios.append(pool_count / total_relevant)
    return sum(ratios) / len(ratios) if ratios else None


def build_run_dict(records: Sequence[dict], *, run_idx: int = 0) -> dict[str, dict[str, float]]:
    """ranx Run mapping from raw records: query_id -> {doc_id: score}."""
    run: dict[str, dict[str, float]] = {}
    for record in records:
        if int(record.get("run_idx", 0)) != run_idx:
            continue
        scores = {doc_id: float(score) for doc_id, score in record["retrieved"]}
        if not scores:
            continue
        run[record["query_id"]] = scores
    return run


def evaluate_quality(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    metrics: Sequence[str] = QUALITY_METRICS,
    depth: int | None = None,
) -> dict[str, float | None]:
    """Evaluate one run. Returns None for metrics the depth cannot support."""
    from ranx import Qrels, Run, evaluate

    if not run:
        return {m: None for m in metrics}

    applicable = [m for m in metrics if _metric_supported(m, depth)]
    # Restrict qrels to the evaluated queries so ranx does not count unanswered
    # queries as zeros and silently penalise a partial resume.
    scoped_qrels = {qid: rel for qid, rel in qrels.items() if qid in run}
    missing = set(run) - set(scoped_qrels)
    if missing:
        raise MetricsError(
            f"{len(missing)} evaluated queries have no qrels entry, e.g. {sorted(missing)[:3]}"
        )

    scores = evaluate(
        Qrels(scoped_qrels),
        Run(run),
        list(applicable),
        make_comparable=True,
    )
    if isinstance(scores, float):  # ranx returns a bare float for a single metric
        scores = {applicable[0]: scores}

    out: dict[str, float | None] = {m: None for m in metrics}
    for metric in applicable:
        out[metric] = float(scores[metric])
    return out


def _metric_supported(metric: str, depth: int | None) -> bool:
    if depth is None:
        return True
    cutoff = _metric_cutoff(metric)
    return cutoff is None or cutoff <= depth


def _metric_cutoff(metric: str) -> int | None:
    if "@" not in metric:
        return None
    try:
        return int(metric.split("@", 1)[1])
    except ValueError:
        return None


# ------------------------------------------------------------------- significance


@dataclass
class ComparisonResult:
    metric: str
    pairs: pd.DataFrame  # arm_a, arm_b, p_value, p_bonferroni, significant_*
    n_pairs: int
    n_queries: int
    stat_test: str
    alpha: float


def compare_arms(
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, dict[str, float]]],
    *,
    metric: str = PRIMARY_METRIC,
    alpha: float = 0.05,
    stat_test: str = "fisher",
) -> ComparisonResult:
    """Pairwise significance across arms, with an explicit Bonferroni correction.

    Uses ranx.compare at max_p=0.05. With this many arms the
    uncorrected p-values are misleading, so both the raw and corrected verdicts are
    reported. All arms share one query set, which is what makes the
    paired test valid.
    """
    from ranx import Qrels, Run, compare

    names = sorted(runs)
    if len(names) < 2:
        raise MetricsError("need at least two arms to compare")

    query_sets = {name: frozenset(run) for name, run in runs.items()}
    if len({*query_sets.values()}) != 1:
        sizes = {name: len(q) for name, q in query_sets.items()}
        raise MetricsError(
            f"arms do not share an identical query set, so a paired test is invalid: {sizes}"
        )

    shared = set(next(iter(query_sets.values())))
    scoped_qrels = {qid: rel for qid, rel in qrels.items() if qid in shared}

    report = compare(
        qrels=Qrels(scoped_qrels),
        runs=[Run(runs[name], name=name) for name in names],
        metrics=[metric],
        max_p=alpha,
        stat_test=stat_test,
        make_comparable=True,
    )

    n_pairs = len(names) * (len(names) - 1) // 2
    rows = []
    for a, b in itertools.combinations(names, 2):
        p_value = _extract_p(report, a, b, metric)
        p_corrected = None if p_value is None else min(1.0, p_value * n_pairs)
        rows.append(
            {
                "metric": metric,
                "arm_a": a,
                "arm_b": b,
                "p_value": p_value,
                "p_bonferroni": p_corrected,
                "significant_raw": None if p_value is None else p_value < alpha,
                "significant_bonferroni": None if p_corrected is None else p_corrected < alpha,
            }
        )

    return ComparisonResult(
        metric=metric,
        pairs=pd.DataFrame(rows),
        n_pairs=n_pairs,
        n_queries=len(shared),
        stat_test=stat_test,
        alpha=alpha,
    )


def _extract_p(report, arm_a: str, arm_b: str, metric: str) -> float | None:
    """Pull a pairwise p-value out of a ranx `Report`.

    `report.comparisons` is a `FrozensetDict` keyed by `frozenset({arm_a, arm_b})`,
    NOT a plain dict — `isinstance(table, dict)` on it is always False, so any lookup
    gated on that check returns `None` for every pair with no error at all. Verified
    directly against the installed ranx version:
    `report.comparisons[frozenset(['a','b'])]` -> `{'ndcg@10': {'p_value': ..., 'significant': ...}}`.
    """
    entry = report.comparisons.get(frozenset({arm_a, arm_b}))
    if not entry:
        return None
    value = entry.get(metric)
    if isinstance(value, dict):
        value = value.get("p_value")
    return float(value) if value is not None else None


# ------------------------------------------------------------------------ latency


def latency_percentiles(records: Sequence[dict]) -> dict[str, float | None]:
    """p50/p95 per component and for the total, over every timing run."""
    out: dict[str, float | None] = {}
    for component in LATENCY_COMPONENTS:
        values = np.asarray(
            [float(r[component]) for r in records if r.get(component) is not None],
            dtype=np.float64,
        )
        for pct in PERCENTILES:
            key = f"{component}_p{pct}"
            out[key] = float(np.percentile(values, pct)) if values.size else None
        out[f"{component}_mean"] = float(values.mean()) if values.size else None
    out["n_latency_samples"] = float(len([r for r in records if r.get("t_total_ms") is not None]))
    return out


# --------------------------------------------------------------------- aggregation


def aggregate(
    records: Sequence[dict],
    qrels: dict[str, dict[str, int]],
    *,
    memory_by_representation: dict[str, dict] | None = None,
    arm_metadata: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """One row per (dataset, arm, depth): quality, latency percentiles, memory."""
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    frame["depth"] = frame["depth"].where(frame["depth"].notna(), None)
    rows = []

    for (dataset, arm, depth), group in frame.groupby(["dataset", "arm", "depth"], dropna=False):
        group_records = group.to_dict("records")
        depth_value = (
            None
            if depth is None or (isinstance(depth, float) and math.isnan(depth))
            else int(depth)
        )

        run = build_run_dict(group_records, run_idx=0)
        quality = evaluate_quality(qrels, run, depth=depth_value)
        latency = latency_percentiles(group_records)
        meta = (arm_metadata or {}).get(arm, {})

        row = {
            "dataset": dataset,
            "arm": arm,
            "depth": depth_value,
            # The baseline arm this row's quality gain is measured against.
            "baseline": meta.get("baseline"),
            "n_queries": len(run),
            **quality,
            "recall_at_pool_depth": pool_recall(group_records, qrels),
            **latency,
            "client_side_rerank": meta.get("client_side_rerank"),
            "license_note": meta.get("license_note"),
        }
        if memory_by_representation:
            # One column set per named vector the arm reads. The breakdown is exported
            # alongside the total because raw storage and the HNSW graph scale differently.
            for rep in meta.get("requires", ()):
                mem = memory_by_representation.get(rep) or {}
                total = mem.get("total") or {}
                row[f"mem_{rep}_disk_bytes"] = total.get("disk_bytes")
                row[f"mem_{rep}_working_set_bytes"] = total.get("working_set_bytes")
                row[f"mem_{rep}_storage_disk_bytes"] = (mem.get("storage") or {}).get("disk_bytes")
                row[f"mem_{rep}_index_disk_bytes"] = (mem.get("index") or {}).get("disk_bytes")
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = ["dataset", "arm", "depth"]
    return out.sort_values(sort_cols, na_position="first").reset_index(drop=True)


def gain_versus_depth(agg: pd.DataFrame, *, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """The primary deliverable curve, with monotonicity classified per arm.

    Only ~72.6% of configurations improve monotonically with depth.
    Non-monotonic results are preserved and labelled, never smoothed.
    """
    swept = agg[agg["depth"].notna()].copy()
    if swept.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, arm), group in swept.groupby(["dataset", "arm"]):
        ordered = group.sort_values("depth")
        values = ordered[metric].to_numpy(dtype=np.float64)
        depths = ordered["depth"].to_numpy(dtype=np.int64)
        deltas = np.diff(values)
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "metric": metric,
                "depths": depths.tolist(),
                "values": values.tolist(),
                "monotonic_increasing": bool(np.all(deltas >= -1e-12)) if deltas.size else None,
                "best_depth": int(depths[int(np.nanargmax(values))]) if values.size else None,
                "best_value": float(np.nanmax(values)) if values.size else None,
                "value_at_max_depth": float(values[-1]) if values.size else None,
                # Where the curve stops paying off: the shallowest depth within 1% of best.
                "knee_depth": _knee(depths, values),
            }
        )
    return pd.DataFrame(rows)


def _knee(depths: np.ndarray, values: np.ndarray, *, tolerance: float = 0.01) -> int | None:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    best = float(np.nanmax(values))
    threshold = best * (1.0 - tolerance)
    for depth, value in zip(depths.tolist(), values.tolist()):
        if not math.isnan(value) and value >= threshold:
            return int(depth)
    return None


def top_k_overlap(a: Sequence[str], b: Sequence[str], k: int = 10) -> float:
    set_a, set_b = set(a[:k]), set(b[:k])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a), len(set_b), 1)


def kendall_tau_b(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Kendall's tau-b over the union of two lists, absentees ranked last.

    Plain tau is undefined on partially-overlapping lists, so the variant is named
    explicitly rather than left implicit.
    """
    from scipy.stats import kendalltau

    union = list(dict.fromkeys([*a, *b]))
    if len(union) < 2:
        return None
    rank_a = {doc: i for i, doc in enumerate(a)}
    rank_b = {doc: i for i, doc in enumerate(b)}
    missing_a, missing_b = len(a), len(b)
    x = [rank_a.get(doc, missing_a) for doc in union]
    y = [rank_b.get(doc, missing_b) for doc in union]
    result = kendalltau(x, y, variant="b")
    tau = float(result.correlation)
    return None if math.isnan(tau) else tau


def rbo(a: Sequence[str], b: Sequence[str], p: float = 0.9) -> float:
    """Rank-biased overlap, extrapolated variant (RBO_EXT, Webber et al. 2010).

    Top-weighted and defined on truncated lists of unequal length. The extrapolated
    variant is the one to use here: the plain truncated sum scores two *identical*
    10-element lists at 0.65 rather than 1.0, which would read as divergence where there
    is none — exactly the wrong error for a divergence measurement.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    s, l = len(short), len(long_)

    seen_short: set[str] = set()
    seen_long: set[str] = set()
    overlap = [0] * (l + 1)
    for d in range(1, l + 1):
        if d <= s:
            seen_short.add(short[d - 1])
        seen_long.add(long_[d - 1])
        overlap[d] = len(seen_short & seen_long)

    x_s, x_l = overlap[s], overlap[l]
    term_seen = sum(overlap[d] / d * p**d for d in range(1, l + 1))
    # Beyond the shorter list's depth, its agreement is held at its final value.
    term_tail = sum(x_s * (d - s) / (s * d) * p**d for d in range(s + 1, l + 1))
    term_extrapolated = ((x_l - x_s) / l + x_s / s) * p**l
    return (1.0 - p) / p * (term_seen + term_tail) + term_extrapolated
