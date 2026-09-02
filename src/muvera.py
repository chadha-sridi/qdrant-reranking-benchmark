"""MUVERA Fixed Dimensional Encoding, via FastEmbed's official implementation.

The FDE construction comes from `fastembed.postprocess.Muvera` (requires
`fastembed>=0.7.2`), the reference implementation the MUVERA/Qdrant ecosystem itself
uses. This module is a thin wrapper around it, so the question it has to answer is not
"is the algorithm right" but "does the recommended default work well enough to trust as
a prefetch" — which `calibrate()` below measures on real embeddings.

Defaults match fastembed's own: `k_sim=5, dim_proj=16, r_reps=20` -> output dimension
`r_reps * 2**k_sim * dim_proj` = 20*32*16 = 10240.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MuveraModel
from .platform_info import stable_digest


def build_muvera(cfg: MuveraModel, token_dim: int, seed: int):
    """Construct fastembed's Muvera postprocessor directly (no model object needed)."""
    from fastembed.postprocess import Muvera

    return Muvera(
        dim=token_dim,
        k_sim=cfg.k_sim,
        dim_proj=cfg.d_proj,
        r_reps=cfg.n_reps,
        random_seed=seed,
    )


def matrices_digest(muvera, cfg: MuveraModel, token_dim: int) -> str:
    """SHA-256 fingerprint of the encoder's ACTUAL random matrices, for the manifest.

    fastembed does not expose the seed it was built with — there is no `random_seed`
    attribute on the class — but it does expose the matrices themselves
    (`simhash_projections[i].simhash_vectors`, `dim_reduction_projections`). Hashing
    those cannot misrepresent what was actually constructed, whereas hashing config plus
    an assumed seed attribute yields identical digests for different seeds.
    """
    import hashlib

    h = hashlib.sha256()
    for sh in muvera.simhash_projections:
        h.update(np.ascontiguousarray(sh.simhash_vectors).tobytes())
    h.update(np.ascontiguousarray(muvera.dim_reduction_projections).tobytes())
    matrices_sha = h.hexdigest()

    return stable_digest(
        {
            "impl": "fastembed.postprocess.Muvera",
            "k_sim": cfg.k_sim,
            "n_reps": cfg.n_reps,
            "d_proj": cfg.d_proj,
            "token_dim": token_dim,
            "matrices_sha256": matrices_sha,
            "embedding_size": muvera.embedding_size,
        }
    )


def chamfer_similarity(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Exact max_sim / Chamfer score. The quantity the FDE approximates."""
    sims = np.asarray(query_tokens, dtype=np.float32) @ np.asarray(doc_tokens, dtype=np.float32).T
    return float(sims.max(axis=1).sum())


@dataclass(frozen=True)
class CalibrationResult:
    n_queries: int
    n_documents: int
    dim: int
    recall_at: dict[int, float]
    bag_recall_at: dict[int, float]
    spearman_vs_chamfer: float
    beats_bag_of_tokens: bool

    def as_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "n_documents": self.n_documents,
            "dim": self.dim,
            "recall_at": {str(k): v for k, v in self.recall_at.items()},
            "bag_recall_at": {str(k): v for k, v in self.bag_recall_at.items()},
            "spearman_vs_chamfer": self.spearman_vs_chamfer,
            "beats_bag_of_tokens": self.beats_bag_of_tokens,
        }


def calibrate(
    muvera,
    doc_tokens: list[np.ndarray],
    query_tokens: list[np.ndarray],
    *,
    depths: Sequence[int] = (10, 50, 100, 200),
    truth_k: int = 10,
) -> CalibrationResult:
    """Measure how well the FDE recovers the exact max_sim top-k, against a bag baseline.

    MUVERA is a prefetch, so what matters is whether the true max_sim top-k survives
    into the candidate set — not whether the scores correlate. `muvera` is a
    `fastembed.postprocess.Muvera` instance; `process_document`/`process_query` are its
    asymmetric encode methods (fill_empty_clusters + normalize_by_count differ between
    the two, matching the MUVERA paper's aggregation).

    Must be run on REAL embeddings: isotropic random vectors give SimHash no cluster
    structure to exploit, so synthetic data cannot validate an FDE at any parameter
    setting. Calibration on real data is a requirement, not a nicety.
    """
    from scipy.stats import spearmanr

    encoded_docs = np.stack([muvera.process_document(d) for d in doc_tokens])
    bag_docs = np.stack([np.asarray(d, dtype=np.float32).sum(axis=0) for d in doc_tokens])

    recall = {depth: 0.0 for depth in depths}
    bag_recall = {depth: 0.0 for depth in depths}
    rhos: list[float] = []

    for query in query_tokens:
        exact = np.array([chamfer_similarity(query, doc) for doc in doc_tokens])
        truth = set(np.argsort(-exact)[:truth_k].tolist())

        approx = encoded_docs @ muvera.process_query(query)
        bag = bag_docs @ np.asarray(query, dtype=np.float32).sum(axis=0)

        for depth in depths:
            recall[depth] += len(truth & set(np.argsort(-approx)[:depth].tolist())) / truth_k
            bag_recall[depth] += len(truth & set(np.argsort(-bag)[:depth].tolist())) / truth_k

        rho = spearmanr(exact, approx).statistic
        if not np.isnan(rho):
            rhos.append(float(rho))

    n = max(len(query_tokens), 1)
    recall = {d: v / n for d, v in recall.items()}
    bag_recall = {d: v / n for d, v in bag_recall.items()}
    reference_depth = max(depths)

    return CalibrationResult(
        n_queries=len(query_tokens),
        n_documents=len(doc_tokens),
        dim=muvera.embedding_size,
        recall_at=recall,
        bag_recall_at=bag_recall,
        spearman_vs_chamfer=float(np.mean(rhos)) if rhos else float("nan"),
        beats_bag_of_tokens=recall[reference_depth] > bag_recall[reference_depth],
    )
