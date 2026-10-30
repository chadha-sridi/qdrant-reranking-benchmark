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

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MuveraModel
from .platform_info import stable_digest

# Set once per worker process by _init_worker; the matrices are deterministic in
# (config, token_dim, seed), so every worker holds the same encoder.
_WORKER_MUVERA = None


def _init_worker(cfg_dump: dict, token_dim: int, seed: int) -> None:
    global _WORKER_MUVERA
    _WORKER_MUVERA = build_muvera(MuveraModel(**cfg_dump), token_dim, seed)


def _encode_slice(args: tuple[list[np.ndarray], np.ndarray | None]) -> np.ndarray:
    doc_tokens, mean = args
    return np.stack(
        [_WORKER_MUVERA.process_document(center_tokens(d, mean)) for d in doc_tokens]
    ).astype(np.float32)


def center_tokens(tokens: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """Token vectors with the corpus mean removed, for FDE encoding only."""
    tokens = np.asarray(tokens, dtype=np.float32)
    if mean is None:
        return tokens
    return tokens - np.asarray(mean, dtype=np.float32)


class TokenMean:
    """Streaming mean of a corpus's late-interaction token vectors.

    Accumulated in float64 over chunks so the result does not depend on how the corpus
    was split, and so a corpus too large to hold at once still yields the exact mean.
    """

    def __init__(self, dim: int) -> None:
        self._total = np.zeros(dim, dtype=np.float64)
        self._count = 0

    def update(self, doc_tokens: Sequence[np.ndarray]) -> None:
        for doc in doc_tokens:
            doc = np.asarray(doc, dtype=np.float64)
            self._total += doc.sum(axis=0)
            self._count += doc.shape[0]

    def result(self) -> np.ndarray:
        if self._count == 0:
            raise ValueError("cannot take the mean of an empty corpus")
        return (self._total / self._count).astype(np.float32)


def encode_query(muvera, query_tokens: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """FDE-encode one query's tokens, centered to match the documents."""
    return muvera.process_query(center_tokens(query_tokens, mean))


def encode_documents(
    cfg: MuveraModel,
    token_dim: int,
    seed: int,
    doc_tokens: Sequence[np.ndarray],
    *,
    mean: np.ndarray | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """FDE-encode a corpus, split across worker processes.

    `process_document` is NumPy-light per document, so a plain loop runs on one core.
    Documents are encoded independently, and the matrices rebuild deterministically from
    (config, token_dim, seed), so the split cannot change any output. `workers=None`
    picks one per CPU, scaled down so small corpora stay serial.
    """
    doc_tokens = list(doc_tokens)
    if not doc_tokens:
        return np.zeros((0, cfg.dim), dtype=np.float32)
    if workers is None:
        workers = min(os.cpu_count() or 1, max(1, len(doc_tokens) // 256))
    if workers <= 1:
        muvera = build_muvera(cfg, token_dim, seed)
        return np.stack(
            [muvera.process_document(center_tokens(d, mean)) for d in doc_tokens]
        ).astype(np.float32)
    bounds = np.linspace(0, len(doc_tokens), workers + 1, dtype=int)
    slices = [(doc_tokens[a:b], mean) for a, b in zip(bounds, bounds[1:]) if b > a]
    with ProcessPoolExecutor(
        max_workers=len(slices), initializer=_init_worker,
        initargs=(cfg.model_dump(), token_dim, seed),
    ) as pool:
        return np.concatenate(list(pool.map(_encode_slice, slices)), axis=0)


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


def calibration_inputs(
    cfg, bundle, encoders
) -> tuple[list[np.ndarray], list[np.ndarray], "np.ndarray | None"]:
    """Document tokens, held-out warmup query tokens, and the MUVERA centering mean.

    Calibration and tuning use only warmup queries, which are never evaluated, so no
    reported number comes from a query the FDE was checked or tuned on.
    """
    from .indexing import late_interaction_corpus, muvera_token_mean

    doc_tokens = late_interaction_corpus(cfg, bundle, encoders)
    query_tokens = [
        encoders.embed_query_late_interaction(bundle.all_queries[qid]) for qid in bundle.warmup_ids
    ]
    return doc_tokens, query_tokens, muvera_token_mean(cfg, bundle, encoders)


def chamfer_similarity(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Exact max_sim / Chamfer score. The quantity the FDE approximates."""
    sims = np.asarray(query_tokens, dtype=np.float32) @ np.asarray(doc_tokens, dtype=np.float32).T
    return float(sims.max(axis=1).sum())


def chamfer_blocks(
    doc_tokens: Sequence[np.ndarray], max_block_tokens: int = 1_000_000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Corpus token matrices packed for `exact_chamfer_scores`, bounded per block.

    Each block is (concatenated tokens, per-document start offsets); the block budget
    caps the similarity matrix one query allocates at ~128 MB for a 32-token query.
    """
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    pending: list[np.ndarray] = []
    pending_tokens = 0

    def pack() -> None:
        lens = np.array([d.shape[0] for d in pending], dtype=np.int64)
        starts = np.zeros(len(pending), dtype=np.int64)
        np.cumsum(lens[:-1], out=starts[1:])
        blocks.append((np.concatenate(pending, axis=0), starts))
        pending.clear()

    for doc in doc_tokens:
        doc = np.asarray(doc, dtype=np.float32)
        if doc.shape[0] == 0:
            raise ValueError("cannot score an empty document multivector")
        pending.append(doc)
        pending_tokens += doc.shape[0]
        if pending_tokens >= max_block_tokens:
            pack()
            pending_tokens = 0
    if pending:
        pack()
    return blocks


def exact_chamfer_scores(
    query_tokens: np.ndarray, blocks: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Every document's exact max_sim score for one query, equal to
    `chamfer_similarity` per document but computed corpus-at-a-time."""
    query = np.asarray(query_tokens, dtype=np.float32)
    parts = []
    for tokens, starts in blocks:
        sims = query @ tokens.T
        parts.append(np.maximum.reduceat(sims, starts, axis=1).sum(axis=0))
    return np.concatenate(parts)


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
    muvera_cfg: MuveraModel,
    token_dim: int,
    seed: int,
    doc_tokens: list[np.ndarray],
    query_tokens: list[np.ndarray],
    *,
    depths: Sequence[int] = (10, 50, 100, 200),
    truth_k: int = 10,
    mean: np.ndarray | None = None,
    workers: int | None = None,
) -> CalibrationResult:
    """Measure how well the FDE recovers the exact max_sim top-k, against a bag baseline.

    MUVERA is a prefetch, so what matters is whether the true max_sim top-k survives
    into the candidate set — not whether the scores correlate. The encoder is built from
    (muvera_cfg, token_dim, seed); documents are FDE-encoded in parallel and queries with
    fastembed's asymmetric `process_query` (fill_empty_clusters + normalize_by_count
    differ between the two sides, matching the MUVERA paper's aggregation).

    Must be run on REAL embeddings: isotropic random vectors give SimHash no cluster
    structure to exploit, so synthetic data cannot validate an FDE at any parameter
    setting. Calibration on real data is a requirement, not a nicety.
    """
    from scipy.stats import spearmanr

    muvera = build_muvera(muvera_cfg, token_dim, seed)
    encoded_docs = encode_documents(
        muvera_cfg, token_dim, seed, doc_tokens, mean=mean, workers=workers
    )
    bag_docs = np.stack([np.asarray(d, dtype=np.float32).sum(axis=0) for d in doc_tokens])
    blocks = chamfer_blocks(doc_tokens)

    recall = {depth: 0.0 for depth in depths}
    bag_recall = {depth: 0.0 for depth in depths}
    rhos: list[float] = []

    for query in query_tokens:
        exact = exact_chamfer_scores(query, blocks)
        truth = set(np.argsort(-exact)[:truth_k].tolist())

        approx = encoded_docs @ encode_query(muvera, query, mean)
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
