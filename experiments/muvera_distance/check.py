"""Compare dot product and cosine as the MUVERA FDE similarity.

Reuses the benchmark's calibration inputs and encoders unchanged, and measures, for each
dataset's held-out warmup queries, what share of the exact MaxSim top 10 the FDE ranking
recovers within each depth. Both metrics are brute force (no HNSW), so the only
difference between the two columns is the similarity function.

Run from the benchmark repository root:
    .venv/bin/python experiments/muvera_distance/check.py nfcorpus scifact
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from src.cli import build_encoders, load_config, load_dataset_bundle  # noqa: E402
from src.muvera import (  # noqa: E402
    build_muvera,
    calibration_inputs,
    chamfer_blocks,
    encode_documents,
    encode_query,
    exact_chamfer_scores,
)

DEPTHS = (10, 25, 50, 100, 200, 500, 1000)
OUT = Path(__file__).resolve().parent / "results.json"


def unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def check(dataset: str) -> dict:
    cfg = load_config()
    encoders = build_encoders(cfg)
    bundle = load_dataset_bundle(cfg, dataset)
    doc_tokens, query_tokens, mean = calibration_inputs(cfg, bundle, encoders)

    muvera_cfg = cfg.models.muvera
    token_dim = cfg.models.late_interaction.dim
    seed = cfg.run.seeds.muvera
    truth_k = cfg.run.execution.final_limit

    muvera = build_muvera(muvera_cfg, token_dim, seed)
    docs = encode_documents(muvera_cfg, token_dim, seed, doc_tokens, mean=mean)
    docs_unit = unit(docs)
    blocks = chamfer_blocks(doc_tokens)

    recall = {"dot": dict.fromkeys(DEPTHS, 0.0), "cosine": dict.fromkeys(DEPTHS, 0.0)}
    for query in query_tokens:
        truth = set(np.argsort(-exact_chamfer_scores(query, blocks))[:truth_k].tolist())
        q = encode_query(muvera, query, mean)
        rankings = {"dot": np.argsort(-(docs @ q)), "cosine": np.argsort(-(docs_unit @ unit(q)))}
        for metric, order in rankings.items():
            for depth in DEPTHS:
                recall[metric][depth] += len(truth & set(order[:depth].tolist())) / truth_k

    n = len(query_tokens)
    norms = np.linalg.norm(docs, axis=1)
    return {
        "n_documents": len(doc_tokens),
        "n_queries": n,
        "doc_fde_norm_cv": float(norms.std() / norms.mean()),
        "recall_at": {m: {str(d): v / n for d, v in r.items()} for m, r in recall.items()},
    }


def main() -> None:
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for dataset in sys.argv[1:] or ["nfcorpus", "scifact"]:
        start = time.time()
        res = check(dataset)
        res["seconds"] = round(time.time() - start)
        results[dataset] = res
        OUT.write_text(json.dumps(results, indent=2))
        print(f"== {dataset} ({res['n_documents']} docs, {res['n_queries']} queries, "
              f"{res['seconds']} s, doc FDE norm CV {res['doc_fde_norm_cv']:.3f})", flush=True)
        for depth in DEPTHS:
            d, c = res["recall_at"]["dot"][str(depth)], res["recall_at"]["cosine"][str(depth)]
            print(f"  depth {depth:>4}: dot {d:.3f}  cosine {c:.3f}  diff {c - d:+.3f}", flush=True)


if __name__ == "__main__":
    main()
