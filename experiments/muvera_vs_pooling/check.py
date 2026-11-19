"""Compare the MUVERA FDE with pooling the token vectors into one vector.

For each dataset's held-out warmup queries, measures what share of the exact MaxSim
top 10 each single-vector stand-in recovers within each depth:

- fde_cosine: the benchmark's FDE (centered inputs), scored with cosine.
- mean_centered: the mean of the corpus-mean-centered token vectors, scored with cosine.
- mean_raw: the mean of the raw token vectors, scored with cosine.

All rankings are brute force, with no HNSW index.

Run from the benchmark repository root:
    .venv/bin/python experiments/muvera_vs_pooling/check.py nfcorpus scifact
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
    mean = np.zeros(doc_tokens[0].shape[1], dtype=np.float32) if mean is None else mean

    muvera_cfg = cfg.models.muvera
    token_dim = cfg.models.late_interaction.dim
    seed = cfg.run.seeds.muvera
    truth_k = cfg.run.execution.final_limit

    muvera = build_muvera(muvera_cfg, token_dim, seed)
    docs = {
        "fde_cosine": unit(encode_documents(muvera_cfg, token_dim, seed, doc_tokens, mean=mean)),
        "mean_centered": unit(np.stack([(d - mean).mean(axis=0) for d in doc_tokens])),
        "mean_raw": unit(np.stack([np.asarray(d).mean(axis=0) for d in doc_tokens])),
    }
    blocks = chamfer_blocks(doc_tokens)

    recall = {name: dict.fromkeys(DEPTHS, 0.0) for name in docs}
    for query in query_tokens:
        truth = set(np.argsort(-exact_chamfer_scores(query, blocks))[:truth_k].tolist())
        queries = {
            "fde_cosine": unit(encode_query(muvera, query, mean)),
            "mean_centered": unit((query - mean).mean(axis=0)),
            "mean_raw": unit(np.asarray(query).mean(axis=0)),
        }
        for name, matrix in docs.items():
            order = np.argsort(-(matrix @ queries[name]))
            for depth in DEPTHS:
                recall[name][depth] += len(truth & set(order[:depth].tolist())) / truth_k

    n = len(query_tokens)
    return {
        "n_documents": len(doc_tokens),
        "n_queries": n,
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
              f"{res['seconds']} s)", flush=True)
        r = res["recall_at"]
        for depth in DEPTHS:
            k = str(depth)
            print(f"  depth {depth:>4}: FDE {r['fde_cosine'][k]:.3f}  "
                  f"mean centered {r['mean_centered'][k]:.3f}  mean raw {r['mean_raw'][k]:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
