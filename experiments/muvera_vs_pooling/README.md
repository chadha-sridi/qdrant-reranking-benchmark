# MUVERA FDE vs Mean Pooling

A side experiment, kept apart from the main benchmark code. When the only embeddings you
have are multivectors, the simplest single-vector stand-in for a first stage is to pool
each document's token vectors into one vector. This experiment compares that with the
MUVERA FDE.

## Method

`check.py` reuses the benchmark's encoders, calibration inputs, and MUVERA settings
(LateOn tokens, `k_sim=4`, `dim_proj=16`, `r_reps=40`). For each dataset's 50 held-out
warmup queries, it takes the exact MaxSim top 10 over the whole corpus and records, at
each depth, the share of those 10 documents found by three single-vector rankings:

- **FDE:** the benchmark's FDE, built from corpus-mean-centered tokens, scored with cosine.
- **Mean, centered:** the mean of each document's centered token vectors, scored with
  cosine. The query is pooled the same way.
- **Mean, raw:** the mean of the raw token vectors, scored with cosine.

All rankings are brute force. There is no HNSW index.

```bash
.venv/bin/python experiments/muvera_vs_pooling/check.py nfcorpus scifact scidocs
```

## Results

Share of the exact MaxSim top 10 found within each depth:

| Dataset | Depth | FDE | Mean, centered | Mean, raw |
|---|---:|---:|---:|---:|
| NFCorpus | 10 | 38% | 21% | 20% |
| NFCorpus | 100 | 77% | 58% | 57% |
| NFCorpus | 500 | 92% | 86% | 85% |
| NFCorpus | 1000 | 97% | 95% | 93% |
| SciFact | 10 | 45% | 29% | 29% |
| SciFact | 100 | 82% | 69% | 69% |
| SciFact | 500 | 96% | 88% | 89% |
| SciFact | 1000 | 98% | 96% | 96% |
| SCIDOCS | 10 | 17% | 12% | 11% |
| SCIDOCS | 100 | 57% | 29% | 32% |
| SCIDOCS | 500 | 96% | 77% | 85% |
| SCIDOCS | 1000 | 99% | 91% | 93% |

Every depth is in `results.json`.

On all three datasets, the FDE recovered more of the exact MaxSim top 10 than either
pooled vector at every depth. The gap was widest at shallow and middle depths, for
example 57% against 29% to 32% at depth 100 on SCIDOCS, and narrowed by depth 1,000.
Centering didn't consistently help mean pooling.

## Limits

- 50 queries per dataset and no significance test.
- Brute-force ranking only, so HNSW's approximation isn't included.
- One late interaction model (LateOn) and one FDE setting.
- This measures agreement with exact MaxSim, not end-to-end nDCG@10 after rescoring.
- The FDE has 10,240 dimensions and the pooled vectors have 128, so the FDE costs 80
  times the memory per document.
