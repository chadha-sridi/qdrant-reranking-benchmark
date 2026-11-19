# MUVERA FDE Similarity: Dot Product vs Cosine

A side experiment, kept apart from the main benchmark code. It compares two ways of
scoring MUVERA Fixed Dimensional Encodings (FDEs) against each other when they are used
to retrieve candidates for MaxSim rescoring.

## Why

The MUVERA paper defines the FDE so that the inner product of a query FDE and a
document FDE approximates Chamfer (MaxSim) similarity. In this benchmark, the two
MUVERA steps used different similarities:

- `tune-muvera` and `calibrate-muvera` rank documents with a plain dot product in numpy
  (`src/muvera.py`, `calibrate`).
- The Qdrant collection stores the `muvera` vector with `distance: cosine`
  (`config/models.yaml`), as in Qdrant's MUVERA tutorial. Cosine scores unit-length
  vectors, so it ignores differences in FDE length between documents.

This experiment measures whether that difference changes which candidates the FDE
retrieves.

## Method

`check.py` reuses the benchmark's encoders, calibration inputs, and MUVERA settings
unchanged (LateOn tokens, `k_sim=4`, `dim_proj=16`, `r_reps=40`, corpus-mean centering
of the FDE inputs). For each dataset's 50 held-out warmup queries, it:

1. Computes exact MaxSim against every document and takes the top 10 as the reference.
2. Ranks every document by FDE similarity, once with the dot product and once with
   cosine.
3. Records, at each depth, the share of the reference top 10 found in each ranking.

Both rankings are brute force. There is no HNSW index, so the only difference between
the two columns is the similarity function. The dot-product column reproduces the
benchmark's own calibration files exactly.

```bash
.venv/bin/python experiments/muvera_distance/check.py nfcorpus scifact scidocs
```

If a dataset's LateOn token vectors aren't in `cache/` for the current platform, the
benchmark's loader encodes the whole corpus first. On a laptop CPU, that took about 8.7
hours for SCIDOCS.

## Results

Share of the exact MaxSim top 10 found within each depth:

| Depth | NFCorpus dot | NFCorpus cosine | SciFact dot | SciFact cosine | SCIDOCS dot | SCIDOCS cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 28% | 38% | 39% | 45% | 11% | 17% |
| 25 | 47% | 56% | 57% | 64% | 18% | 28% |
| 50 | 62% | 69% | 68% | 75% | 23% | 38% |
| 100 | 76% | 77% | 78% | 82% | 31% | 57% |
| 200 | 85% | 85% | 85% | 88% | 49% | 74% |
| 500 | 92% | 92% | 94% | 96% | 88% | 96% |
| 1000 | 97% | 97% | 98% | 98% | 98% | 99% |

Raw numbers are in `results.json`.

On these three datasets, cosine found as much of the exact MaxSim top 10 as the dot
product or more at every depth, within 0.2 points where it was lower. The gap was
largest at shallow and middle depths, and largest overall on SCIDOCS. At depth 1,000,
the two were close on all three.

The document FDE norms varied by about 9% to 10% around their mean (coefficient of
variation 0.087 to 0.101), so cosine does reorder documents relative to the dot
product. We haven't tested why that reordering helps here.

## Limits

- 50 queries per dataset and no significance test.
- Brute-force ranking only. Results inside Qdrant also depend on the HNSW approximation.
- One late interaction model (LateOn), one FDE setting, and centered inputs only.
- FiQA was stopped before it finished, and Quora wasn't run, because exact MaxSim over
  its full corpus doesn't fit in memory.
- These numbers say nothing about how either choice behaves with other models or
  corpora.

## What It Means for the Benchmark

The MUVERA results in `results/` were produced with cosine, the better of the two in
this check. The calibration files in `results/agg` were produced with the dot product,
so they understate how well the cosine FDE recovered the exact MaxSim top 10 on these
three datasets.
