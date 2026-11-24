# Reranking Benchmark for Qdrant

A reproducible benchmark of reranking inside and around Qdrant. It measures what rescoring a first-stage candidate pool buys in retrieval quality, and what it costs in latency and storage, on five BEIR datasets and at candidate depths from 10 to 1,000.

## Contents

- [What It Compares](#what-it-compares)
- [Headline Results](#headline-results)
- [Quickstart](#quickstart)
- [Reproducing the Published Run](#reproducing-the-published-run)
- [Results Layout](#results-layout)
- [Reading the Numbers](#reading-the-numbers)
- [Known Limitations](#known-limitations)

[docs/methodology.md](docs/methodology.md) covers collection layout, ingestion, timers, memory accounting, execution order, MUVERA calibration, and the validation gates.

## What It Compares

Two first stages, dense search and BM25, are each rescored by the same rerankers. Every arm runs against the same collection, interleaved in one pass.

| Arm | First stage | Reranking |
|---|---|---|
| `dense` | Dense HNSW search (`BAAI/bge-small-en-v1.5`) | None (baseline) |
| `late_interaction_rescore` | Dense | MaxSim over LateOn multivectors, inside Qdrant |
| `muvera_late_interaction_rescore` | MUVERA FDE search | MaxSim over LateOn multivectors, inside Qdrant |
| `cross_encoder` | Dense | `gte-reranker-modernbert-base`, on the client |
| `bm25` | Qdrant BM25 full-text search | None (baseline) |
| `bm25_late_interaction_rescore` | BM25 | MaxSim over LateOn multivectors, inside Qdrant |
| `bm25_cross_encoder` | BM25 | `gte-reranker-modernbert-base`, on the client |

Hybrid search and fusion (RRF, DBSF) are out of scope. This benchmark measures reranking, not first-stage fusion.

The datasets span 144 times in corpus size and from about 1 to 38 relevant documents per query:

| Dataset | Corpus | Evaluated queries |
|---|---:|---:|
| NFCorpus | 3,633 | 273 |
| SciFact | 5,183 | 250 |
| SCIDOCS | 25,657 | 950 |
| FiQA-2018 | 57,638 | 598 |
| Quora | 522,931 | 450 (capped at 500 sampled) |

Every judged query runs except on Quora, and 50 per dataset are held out for warmup. Quora is excluded from `--dataset all` and runs on its own with `--dataset quora`.

## Headline Results

Dense prefetch of 100 candidates rescored with LateOn, compared with dense search alone. Latency is the end-to-end median, from encoding the query to receiving results. Storage is per document on disk.

| Dataset | nDCG@10: dense → rescored | Latency: dense → rescored | Storage: dense vector → added multivectors |
|---|---|---|---|
| NFCorpus | 0.338 → 0.379 (+12%) | 10.5 → 33.6 ms | 1.8 KiB → +210 KiB |
| SciFact | 0.731 → 0.765 (+5%, not significant) | 15.2 → 56.3 ms | 1.7 KiB → +191 KiB |
| SCIDOCS | 0.198 → 0.217 (+10%) | 13.7 → 47.1 ms | 1.6 KiB → +110 KiB |
| FiQA | 0.383 → 0.499 (+30%) | 14.4 → 46.9 ms | 1.6 KiB → +72 KiB |
| Quora | 0.887 → 0.902 (+2%, not significant) | 18.3 → 55.0 ms | 1.6 KiB → +9.1 KiB (on disk) |

Every number comes from `results/agg/`, which also holds the other arms, the full depth curves, and the significance tests.

## Quickstart

Requirements:

- Python 3.11+.
- Docker, with the daemon running.
- About 2 GB of disk for model weights, plus space for corpus caches and Qdrant storage.
- Network access to the Hugging Face Hub (no account needed).

No GPU is needed. Every model runs in float32 on CPU through ONNX Runtime.

```bash
uv sync                          # or: pip install -e . && pip install pytest pytest-timeout
docker compose up -d             # Qdrant v1.20.0, the first release with BM25 scoring
python -m src.cli doctor         # check the environment
python scripts/smoke.py          # trimmed end-to-end run: 2 arms, about 20 queries
python -m src.cli run --dataset nfcorpus
```

On macOS, Docker runs in a VM with its own CPU and memory limits, and that VM is what Qdrant gets. Give it enough memory to keep each collection resident, for example `colima start --cpu 8 --memory 10` for the four default datasets.

Other commands:

| Command | What it does |
|---|---|
| `run` | Index, measure memory, run the interleaved plan, and aggregate |
| `aggregate` | Rebuild the aggregates from raw records without querying again |
| `memory` | Build or reuse a collection and print Qdrant's per-vector memory |
| `tune-muvera` | Sweep MUVERA FDE settings on held-out queries |
| `calibrate-muvera` | Check the FDE's recall against exact MaxSim on one dataset |
| `plan` | Print the execution plan and arm exclusions without touching Qdrant |
| `doctor` | Report platform, ONNX providers, and Qdrant reachability |

Run any command with `--help` for its flags. Every command can be re-run safely, and `run` skips records already on disk unless you pass `--no-resume`.

## Reproducing the Published Run

The published results come from a single machine that ran both the client and Qdrant:

- Google Cloud VM, AMD EPYC 7B12, 16 vCPUs, 62.8 GB RAM, Linux x86_64.
- Qdrant v1.20.0 in Docker, with the whole machine available to it.
- Commit and every package version recorded in `results/manifest.json`.

```bash
for ds in nfcorpus scifact scidocs fiqa; do
  python -m src.cli calibrate-muvera --dataset "$ds"
done
python -m src.cli run --dataset all
python -m src.cli run --dataset quora
```

The MUVERA setting in `config/models.yaml` is already the one `tune-muvera` selected, so tuning is optional. Quora isn't calibrated, because `calibrate-muvera` loads every document's multivectors into memory to compute exact MaxSim.

Query time in the published run, excluding corpus embedding and indexing:

| Dataset | Hours |
|---|---:|
| NFCorpus | 2.0 |
| SciFact | 2.1 |
| SCIDOCS | 6.5 |
| FiQA | 4.6 |
| Quora | 0.7 |

The two cross-encoder arms take most of that time, at about 14 seconds per query at depth 50, against 30 to 55 ms for the arms that rescore inside Qdrant. To cut runtime, leave them out with `--arms`. Corpus embeddings are cached per machine, so later runs skip encoding.

Latency depends on the machine, so results from different hardware aren't directly comparable.

## Results Layout

```
results/
├── agg/            aggregates, depth curves, significance tables, memory, MUVERA tuning and calibration
├── queries/        the query sample, the warmup holdout, and the document id maps
└── manifest.json   versions, seeds, model file digests, hardware, and per-dataset MUVERA verdicts
```

`agg/` holds, per dataset:

- `DATASET__aggregate.csv` with quality, latency percentiles, and memory per arm and depth
- `DATASET__gain_vs_depth.csv` with the gain curve and the knee, the shallowest depth within 1% of the best
- `DATASET__significance.csv` with paired tests, raw and Bonferroni-corrected
- `DATASET__memory.json` with Qdrant's memory accounting per named vector
- `DATASET__muvera_calibration.json` with the FDE's recall against exact MaxSim

`all_datasets__aggregate.csv` combines the per-dataset aggregates, and `muvera_tuning.json` holds the FDE tuning sweep.

The raw per-query records (`results/raw/`, about 215 MB) aren't in the repository. Every number in `agg/` was computed from them with `aggregate`.

## Reading the Numbers

- **Latency is unloaded, serial latency.** One client sent one query at a time, so it's a lower bound on production latency, not a throughput measurement.
- **Client and server shared one machine.** The client/server split compares two processes on the same hardware.
- **Memory isn't additive per arm.** Several arms read the same vector, so summing per-arm memory double-counts. Use the per-vector figures.
- **Dense latency is higher than in a dense-only deployment,** because the collection also holds the multivectors and MUVERA vectors.
- **Everything runs on one shard.**
- **Depth curves aren't smoothed.** A non-monotonic curve is reported as measured.
- **Depths 500 and 1,000 are diagnostic points.** On NFCorpus, depth 1,000 is 28% of the corpus.

## Known Limitations

- **MUVERA failed calibration on FiQA, and Quora was never calibrated.** FiQA's FDE recovered 75% of the exact MaxSim top 10 at depth 500, below the 80% bar. The run used `require_calibration: false`, so both datasets ran the MUVERA arm anyway. FiQA's calibration file was written after the run, so the manifest records it as uncalibrated.
- **ColBERTv2 was compared in an earlier run** on the same pipeline. Those results aren't in this repository.
- **The cross-encoder stops at depth 50,** because it ran on CPU.
- **Prefetch and rescoring weren't timed separately,** and neither were ingestion and index build.
- **The benchmark sets `on_disk` for Quora's multivectors.** Qdrant 1.19 introduced memory tiers, which replace that setting.
- **Memory figures are Qdrant's own estimates,** read once. `ram_bytes` runs 10 to 15% low, so `disk_bytes` is the most reliable column.
