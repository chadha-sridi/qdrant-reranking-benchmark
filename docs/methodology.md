# Methodology

How the benchmark builds its collections, runs its queries, and measures quality, latency, and memory. The [README](../README.md) covers setup and the published results.

## Contents

- [Configuration](#configuration)
- [Collection Layout and Ingestion](#collection-layout-and-ingestion)
- [Candidate Depths](#candidate-depths)
- [Execution Order and Warmup](#execution-order-and-warmup)
- [What Gets Measured](#what-gets-measured)
- [MUVERA Tuning and Calibration](#muvera-tuning-and-calibration)
- [Validation Gates](#validation-gates)

## Configuration

Three YAML files under `config/` are loaded once into a validated, immutable `Config` object (`src/config.py`). Nothing else in the codebase reads YAML.

| File | Controls |
|---|---|
| `run.yaml` | Seeds, query sampling, repeats, candidate depths, Qdrant connection and HNSW/indexing thresholds, and which arms run by default |
| `models.yaml` | Model names and dimensions, late interaction and MUVERA parameters, and the Gate 1 reference scores |
| `datasets.yaml` | Which BEIR datasets are enabled, their pinned Hugging Face revisions, which representations to index, and per-dataset overrides |

Most values can be overridden per invocation with CLI flags (see `python -m src.cli run --help`). Flags win, and every effective value is recorded in `results/manifest.json`.

## Collection Layout and Ingestion

Each dataset gets one collection. Every representation is a named vector on the same points: `dense`, `muvera`, and `late_interaction`, all float32, plus a BM25-scored full-text index on the `text` payload. A run builds only the representations its arms read. Every arm queries the same collection, with the same HNSW graph, payload, and page cache, so latency stays comparable across arms.

No vector is quantized, so every arm reads full precision. `late_interaction` is built with `m=0` (no HNSW graph), because MaxSim is not symmetric and is only used to rescore candidates found some other way.

`hnsw.full_scan_threshold_kb` is 10, not Qdrant's default of 10,000. At the default, the small NFCorpus and SciFact segments would be searched by brute force while larger corpora used the graph, so the corpus-size comparison would compare two algorithms.

Ingestion runs in this order for each dataset:

```
wipe storage and restart Qdrant  (skipped by --resume-ingestion)
  -> embed corpus                (disk-cached, keyed by model, corpus, and platform)
  -> create collection           (named vectors, m=0 on late_interaction)
  -> disable indexing            (indexing_threshold = 0 during upload)
  -> upload in size-aware batches (under Qdrant's 32 MB request limit)
  -> restore indexing_threshold  (triggers one clean HNSW build)
  -> wait for green
  -> verify                      (exact point count, green status, index built)
```

Restoring the threshold after the upload is what triggers the index build. Restoring it before leaves the collection unindexed while it still reports green.

Point ids follow corpus order, so a re-sent point overwrites itself. After a crash during upload, `--resume-ingestion` reuses a collection whose configuration matches and sends only the missing ids. Embedding resumes per chunk (`ingestion.chunk_size`), so a crash loses at most one chunk.

Verification failures stop the run. A collection with missing points inflates latency and deflates recall while looking valid, and nothing downstream can detect it.

Quora's multivectors are stored on disk (`late_interaction_on_disk_threshold: 0`), the only dataset that exercises that setting. `run` refuses a collection whose working set exceeds 80% of the server's memory unless you pass `--allow-disk-bound`, since the large-vector arms would otherwise be measured from disk while `dense` was not.

Qdrant's storage is a Docker named volume, not a host bind mount. On macOS, a bind mount goes through the VM's virtiofs passthrough, which disables Qdrant's multi-mmap storage and degrades the latency and memory this benchmark measures. `run.qdrant.storage_volume` must match the volume name in `docker-compose.yml`.

## Candidate Depths

Every dataset uses one ladder: 10, 25, 50, 100, 200, 500, and 1,000. A shared ladder keeps paired significance tests possible, so an arm that can't afford the deep end is truncated to a prefix of the ladder rather than given its own.

| Arm | Depths |
|---|---|
| `late_interaction_rescore`, `bm25_late_interaction_rescore`, `muvera_late_interaction_rescore` | 10 to 1,000 |
| `cross_encoder`, `bm25_cross_encoder` | 10, 25, 50 |
| `dense`, `bm25` | Baselines, no depth |

Depths 500 and 1,000 exist for MUVERA. Its single-vector prefetch is less accurate than dense search, so it needs a deeper pool to catch up: by depth 25 to 50 on most datasets, but only between 200 and 500 on SCIDOCS.

The cross-encoder runs on the client and scores every candidate with a full transformer pass. On CPU that took about 14 seconds per query at depth 50, so it stops there.

Relevant documents per query explain where the first rungs sit:

| Dataset | Relevant per query (mean / median / p75) | Corpus | Depth 1,000 as % of corpus |
|---|---|---:|---:|
| NFCorpus | 38.2 / 16 / 40 | 3,633 | 28% |
| SciFact | 1.1 / 1 / 1 | 5,183 | 19% |
| SCIDOCS | 4.9 / 5 / 5 | 25,657 | 3.9% |
| FiQA-2018 | 2.6 / 2 / 3 | 57,638 | 1.7% |
| Quora | 1.6 / 1 / 1 | 522,931 | 0.2% |

Caveats:

- Deep rungs are diagnostic points, not operating points. No deployment would rerank a quarter of its corpus.
- Relevance density is a ceiling from the qrels, not first-stage recall. Recall is measured separately as `recall_at_pool_depth`.
- SciFact has the least statistical power (250 queries, about one relevant document each), so a real small gain can still fail significance.

## Execution Order and Warmup

Arms don't run in blocks. For each dataset:

```
units = build_plan(arms, dataset)                  # every (arm, depth) pair
warmup(units)                                      # once, before any measurement

for run_idx in range(repeats):                     # 1 in the published run
    for query in seeded_shuffle(eval_queries):     # new order every repeat
        for unit in seeded_shuffle(units):         # new order every query
            append_jsonl(unit.arm(query, unit.depth))
```

Shuffling the units for every query keeps any one arm from consistently paying a query's cold-cache cost. Shuffling the queries for every repeat keeps the same queries from always landing at the same point in the run's drift. Together they spread slow-moving effects, such as heat or memory pressure, thinly across every arm.

Warmup runs every unit once on 50 held-out queries and discards the results, so the measured run doesn't pay one-time costs like model loading, connection setup, or cold index pages. The warmup queries are a fixed, seeded holdout, disjoint from the evaluated queries. Nothing is trained, so there is no train/test split.

## What Gets Measured

### Latency

Each query records independent timers. They are kept raw and aggregated only at analysis time.

| Field | Meaning |
|---|---|
| `t_embed_dense_ms` | Client-side dense query encoding |
| `t_embed_late_interaction_ms` | Client-side late interaction query encoding |
| `t_embed_muvera_ms` | MUVERA FDE transform of the query tokens |
| `t_embed_ms` | Sum of the encoding timers |
| `t_qdrant_ms` | Client-observed round trip, summed across all Qdrant calls |
| `t_qdrant_server_ms` | Qdrant's self-reported processing time for the same calls |
| `t_qdrant_overhead_ms` | `t_qdrant_ms - t_qdrant_server_ms`: transport and deserialization |
| `t_rerank_ms` | Client-side reranking (cross-encoder arms only) |
| `t_total_ms` | `t_embed_ms + t_qdrant_ms + t_rerank_ms` |

Each arm encodes its own query, so its `t_embed_ms` is what a deployment running that arm alone would pay. Queries go over gRPC, which is what exposes Qdrant's own processing time. Results report p50 and p95. With one pass per query, p99 would rest on too few samples.

### Memory

Memory comes from Qdrant's own accounting (`GET /collections/{name}/memory`, Qdrant 1.18+), split per named vector:

| Field | Meaning |
|---|---|
| `storage` | The vectors themselves |
| `index` | The HNSW graph (about zero for `late_interaction`, which has `m=0`) |
| `payload` | `doc_id` and `text` |
| `payload_indexes` | The `text` entry is the BM25 index, reported as the `bm25` representation |
| `id_tracker` | Remaining tracked structures |
| `n_token_vectors` | Total late interaction token vectors |

Each field is reported as `disk_bytes`, `ram_bytes` (not evictable), `cached_bytes` (page cache, Unix only), and `expected_cache_bytes`. `working_set_bytes = ram_bytes + expected_cache_bytes` is the figure for how much RAM a vector needs to stay fast, since memory-mapped vectors report almost no `ram_bytes`. Qdrant's documentation puts `ram_bytes` 10 to 15% low, so `disk_bytes` is the most reliable column. On a small collection, segment pre-allocation dominates the per-vector disk figures.

### Quality

Quality uses `ranx`: nDCG@10 (primary) and Recall@10, computed from the stored rankings of `run_idx` 0. Retrieval is deterministic, so repeated passes would only inflate the sample size of the significance tests. Pairwise significance uses `ranx.compare`, with raw and Bonferroni-corrected verdicts.

`recall_at_pool_depth` is the share of relevant documents present in the candidate pool before rescoring, which caps Recall@10. It separates a reranker that failed to promote a document it had from a pool that never contained it. It is `null` for the baselines, which have no separate pool. Measuring it costs the rescoring arms one extra, untimed Qdrant call per query (`capture_pool_recall`).

## MUVERA Tuning and Calibration

MUVERA encodes mean-centered tokens (`muvera.center`). Late interaction token vectors sit in a narrow cone, and SimHash's random hyperplanes can't split a cone until the corpus mean is subtracted. The mean is computed once, cached, and recorded in the manifest. The multivectors used for rescoring are stored uncentered.

`tune-muvera` sweeps `muvera.tuning_grid` on the held-out warmup queries of NFCorpus and SciFact. It selects the smallest FDE that recovers at least 80% (`min_recall`) of the exact MaxSim top 10 at depth 500 (`calibration_depth`) on both, and writes `agg/muvera_tuning.json`. The published setting, `k_sim=4, d_proj=16, n_reps=40` (10,240 dimensions), came from that sweep. Its output is in `results/agg/muvera_tuning.json`.

`calibrate-muvera --dataset NAME` then checks the chosen setting on each dataset's whole corpus. It compares FDE prefetch recall against exact MaxSim and against a bag-of-tokens baseline, and writes `agg/NAME__muvera_calibration.json`.

`muvera.require_calibration` decides what a failing dataset does. When it is `true`, the arm is excluded for that dataset. In the published run it was `false`, so every dataset ran the arm and its verdict was recorded in the manifest. The 80% bar is a project convention, not a published standard.

## Validation Gates

The gates are correctness checks you run before trusting results, not part of a run. They are pytest tests marked `gate` and `slow`, and they need Docker, the models, and a live Qdrant. Run them again after changing models, datasets, or indexing.

| Gate | Checks |
|---|---|
| 1: `test_gate1_baseline` | `dense` and `bm25` each reproduce a published nDCG@10 on NFCorpus within 0.02 (`dense`: 0.343 from the bge-small model card; `bm25`: 0.325 from the BEIR paper). Catches broken qrels parsing, id mapping, or metric calls. |
| 2: `test_gate2_arms_distinct` | No two arms return identical rankings, and every arm either runs or is excluded with a reason. |
| 3: `test_gate3_hnsw_variance` | Two independent index builds agree within 0.005 nDCG@10, which sets the noise floor. |
| 4: `test_gate4_unattended` | The pipeline runs end to end on a subset, resumes idempotently, and aggregates cleanly. |
| MUVERA: `test_muvera_calibration` | The FDE prefetch beats bag-of-tokens and recovers at least 80% of the exact top 10 at depth 500. |

All gates passed on the machine that produced the published results.

```bash
pytest -m "not slow"   # unit tests: no Docker, no model downloads, seconds
pytest -m gate         # validation gates: needs the live stack, about an hour on a 16-vCPU VM
```
