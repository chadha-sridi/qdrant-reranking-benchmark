# Reranking Benchmark for Qdrant

A reproducible benchmark evaluating reranking strategies built around Qdrant, measuring
retrieval quality, latency, and memory cost across corpus sizes. Experiments can be run
at different candidate depths to see how each strategy scales.

---

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Arms](#arms)
- [Candidate depths](#candidate-depths)
- [Cost and runtime](#cost-and-runtime)
- [Collection layout and ingestion](#collection-layout-and-ingestion)
- [What gets measured](#what-gets-measured)
- [Execution order and warmup](#execution-order-and-warmup)
- [Running the benchmark](#running-the-benchmark)
- [Interpreting the numbers](#interpreting-the-numbers)
- [Validation gates](#validation-gates)
- [Output](#output)
- [Known limitations](#known-limitations)

---

## Prerequisites

- **Python 3.11+**
- **Docker**, daemon running. `docker compose` is used if present, otherwise plain
  `docker run`.
- **~2 GB disk** for model weights (dense + late interaction + cross-encoder, downloaded
  on first use via FastEmbed/Hugging Face Hub), plus corpus caches and Qdrant storage.
- **Network access** to Hugging Face Hub for datasets and model weights (no account or
  token needed).
- **No GPU** — everything runs CPU-side (ONNX Runtime via FastEmbed), all models in
  float32.

There is deliberately **no fixed target machine**: `docker-compose.yml` runs Qdrant
native to whatever host runs it (arm64 on Apple Silicon, amd64 on x86, never emulated).
Latency numbers are only meaningful for the machine that produced them;
`results/manifest.json` records which one that was.

On macOS, Docker runs inside a VM sized independently of the Mac — that VM, not the Mac,
is what Qdrant gets. Size it so each dataset's collection stays resident; this repo's
results use 8 CPUs / 10 GiB on a 10-core, 16 GB M4:

```bash
colima stop && colima start --cpu 8 --memory 10
```

`run` refuses a collection whose working set exceeds 80% of the server's memory (recorded
in the manifest under `qdrant.server_resources`), since large-vector arms would otherwise
be measured on disk while `dense` would not.

Qdrant's storage is a **Docker named volume**, not a host bind mount — a bind mount from
macOS hits virtiofs passthrough, which disables Qdrant's multi-mmap storage and directly
pollutes the latency/memory figures this benchmark exists to produce. Inspect storage
with `docker exec reranking_benchmark_qdrant ls /qdrant/storage` instead of the host
filesystem. `run.qdrant.storage_volume` must match the `name:` under `volumes:` in
`docker-compose.yml`.

---

## Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"          # or: uv sync

# 2. Start Qdrant (pinned to v1.20.0, first release with BM25 scoring on a text index).
#    On macOS, size the Docker VM first (see above).
docker compose up -d

# 3. Sanity-check the environment
python -m src.cli doctor

# 4. Smoke test — fast, 2 arms, capped queries, no CLI needed
python scripts/smoke.py

# 5. A real run on the smallest dataset
python -m src.cli run --dataset nfcorpus
```

The package also installs a `benchmark` console script (`benchmark run ...`); every
example below uses `python -m src.cli` instead, since that works without installing
anything.

---

## Repository layout

```
config/
├── run.yaml           seeds, query sampling, execution (repeats/depths), Qdrant
│                       connection, arm enablement
├── models.yaml         model names/dims, late-interaction/MUVERA params,
│                       cross-encoder(s), the Gate 1 reference NDCG
└── datasets.yaml       per-dataset HF source, size, representations, depth overrides

src/
├── cli.py              the `run`/`aggregate`/`memory`/`tune-muvera`/
│                       `calibrate-muvera`/`plan`/`doctor` commands
├── config.py            typed, validated config — nothing else reads YAML directly
├── datasets.py          BEIR loading, query subsampling, warmup holdout,
│                       bidirectional doc_id <-> Qdrant point-id mapping
├── embedding.py         model wrappers (FastEmbed) + platform-keyed embedding cache
├── muvera.py            thin wrapper around fastembed's official Muvera FDE
│                       postprocessor, plus the real-embedding calibration routine
├── indexing.py          collection creation, resumable upsert, index build, and
│                       ingestion verification
├── memory.py            per-named-vector memory, read from Qdrant's own accounting
├── qdrant_control.py    container lifecycle (start/stop/wipe)
├── arms.py              the arm registry — one `@register`-decorated function per
│                       reranking strategy
├── timing.py             the per-query timer set (embed per encoder / round-trip /
│                       server / overhead / rerank)
├── storage.py           raw JSONL writer and crash-safe resume
├── runner.py             execution plan (shuffle + interleave), warmup, the main loop
├── metrics.py            ranx wrappers, latency percentiles, significance testing
├── analysis.py           raw records -> published CSVs
└── manifest.py           the run manifest (versions, seeds, digests, exclusions)

tests/
├── test_units.py                fast, no Docker/models needed
├── test_muvera.py                FDE properties provable on synthetic data
├── test_gate1_baseline.py       Gate 1 — baseline reproduces published NDCG
├── test_gate2_arms_distinct.py  Gate 2 — no arm silently collapses into another
├── test_gate3_hnsw_variance.py  Gate 3 — HNSW build variance across two builds
├── test_gate4_unattended.py     Gate 4 — pipeline runs, resumes, and aggregates
└── test_muvera_calibration.py   MUVERA gate — real-embedding recall check

scripts/
└── smoke.py             trimmed end-to-end run (2 arms, ~20 queries) for a fast check

docker-compose.yml           pinned Qdrant image, no forced platform (native by default)
```

---

## Configuration

Three YAML files under `config/`, loaded once into a validated, immutable `Config`
object (`src/config.py`) — nothing else in the codebase reads YAML.

| File | Controls |
|---|---|
| `run.yaml` | seeds, how many queries are sampled and how they're split, repeats, candidate depths, Qdrant connection and HNSW/indexing thresholds, which arms run by default |
| `models.yaml` | model names and dimensions, late-interaction/MUVERA parameters, the pinned Gate 1 reference score |
| `datasets.yaml` | which BEIR datasets are enabled, their HF source, corpus size, which representations to index, per-dataset depth overrides |

Most values are also overridable per invocation via CLI flags (below) — flags win, and
every effective value is recorded in `results/manifest.json` for that run.

---

## Datasets

Five BEIR datasets, run smallest corpus first.

| Dataset | Corpus size | Description |
|---|---:|---|
| **NFCorpus** | 3,633 | Medical/nutrition IR. The only graded dataset (0/1/2) and the densest relevance judgments (38 relevant docs/query), giving it real headroom at deep depths. Also the Gate 1 reference and smoke-test target. |
| **SciFact** | 5,183 | Scientific claim verification. ~1.1 relevant docs/query — effectively one correct answer each. |
| **SCIDOCS** | 25,657 | Scientific paper retrieval from citation context. Exactly 5 relevant docs/query, always. |
| **FiQA-2018** | 57,638 | Financial opinion QA over investor forum posts. ~2.6 relevant docs/query. |
| **Quora** *(out of the default set)* | 522,931 | Duplicate-question retrieval. ~1.6 relevant docs/query. Forces late-interaction multivectors on-disk — the only dataset exercising that regime. Capped at 500 queries (all 10,000 would be ~80% of the benchmark's runtime); run explicitly with `--dataset quora`. |

Together they span 144x in corpus size and 1.1→38 in relevance density.

**Query counts** — every default-set dataset runs every judged query it has:

| Dataset | Judged queries | Warmup holdout | **Evaluated** |
|---|---:|---:|---:|
| NFCorpus | 323 | 50 | **273** |
| SciFact | 300 | 50 | **250** |
| SCIDOCS | 1,000 | 50 | **950** |
| FiQA-2018 | 648 | 50 | **598** |
| | | | **2,071 total (default set)** |
| Quora *(separate run)* | 500 of 10,000 | 50 | **450** |

Warmup queries are held out so no evaluated query is served from a cache the warmup pass
warmed — nothing is trained, there is no train/test split.

`queries.cap` in `config/run.yaml` is `null` and is a global smoke-test knob; a dataset's
own `query_cap` in `config/datasets.yaml` takes precedence (that's what caps Quora).

Only NFCorpus ships graded relevance; the loader asserts the observed grade set matches
config, so a parser bug can't silently flatten it to binary.

---

## Arms

**Two first stages, each a baseline, each reranked by the same rerankers.** `dense`
(bge-small) and `bm25` (sparse) are both first-stage retrievers; every reranking arm
starts from one of them and is named after it. Hybrid-search/fusion strategies (RRF,
DBSF, weighted RRF) are out of scope — this is a *reranking* benchmark, not a
first-stage-fusion one.

| Arm | Baseline | Sweeps depth | Mechanism |
|---|---|:---:|---|
| `dense` | itself | | A single dense HNSW search, no reranking |
| `late_interaction_rescore` | `dense` | ✓ | Dense prefetch → late-interaction `max_sim` |
| `muvera_late_interaction_rescore` | `dense` | ✓ | MUVERA prefetch → exact `max_sim` (⚠️ see [Known limitations](#known-limitations)) |
| `cross_encoder` | `dense` | ✓ | Dense prefetch → client-side cross-encoder rerank |
| `bm25` | itself | | A single BM25 search (Qdrant full-text index), no reranking |
| `bm25_late_interaction_rescore` | `bm25` | ✓ | BM25 prefetch → late-interaction `max_sim` |
| `bm25_cross_encoder` | `bm25` | ✓ | BM25 prefetch → client-side cross-encoder rerank |

All seven run by default, interleaved in the same pass against the same collection, so
both baselines are measured under identical conditions. To run one family only, name its
arms: `--arms bm25,bm25_late_interaction_rescore,bm25_cross_encoder`.
`muvera_late_interaction_rescore` has no BM25 counterpart — it's measured against `dense`,
the other single-vector first stage.

Notes worth knowing:

- **BM25** is scored by Qdrant's own full-text index (1.20+): word tokenizer,
  lowercasing, English Snowball stemming and stopwords, `k=1.2`/`b=0.75`. Nothing is
  computed client-side, so BM25 has no query-embedding time.
- **The dense model is a runtime parameter**: override with `--dense-model NAME`
  (+ `--dense-model-dim N` if FastEmbed can't resolve it). Affects only the `dense`
  family and invalidates the Gate 1 reference score — the CLI says so.
- **"Late interaction" names the mechanism, not a model.** Currently `lightonai/LateOn`
  via FastEmbed/ONNX, chosen over `answerai-colbert-small-v1` and `mLateOn` on NFCorpus
  (see DECISIONS.md O6). Every model keeps its own context window (no truncation) — bge
  and answerai at 512 tokens, LateOn at 300, mLateOn at 8,192.
- **Late-interaction arms use `max_sim`**, Qdrant's only multivector comparator. The
  vector is indexed with HNSW `m=0` (no graph) since `max_sim` isn't symmetric — it's
  used only for rescoring a candidate set that arrived some other way, never for ANN
  prefetch.

Run `python -m src.cli plan` to see every arm exclusion with its reason, for any dataset,
without touching Qdrant.

---

## Candidate depths

One ladder — **10, 25, 50, 100, 200, 500, 1000** — on every dataset, with a per-arm
ceiling for arms that can't afford its deep end. A shared ladder is what makes paired
significance tests work: a capped arm is truncated to a prefix of the ladder rather than
given one of its own. Override with `--depths` (lifts caps too), per-dataset `depths:` in
`config/datasets.yaml`, or per-arm `execution.arm_depth_caps` in `config/run.yaml`.

| Arm | Depths swept |
|---|---|
| `late_interaction_rescore`, `bm25_late_interaction_rescore` | 10 … 1000 |
| `muvera_late_interaction_rescore` | 10 … 1000 |
| `cross_encoder`, `bm25_cross_encoder` | 10, 25, 50 |
| `dense`, `bm25` | baselines, no sweep |

**MUVERA sweeps to 1000** to find its break-even point against dense — near depth 100 on
SciFact, between 500 and 1000 on NFCorpus and SciDocs. Server-side rescoring costs
20–50 ms/query at any depth, so the extra rungs are cheap (~15 min total).

**The cross-encoder stops at 50**: it's the one arm that reranks client-side (a
transformer forward pass per query/document pair, ~2.8 pairs/s), and its curve is already
flat by depth 50 on every dataset. Sweeping to 1000 would cost ~155h for no new signal.

Relevance density, the reason the first few rungs are where they are:

| Dataset | rel/query (avg / med / p75) | Corpus | Depth 1000 as % of corpus |
|---|---|---:|---:|
| **NFCorpus** | 38.2 / 16 / 40 | 3,633 | 28% |
| SciFact | 1.1 / 1 / 1 | 5,183 | 19% |
| SCIDOCS | 4.9 / 5 / 5 | 25,657 | 3.9% |
| FiQA-2018 | 2.6 / 2 / 3 | 57,638 | 1.7% |
| Quora | 1.6 / 1 / 1 | 522,931 | 0.2% |

Caveats:

- **Deep rungs are diagnostic points, not operating points** — no deployment would
  rerank a quarter of its corpus. Report them as a recall curve, not a recommendation.
- **The cross-encoder's curve ends at 50** by budget decision, not measurement — it has
  no depth-100+ rows to compare.
- **Relevance density is a capacity ceiling from qrels, not first-stage recall** — actual
  recall is measured separately (`recall_at_pool_depth`).
- **SciFact has the least statistical power** (250 queries, 1.1 relevant docs each), so a
  real small gain can still fail significance.

---

## Cost and runtime

Measured on this machine (Apple M4, 16 GB RAM, CPU-only):

| | Measured |
|---|---:|
| Cross-encoder rerank | **28.8 pairs/sec** |
| Dense embedding | 121 docs/sec |
| ColBERTv2 embedding | 87 docs/sec |

With `execution.share_pointwise_scores` (default on), each (query, document) pair is
scored once per query and reused across units that need it, cutting the cross-encoder
arms' wall-clock by ~45%.

**One-time, cached** — corpus embedding (615,042 docs across all five datasets): dense
~1.4h, ColBERTv2 ~2.0h, **total ~3.4h** (default set ~0.5h; Quora alone ~2.9h). BM25
needs no embedding — Qdrant builds its index from stored text on upload.

**Per full run** — all seven arms, every evaluated query, `repeats: 1`. The two
cross-encoder arms dominate even capped at depth 50:

| Dataset | Evaluated queries | Est. hours |
|---|---:|---:|
| nfcorpus | 273 | **~2.4h** |
| scifact | 250 | **~2.2h** |
| scidocs | 950 | **~7.9h** |
| fiqa | 598 | **~5.3h** |
| **`--dataset all` total** | **2,071** | **~17.8h** |
| quora *(capped at 500 queries)* | 450 | ~3.9h |
| quora *(uncapped)* | 9,950 | ~85h |

Per-query cost at depth 50 shows where it goes — three orders of magnitude between the
cross-encoder (client-side transformer forward pass) and everything else (Qdrant-side
vector arithmetic):

| Arm | ms/query @ depth 50 |
|---|---:|
| **`cross_encoder`** | **~19,000** |
| `late_interaction_rescore` | 41 |
| `muvera_late_interaction_rescore` | 39 |
| `dense` (no depth) | 13 |

Levers, in order of effect:

| Change | Saves | Costs you |
|---|---|---|
| Lift Quora's 500-query cap | costs ~81h | tighter CI on the one large-corpus dataset |
| Drop both cross-encoder arms | ~17h of ~17.8h | your only client-side reranking arms |
| Lower `arm_depth_caps` to 25 | ~9h | the depth-50 point on both cross-encoder arms |
| Swap in a smaller cross-encoder | up to ~16h | quality: MiniLM-L6 is ~12x faster and loses to dense on SciFact |
| Run one family only (`--arms`) | ~9h of ~17.8h | the other first stage |
| Drop depths 500 and 1000 | ~15 min | MUVERA's crossover, the thing those rungs exist to find |

---

## Collection layout and ingestion

**One collection per dataset**, holding every representation as a **named vector** on
the same points — `dense`, `muvera`, `late_interaction` (all fp32), plus a BM25-scored
full-text index on `text`. A run builds only the representations its requested arms read.
Every arm queries the same collection (same HNSW graph, same payload, same page cache) so
arm-to-arm latency stays comparable — the benchmark's primary axis.

No vector carries a quantization config, so every arm reads full precision. `late_interaction`
is built with `m=0` (no HNSW graph) since `max_sim` isn't a metric space.

`hnsw.full_scan_threshold_kb` is **10**, not Qdrant's default 10,000 — at the default,
small segments (NFCorpus, SciFact) would silently brute-force while larger corpora used
ANN, turning the corpus-size curve into a comparison of two algorithms.

**Ingestion flow**, per dataset:

```
wipe storage + restart Qdrant   (clean slate; skipped by --resume-ingestion)
  -> embed corpus               (disk-cached, keyed by model + corpus + platform)
  -> create collection          (named vectors, m=0 on late_interaction)
  -> disable indexing           (indexing_threshold = 0 for the upload)
  -> upload in size-aware batches   (sized from actual vector width, under Qdrant's 32 MB cap)
  -> restore indexing_threshold     (this is what triggers one clean HNSW build)
  -> wait for green
  -> verify                     (exact point count, green status, index actually built)
```

The order matters — restoring the threshold *after* upload is what triggers the index
build; doing it before leaves the collection permanently unindexed while reporting green.

**Interruption is recoverable.** Point ids are the corpus enumeration order, so a
re-sent point overwrites itself. `--resume-ingestion` reuses a config-matching collection
and re-sends only missing ids; the collection is dropped and rebuilt if its vector config
or recorded models don't match what the run asks for.

**Verification is a hard failure, not a warning** — a short collection inflates latency
and deflates recall while looking like a valid run, and nothing downstream can detect
that after the fact. Ingestion aborts on any mismatch.

Embedding resumes per chunk (`ingestion.chunk_size`), so a crash loses at most one chunk.
MUVERA document encoding runs across `ingestion.muvera_workers` processes.

---

## What gets measured

**Latency**, independent timers per query, recorded raw and aggregated only at analysis
time:

| Field | What it is |
|---|---|
| `t_embed_dense_ms` | client-side dense query embedding |
| `t_embed_late_interaction_ms` | client-side late-interaction query embedding |
| `t_embed_muvera_ms` | the MUVERA FDE transform of the late-interaction query tokens |
| `t_embed_ms` | sum of the embedding buckets |
| `t_qdrant_ms` | client-observed round trip, summed across all Qdrant calls |
| `t_qdrant_server_ms` | Qdrant's own self-reported processing time for the same calls |
| `t_qdrant_overhead_ms` | `t_qdrant_ms - t_qdrant_server_ms` — transport + deserialization |
| `t_rerank_ms` | client-side reranking (non-zero only for the two cross-encoder arms) |
| `t_total_ms` | `t_embed_ms + t_qdrant_ms + t_rerank_ms` |

Each arm embeds its own query, so its `t_embed_ms` is what a deployment running that arm
alone would pay; a bucket an arm doesn't use is recorded as 0. Reported as p50/p95 — p99
is skipped since one pass per query means one sample, making p99 false precision.

Queries go over **gRPC** and read Qdrant's self-reported processing time directly off the
response, which is how the server/transport split is measurable at all.

**Memory**, read from Qdrant's own per-collection accounting
(`GET /collections/{name}/memory`, Qdrant 1.18+), split per named vector:

| Field | What it is |
|---|---|
| `storage` | the fp32 vectors themselves |
| `index` | the HNSW graph (≈0 for `late_interaction`, which has `m=0`) |
| `payload` | `doc_id` + `text` |
| `payload_indexes` | the `text` entry is BM25's index, reported as the `bm25` representation |
| `id_tracker` | remaining accounted-for structure |
| `n_token_vectors` | late-interaction token-vector total |

Each is reported in `disk_bytes`, `ram_bytes` (non-evictable), `cached_bytes` (page
cache, Unix only), and `expected_cache_bytes`. `working_set_bytes` = `ram_bytes` +
`expected_cache_bytes` is the "how much RAM does this need" figure — mmap'd vectors
report ~0 `ram_bytes` while still needing residency to be fast. Qdrant's own docs put
`ram_bytes` 10–15% low.

**Quality**, via `ranx`: **NDCG@10** (primary) and **Recall@10**, computed from `run_idx 0`
only (retrieval is deterministic, so repeated passes would inflate significance-test
sample size). Pairwise significance uses `ranx.compare` with raw and Bonferroni-corrected
verdicts. Metrics are always recomputed post-hoc from the stored ranking.

**`recall_at_pool_depth`** — the ceiling on Recall@10: what fraction of relevant docs
were present in the prefetch pool before rescoring. Separates "reranker failed to promote
a doc it had" from "the pool never had it, go deeper." `null` for `dense`, which has no
separate pool; at depth 10 it equals Recall@10 exactly.

---

## Execution order and warmup

No hardcoded run order, no arm-by-arm block execution. Per dataset:

```
run_dataset(dataset):
    units = build_plan(arms, dataset)        # every (arm, depth) pair
    warmup(units)                            # once, before any measurement

    for run_idx in 1..repeats:                # 3 by default
        query_order = seeded_shuffle(eval_queries)     # new order every repeat
        for query in query_order:
            unit_order = seeded_shuffle(units)          # new order every query
            for unit in unit_order:
                result = unit.arm(query, unit.depth)
                append_jsonl(arm, depth, query, run_idx, result)
```

- **`unit_order`** reshuffles every query so whichever arm runs first doesn't
  consistently eat a query's cold-cache cost, biasing later arms in its favor.
- **`query_order`** reshuffles every repeat so later-sorted queries don't always land on
  the same point in the run's drift (thermal, memory pressure).
- Both are what make interleaving arms (instead of running them block-by-block) actually
  work: any slow-moving confound gets smeared thinly across every arm and query instead
  of concentrating on whichever happened to run during a bad stretch.

**Warmup**: every unit runs once against a small held-out query pool, results discarded,
before any repeat begins — so the measured run doesn't pay one-time costs (model
lazy-loading, connection setup, cold index pages, CPU ramp-up) that have nothing to do
with the arm under test. The **50 warmup queries are a fixed, seeded holdout**, disjoint
from evaluated queries, persisted per dataset.

---

## Running the benchmark

All commands use `python -m src.cli <command>` (swap in `benchmark <command>` if
installed). Every command is safe to re-run; `run` skips records already on disk unless
you pass `--no-resume`.

### `run` — index, measure memory, execute, aggregate

```bash
python -m src.cli run --dataset all              # the four-dataset default set, ~17.8h
python -m src.cli run --dataset quora            # Quora on its own, capped at 500 queries
python -m src.cli run --dataset nfcorpus
python -m src.cli run --dataset all --arms dense,late_interaction_rescore
python -m src.cli run --dataset fiqa --dense-model BAAI/bge-base-en-v1.5
```

`--dataset all` runs the default set only; Quora is configured but excluded
(`enabled: false`) because uncapped it's ~80% of the benchmark's runtime. **Naming a
dataset explicitly runs it regardless of that flag.** Running Quora separately doesn't
affect other datasets' numbers — each has its own collection, execution plan, and
aggregation.

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `nfcorpus` | dataset name, or `all` for every enabled dataset |
| `--arms` | `all` | comma-separated arm names, or `all` for the configured default set |
| `--depths` | config | override the depth sweep, e.g. `10,50,100` |
| `--resume` / `--no-resume` | resume | skip records already on disk |
| `--overwrite` | off | with `--no-resume`, delete this dataset's existing raw records first |
| `--measure-memory` / `--no-measure-memory` | on | read Qdrant's per-vector memory accounting |
| `--resume-ingestion` | off | re-send only missing points instead of wiping storage — use after a crash mid-upload |
| `--allow-unverified-baseline` | off | proceed when the dense model's Gate 1 reference is `verified: false` |
| `--dense-model` | config | override the dense first-stage model |
| `--dense-model-dim` | auto | dimension of `--dense-model`, if FastEmbed can't resolve it |
| `--analyse` / `--no-analyse` | on | aggregate after the run completes |
| `--allow-disk-bound` | off | run even when a collection's working set exceeds 80% of Qdrant's memory |
| `--results-dir` | `results` | write to another directory (embedding cache stays shared) |
| `--allow-partial` | off | aggregate even with incomplete `(arm, depth)` units — for inspection only, never for a trusted result |

`aggregate` accepts the same `--allow-partial` flag.

### `aggregate` — re-aggregate without re-querying

```bash
python -m src.cli aggregate --dataset nfcorpus
```

Rebuilds every CSV from the raw JSONL already on disk. Use after tweaking analysis code,
or to regenerate deleted output.

### `memory` — memory phase only

```bash
python -m src.cli memory --dataset nfcorpus
```

Builds (or reuses) the collection, reads Qdrant's per-vector memory accounting, prints it
as JSON.

MUVERA encodes **mean-centered** tokens (`muvera.center`): late-interaction embeddings
collapse into a narrow cone that SimHash can't partition without centering. The corpus
mean is computed once, cached, and recorded in the manifest.

### `tune-muvera` — choose the FDE setting on held-out queries

```bash
python -m src.cli tune-muvera
```

Sweeps `models.yaml`'s `muvera.tuning_grid` on held-out warmup queries. Selects the
smallest FDE that reaches `muvera.min_recall` (0.80) at the deepest depth on every tuning
dataset, writes `agg/muvera_tuning.json`.

### `calibrate-muvera` — validate the FDE on real embeddings

```bash
python -m src.cli calibrate-muvera --dataset nfcorpus
```

Required before trusting any `muvera_late_interaction_rescore` number — see
[Known limitations](#known-limitations). Compares FDE-prefetch recall against exact
`max_sim` and a bag-of-tokens baseline, over the whole corpus and the held-out warmup
queries.

### `plan` — print the execution plan, touch nothing

```bash
python -m src.cli plan --dataset nfcorpus
```

Prints the shuffled `(arm, depth)` unit list and every arm exclusion with its reason. No
Qdrant connection, no embeddings — the fastest way to check what a `run` would do.

### `doctor` — environment check

```bash
python -m src.cli doctor
```

Reports platform info, ONNX providers, Qdrant reachability and current memory, plus
warnings for anything unverified before a real run.

---

## Interpreting the numbers

Read this before quoting anything:

- **Latency is unloaded serial latency** — one client, one query in flight. A lower
  bound on real-world latency, not a throughput measurement.
- **Client and server share one machine** — the client/server split compares two
  processes on the same hardware, not two machines.
- **Memory is not additive per arm** — several arms share a vector (`late_interaction_rescore`
  and `cross_encoder` both read `dense`), so summing per-arm memory columns double-counts.
  The per-vector figures are the real quantities.
- **`dense`'s absolute latency is higher than a dense-only deployment would see** — every
  arm queries a collection that also holds the late-interaction and MUVERA vectors.
- **Everything runs on a single shard.**
- **Non-monotonic depth curves are preserved, not smoothed** — `*__gain_vs_depth.csv`
  reports the knee (shallowest depth within 1% of best) rather than masking a regression.
- **Results from different machines aren't comparable** — no fixed target architecture
  means absolute latency (and low-order-bit quality differences) is tied to whatever
  machine produced it. The embedding cache refuses a cross-architecture hit outright.

---

## Validation gates

Pre-run correctness checks, not part of a run — ordinary pytest tests marked `gate` and
`slow`, needing Docker, the models, and a live Qdrant. Run once before trusting results,
again after changing models, datasets, or indexing.

| Gate | Checks | Status |
|---|---|---|
| **1** `test_gate1_baseline` | Each baseline reproduces a published NDCG@10 on NFCorpus (`dense`: model card; `bm25`: BEIR paper). Catches broken qrels parsing, ID mapping, or metric calls. | ✅ `dense` passes; `bm25` pending its first run on 1.20 |
| **2** `test_gate2_arms_distinct` | No two arms return identical rankings; every arm runs or is excluded with a reason. | ✅ passes |
| **3** `test_gate3_hnsw_variance` | Two independent index builds agree within 0.005 NDCG@10 — establishes the noise floor. | ✅ passes |
| **4** `test_gate4_unattended` | Pipeline runs end to end on a small subset, resumes idempotently, aggregates cleanly. | ✅ scoped to a subset |
| **MUVERA** `test_muvera_calibration` | FDE prefetch beats bag-of-tokens and recovers ≥80% of exact top-10 at the deepest depth. | ❌ fails for LateOn — see [Known limitations](#known-limitations) |

**Gate 1 references**: `dense` — published 0.34305 NDCG@10 on NFCorpus, tolerance 0.02,
all 323 judged queries, BEIR's `{title} {text}` convention. `bm25` — BEIR paper's 0.325
(Elasticsearch), same tolerance, through Qdrant's BM25 text index.

The FDE setting is chosen once by `tune-muvera` and applied to every dataset. A dataset
whose calibration falls short is reported, not dropped:
`muvera.require_calibration: false` runs the arm anyway and records the verdict in the
manifest; set it `true` to gate per dataset instead. Both count only for the
late-interaction model they were calibrated on — a new model requires recalibration.

```bash
pytest -m "not slow"      # unit tests — no Docker, no model downloads, seconds
pytest -m gate            # validation gates — needs the live stack, ~18 min
```

---

## Output

```
results/
├── raw/       one JSONL per (dataset, arm, depth); one line per query per repeat
├── agg/       aggregate CSV, gain-vs-depth curve, significance table, memory JSON
├── queries/   the persisted query sample, its warmup holdout, and the ID map
└── manifest.json
```

Raw records are appended per query with periodic fsync, so a crash costs at most a
handful of queries, never a whole unit's worth of work. A resume is refused outright if
the plan digest (arms × depths × dataset × seeds × model config) doesn't match the
existing run directory.

Aggregation happens only at analysis time from raw records — nothing is pre-averaged.

`results/manifest.json` is the reproducibility record: pinned versions, seeds, model file
digests, hardware description, every arm exclusion with its reason. Report it alongside
any published number.

---

## Known limitations

- **MUVERA does not pass calibration on LateOn's tokens.** On the full NFCorpus corpus
  the FDE beats bag-of-tokens at every depth but recovers only 68% of exact MaxSim top-10
  by depth 500 (recall@10: 0.126; recall@500: 0.676), below the 80% threshold. ColbertV2's
  tokens passed the same settings (recall@500: 0.852). `muvera_late_interaction_rescore`
  is gated per dataset — `resolve_arms` excludes it unless `muvera.calibrated: true` or a
  passing calibration artifact exists. Re-run `calibrate-muvera` for any dataset whose
  token/document distribution might differ meaningfully. At 10,240 float32 dimensions,
  MUVERA's storage footprint also isn't dramatically smaller than the multivectors it
  prefetches for.
- **Learning-to-rank is deliberately not an arm.** It needs per-dataset trained models
  and engineered features this benchmark doesn't compute — a separate study, not a fifth
  row here.
- **Memory figures are Qdrant's own estimates, read once.** `ram_bytes` runs 10–15% low;
  `cached_bytes` is Unix-only. `disk_bytes` is the most reliable column. On a small
  collection, per-vector disk figures are dominated by segment pre-allocation and only
  separate cleanly once a corpus is large enough to fill segments.

Arms where reranking loses, or a depth where quality plateaus, are the deliverable — not
a result to tune away.
