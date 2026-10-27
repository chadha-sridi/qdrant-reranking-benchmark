# Reranking Benchmark for Qdrant

A reproducible benchmark evaluating reranking strategies built around Qdrant, measuring
retrieval quality, latency, and memory cost across corpus sizes.
Experiments can be run at different candidate depths to see how each strategy scales.

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
- **Docker**, with the daemon running. `docker compose` (plugin or standalone) is used
  if present; otherwise the code falls back to plain `docker run` automatically.
- **~2 GB disk** for model weights (dense + late interaction + cross-encoder, downloaded on
  first use via FastEmbed/Hugging Face Hub) plus space for corpus caches and Qdrant
  storage — a few hundred MB for the small datasets, much more for Quora.
- **Network access** to Hugging Face Hub, for both the BEIR datasets and the model
  weights. No account or token is required; you'll see a harmless "unauthenticated
  requests" rate-limit warning.
- No GPU required or used — everything runs CPU-side (ONNX Runtime via FastEmbed).
  Note that FastEmbed serves its own ONNX conversions, which for some models are
  quantized rather than full precision — relevant if you compare a score here against
  a published PyTorch figure (see [Validation gates](#validation-gates)).

There is deliberately **no fixed target machine**. `docker-compose.yml` runs Qdrant
native to whatever host executes it — an Apple
Silicon Mac gets a native arm64 container, an x86 Linux box gets native amd64, neither
is emulated. Latency numbers are only meaningful for the machine that produced them;
`results/manifest.json` records which one that was.

On macOS, Docker runs inside a VM sized independently of the Mac, and that VM — not the
Mac — is what Qdrant gets. Colima's default is 2 CPUs and 2 GiB, which starves the
server side while the client (embedding, the cross-encoder) has the whole machine. Size
the VM so each dataset's collection stays resident; this repo's results use 8 CPUs and
10 GiB on a 10-core, 16 GB M4:

```bash
colima stop && colima start --cpu 8 --memory 10
```

The VM's CPUs are not reserved: queries run one at a time, so an idle Qdrant leaves those
cores to the client. The manifest records the VM's CPUs and memory under
`qdrant.server_resources`, and `run` refuses a collection whose working set exceeds 80%
of that memory, since the arms reading the large vectors (late interaction, MUVERA) would then be
measured on disk while `dense` would not.

Qdrant's storage is a **Docker named volume**, not a host bind mount. This is a
measurement requirement, not a preference: a bind mount from macOS reaches the container
through the VM's virtiofs passthrough, where Qdrant reports `Filesystem check failed …
FUSE filesystems may cause data corruption` and falls back off multi-mmap with `you may
see reduced performance`. Vector storage is mmap-heavy, so that fallback lands directly
on the latency and memory figures this benchmark exists to produce. On a named volume the
same check reports `ext4` and both warnings disappear. The tradeoff is that storage is no
longer browsable from the host — inspect it with
`docker exec reranking_benchmark_qdrant ls /qdrant/storage` instead. `run.qdrant.storage_volume`
must match the `name:` under `volumes:` in `docker-compose.yml`.

---

## Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"          # or: uv sync

# 2. Start Qdrant (pinned to v1.20.0, the first release with BM25 scoring on a text index). On macOS, size the Docker VM first (see above).
docker compose up -d

# 3. Sanity-check the environment
python -m src.cli doctor

# 4. Smoke test — fast, 2 arms, capped queries, no CLI needed
python scripts/smoke.py

# 5. A real run on the smallest dataset
python -m src.cli run --dataset nfcorpus
```

The package also installs a `benchmark` console script (`benchmark run ...`) if you
installed with `pip install -e .`; every example below uses `python -m src.cli`
instead, since that works without installing anything.

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
├── cli.py              the `run`/`aggregate`/`memory`/`calibrate-muvera`/`plan`/
│                       `doctor` commands
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
object (`src/config.py`) — nothing else in the codebase reads YAML. A quick guide to
what lives where, so you know which file to edit:

| File | Controls |
|---|---|
| `run.yaml` | seeds, how many queries are sampled and how they're split, repeats, candidate depths, Qdrant connection and HNSW/indexing thresholds, which arms run by default |
| `models.yaml` | model names and dimensions, late-interaction/MUVERA parameters, the pinned Gate 1 reference score |
| `datasets.yaml` | which BEIR datasets are enabled, their HF source, corpus size, which representations to index, per-dataset depth overrides |

Most of these values are also overridable per invocation via CLI flags (below) without
editing a file — flags win, and every effective value (overridden or not) is recorded
in `results/manifest.json` for that run.

---

## Datasets

Five BEIR datasets, run smallest corpus first.

| Dataset | Corpus size | Description |
|---|---:|---|
| **NFCorpus** | 3,633 | Medical/nutrition IR — NutritionFacts queries against PubMed abstracts. The only graded dataset here (0/1/2) and by far the densest relevance judgments at 38 relevant docs per query, which makes it the one dataset with real headroom at deep candidate depths. Doubles as the Gate 1 reference and smoke-test target. |
| **SciFact** | 5,183 | Scientific claim verification — retrieve abstracts that support or refute a claim. ~1.1 relevant docs per query, so effectively one correct answer each. |
| **SCIDOCS** | 25,657 | Scientific paper retrieval from citation context. Exactly 5 relevant docs per query, always — an unusually uniform relevance profile. |
| **FiQA-2018** | 57,638 | Financial opinion question answering over investor forum posts. ~2.6 relevant docs per query. |
| **Quora** *(out of the default set)* | 522,931 | Duplicate-question retrieval. ~1.6 relevant docs per query. Largest corpus here, so late-interaction multivectors are forced on-disk — the only dataset exercising that regime. Kept out of `--dataset all` and capped at 500 queries because all 10,000 would be ~80% of the benchmark's runtime; run it explicitly with `--dataset quora`. |

Together they span 144× in corpus size and 1.1→38 in relevance density, so a strategy
that only wins on sparse-judgment or only on dense-judgment data can't hide.

**Query counts.** Every dataset in the default set runs every judged query it has:

| Dataset | Judged queries | Warmup holdout | **Evaluated** |
|---|---:|---:|---:|
| NFCorpus | 323 | 50 | **273** |
| SciFact | 300 | 50 | **250** |
| SCIDOCS | 1,000 | 50 | **950** |
| FiQA-2018 | 648 | 50 | **598** |
| | | | **2,071 total (default set)** |
| Quora *(separate run)* | 500 of 10,000 | 50 | **450** |

Warmup queries are held out purely so no evaluated query is served from a cache the
warmup pass warmed — **nothing is trained**, there is no train/test split.

Uncapping the default set is a deliberate precision-over-compute choice: more queries
only ever tighten confidence intervals. Quora is the one exception — its 10,000 queries
carry ~80% of the benchmark's runtime for a dataset that saturates by depth 50, so it
carries a `query_cap` of 500 and sits out of the default set. Metrics are computed per
dataset and never pooled across them, so running Quora separately does not affect any
other dataset's numbers.

`queries.cap` in `config/run.yaml` is `null` and is a smoke-test knob that caps every
dataset at once; a dataset's own `query_cap` in `config/datasets.yaml` takes precedence.

Only NFCorpus ships graded relevance; the loader never collapses grades and asserts the
observed grade set matches config, which would catch a parser that silently flattened
NFCorpus's 0/1/2 down to binary.

---

## Arms

**Two first stages, each a baseline, each reranked by the same rerankers.** `dense`
(bge-small) and `bm25` (sparse) are both first-stage retrievers; every reranking arm
starts from the candidates of one of them and is named after it. Each arm's aggregate
row carries a `baseline` column naming the arm its gain is measured against.
Hybrid-search/fusion strategies (RRF, DBSF, weighted RRF) are out of scope for a
*reranking* benchmark: they combine retrievers rather than rescore a candidate set with
a richer relevance signal.

| Arm | Baseline | Sweeps depth | Mechanism |
|---|---|:---:|---|
| `dense` | itself | | A single dense HNSW search, no reranking stage |
| `late_interaction_rescore` | `dense` | ✓ | Dense prefetch → late-interaction `max_sim` |
| `muvera_late_interaction_rescore` | `dense` | ✓ | MUVERA prefetch → exact `max_sim` (⚠️ see [Known limitations](#known-limitations)) |
| `cross_encoder` | `dense` | ✓ | Dense prefetch → client-side cross-encoder rerank |
| `bm25` | itself | | A single BM25 search, scored by Qdrant on a full-text index over the document text, no reranking stage |
| `bm25_late_interaction_rescore` | `bm25` | ✓ | BM25 prefetch → late-interaction `max_sim` |
| `bm25_cross_encoder` | `bm25` | ✓ | BM25 prefetch → client-side cross-encoder rerank |

All seven run by default. Both families run in the same interleaved pass against the
same collection, so the two baselines are measured under identical conditions and the
significance table compares every pair, across families too. To run one family only,
name its arms: `--arms bm25,bm25_late_interaction_rescore,bm25_cross_encoder`.
`muvera_late_interaction_rescore` has no BM25 counterpart: MUVERA is a first stage of its
own, so it is measured against `dense`, the other single-vector first stage.

**BM25 is scored by Qdrant on a full-text index (Qdrant 1.20+).** The `text` payload
field, which every point already stores for the cross-encoder, carries a full-text
index with BM25 scoring: word tokenizer, lowercasing, English Snowball stemming and
English stopwords (Qdrant's default is lowercasing only, a weaker baseline than
standard English BM25). Queries are `text` queries with `k=1.2`, `b=0.75`. Qdrant
tokenizes documents and queries itself and derives IDF and the average document length
from the indexed corpus; with one shard those are exact corpus statistics. Nothing is
computed client-side, so BM25 has no query-embedding time: its tokenization is inside
Qdrant's own time. The index sets `enable_hnsw: false`, so it adds no links to the dense
HNSW graph every dense arm searches.

**The dense model is a runtime parameter, not just a config edit.** Default is
`BAAI/bge-small-en-v1.5`; override with `--dense-model NAME` (and `--dense-model-dim N`
if FastEmbed's own registry can't resolve the dimension for an unlisted model). It
replaces the `dense` first stage only; the `bm25` family is unaffected. Overriding it
invalidates the pinned Gate 1 reference score for `dense`, and the CLI says so.

**"Late interaction" names the mechanism, not a model.** Any multivector model scored
with MaxSim goes under `late_interaction` in `config/models.yaml`; it is currently
`colbert-ir/colbertv2.0`, and the measurements quoted in this README are from that
model.

**The late-interaction arms use `max_sim`**, the only multivector comparator Qdrant
offers. The late-interaction vector is indexed with HNSW `m=0` — required, not optional, since `max_sim`
isn't symmetric — so it's never used for ANN prefetch, only for rescoring a candidate
set that arrived some other way. That's also why `muvera_late_interaction_rescore` is *exact* MaxSim:
MUVERA supplies the (approximate) prefetch, MaxSim an exact rescore over it.

Exclusions (representation missing for a dataset, arm gated, etc.) are always
reported, never silently dropped — run `python -m src.cli plan` to see them for any
dataset without touching Qdrant.

---

## Candidate depths

Depths are per dataset, set from each one's measured relevance density rather than by
feel. Override globally with `--depths`, or per
dataset via `depths:` in `config/datasets.yaml`.

| Dataset | rel/query (avg / med / p75) | Depths |
|---|---|---|
| **NFCorpus** | 38.2 / 16 / 40 | 10, 25, 50, 100, 200, **500** |
| SciFact | 1.1 / 1 / 1 | 10, 25, 50, 100, 200 |
| SCIDOCS | 4.9 / 5 / 5 | 10, 25, 50, 100, 200 |
| FiQA-2018 | 2.6 / 2 / 3 | 10, 25, 50, 100, 200 |
| Quora | 1.6 / 1 / 1 | 10, 25, 50, 100, 200 |

**NFCorpus is the only dataset that sweeps to 500**, because it is the only one still
gaining quality at that depth — its judgments are an order of magnitude denser than
anything else here (p75 of 40 relevant docs, max 475). It is also the cheapest place to
sweep deep: 323 queries, smallest corpus.

For the other four, every relevant document sits within depth 50, so depth 200 is
already ~4× past saturation — enough to *show* the plateau, which is the deliverable.
Adding 500 there would contribute a second flat point for 56% of the compute, since
runtime scales with the **sum** of the depths, not their count.

Two caveats for any write-up: depth 500 on NFCorpus reranks ~14% of that corpus, a far
higher ratio than production would use; and the relevance figures above are a *capacity
ceiling* from qrels, not first-stage recall — whether the dense baseline actually places
relevant documents in the candidate set is measured, not assumed.

---

## Cost and runtime

All figures measured on this machine (Apple M4, 16 GB RAM, CPU-only). The
cross-encoder figure is measured from 4,110 real NFCorpus records and is consistent
across every candidate depth (28.6–29.1 pairs/sec at depths 10 through 500):

| | Measured |
|---|---:|
| Cross-encoder rerank | **28.8 pairs/sec** |
| Dense embedding | 121 docs/sec |
| ColBERT embedding | 87 docs/sec |

Cross-encoder throughput depends strongly on document length, and these are long
PubMed-style abstracts; expect a different figure on short-document corpora.

**One-time, cached** — corpus embedding (615,042 documents across the five datasets):

| | Hours |
|---|---:|
| Dense (bge-small) | ~1.4h |
| ColBERT | ~2.0h |
| **Total** | **~3.4h** for all five (Quora alone is ~2.9h; the default set is ~0.5h) |

BM25 needs no embedding: Qdrant builds its index from the stored text during upload.

**Per full run** — querying at `repeats: 1`, all seven arms, every evaluated query. The
two cross-encoder arms are ~98% of it, each costing what `cross_encoder` alone measured
in the four-arm run; the figures below are that run's measurements with the second
cross-encoder arm added, not yet re-measured:

| Dataset | Depths | Evaluated queries | Est. hours |
|---|---|---:|---:|
| nfcorpus | 10/25/50/100/200/500 | 273 | ~5.5h |
| scifact | 10/25/50/100/200 | 250 | ~2.4h |
| scidocs | 10/25/50/100/200 | 950 | ~7.7h |
| fiqa | 10/25/50/100/200 | 598 | ~4.9h |
| **`--dataset all` total** | | **2,071** | **~20.5h** |
| quora *(separate, capped)* | 10/25/50/100/200 | 450 | ~3.5h |
| quora *(uncapped, if you lift the cap)* | 10/25/50/100/200 | 9,950 | ~76h |

Quora sits out of the default set for exactly this reason: uncapped it is ~80% of the
total for a dataset that saturates by depth 50.

Measured per-query cost at depth 500 shows where it all goes:

| Arm | ms/query @ depth 500 |
|---|---:|
| **`cross_encoder`** | **17,319** |
| `muvera_late_interaction_rescore` | 129 |
| `late_interaction_rescore` | 112 |
| `dense` (no depth) | 10 |

The cross-encoder arms dominate because they are the only ones that rerank client-side: a
transformer forward pass per (query, document) pair on CPU, where every other arm
pushes the work into Qdrant as vector arithmetic. Its cost scales linearly with the
**sum** of the swept depths, so the deepest point dominates — depth 500 alone is 56% of
NFCorpus's cross-encoder cost.

Levers, in order of effect:

| Change | Saves | Costs you |
|---|---|---|
| Lift Quora's 500-query cap | costs ~72h | tighter CI on the one large-corpus dataset |
| Cap the cross-encoder arms at depth ≤100 | ~75% of their cost | the deep end of their depth curves only |
| Drop NFCorpus depth 500 | ~2.6h | the only depth point past 200 with real headroom |
| Run one family only (`--arms`) | ~10h of the ~20.5h | the other first stage |
| Drop both cross-encoder arms | ~20h of the ~20.5h | your only client-side reranking arms |

Depth 500 on NFCorpus costs 17.4s/query per cross-encoder arm — ~1.3h each, roughly
half of NFCorpus's total. It is kept because NFCorpus (38 relevant docs/query) is the only
dataset with real quality headroom past depth 200.

---

## Collection layout and ingestion

**One collection per dataset, named after the dataset**, holding every representation
as a **named vector** on the same points — `dense`, `muvera`, and `late_interaction`, all
fp32, plus a BM25-scored full-text index on the `text` payload. Every arm queries that one collection, so each arm's prefetch
reads the same HNSW graph, the same payload, and the same page cache; the only thing
that differs between arms is the strategy under test. One collection per representation
would make arm-to-arm latency incomparable, which is the benchmark's primary axis.

No vector carries a quantization config, so every arm reads full precision end to end
and no arm's numbers can be a comparison of two different precisions. `late_interaction` is
deliberately built with `m=0` — no HNSW
graph at all — because `max_sim` is not a metric space, so multivectors are only ever
used for rescoring, never for prefetch.

`hnsw.full_scan_threshold_kb` is set to **10** rather than Qdrant's default of 10,000.
At the default, segments under ~10 MB are brute-forced regardless of the index, which
would silently full-scan NFCorpus and SciFact (~5–8 MB of dense vectors) while the
larger corpora used ANN — turning the corpus-size curve into a comparison of two
different algorithms. 10 is the smallest value Qdrant accepts (it rejects 0).

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

The order matters. Restoring the threshold *after* the upload is what triggers the index
build; setting it to 0 after uploading instead leaves the collection permanently
unindexed, which reports `status: green` while every search is a brute-force scan.

**Interruption is recoverable.** Point ids are the corpus enumeration order, so they are
deterministic: a re-sent point overwrites itself rather than duplicating. `--resume-ingestion`
reuses a config-matching collection and re-sends only the missing ids. The collection is
dropped and rebuilt only when its vector config does not match what the run asks for,
since two vector configs cannot be mixed in one collection.

**Verification is a hard failure, not a warning.** Silent under-ingestion is the failure
mode that matters: a short collection inflates latency (a smaller index is faster) and
deflates recall (missing documents can never be retrieved), while looking exactly like a
valid run. An upload that times out or drops its connection partway leaves exactly that,
with nothing else to notice it, and no downstream check can detect it after the fact —
so ingestion aborts on any mismatch.

The embedding step is the one part that does **not** resume: the cache is written per
representation once the whole corpus is embedded, so a crash partway through Quora's
ColBERT pass (~2.9 h) loses that pass. Once cached, it is reused for free.

---

## What gets measured

**Latency**, independent timers per query, recorded raw (per query, per pass) and
only aggregated at analysis time:

| Field | What it is |
|---|---|
| `t_embed_dense_ms` | client-side dense query embedding (`dense`, `late_interaction_rescore`, `cross_encoder`) |
| `t_embed_late_interaction_ms` | client-side late-interaction query embedding (the three `*_late_interaction_rescore` arms) |
| `t_embed_muvera_ms` | the MUVERA FDE transform of the late-interaction query tokens (`muvera_late_interaction_rescore` only) |
| `t_embed_ms` | sum of the four embedding buckets |
| `t_qdrant_ms` | client-observed round trip, summed across all Qdrant calls an arm makes |
| `t_qdrant_server_ms` | Qdrant's own self-reported processing time for the same calls |
| `t_qdrant_overhead_ms` | `t_qdrant_ms - t_qdrant_server_ms` — transport + response deserialization |
| `t_rerank_ms` | client-side reranking (non-zero only for the two cross-encoder arms) |
| `t_total_ms` | `t_embed_ms + t_qdrant_ms + t_rerank_ms` |

Each arm embeds its own query, so an arm's `t_embed_ms` is what a deployment running
that arm alone would pay. The per-encoder buckets put the encoders side by side on the
same queries: a bucket an arm does not use is recorded as 0.

Reported as p50/p95 across queries. p99 is not reported: at one pass per query there is
one latency sample per query, and a p99 off that would be false precision.

The server/overhead split matters more than it looks — a large share of the
client-observed round trip is transport and response deserialization, not search, so
attributing all of it to "Qdrant" would misstate the client/server cost split this
benchmark exists to measure.

Queries go over **gRPC**, and read Qdrant's self-reported processing time straight off
the response (`QueryResponse.time`). `QdrantClient.query_points()` discards that field,
so `run_query` calls the gRPC stub directly — which is what makes the server/transport
split measurable without paying JSON's serialization cost on every query.

**Memory**, read from Qdrant's own per-collection accounting
(`GET /collections/{name}/memory`, Qdrant 1.18+) on the same collection the arms
query. Every named vector is reported separately, and each is split into the pieces
that cost different things:

| Field | What it is |
|---|---|
| `storage` | the fp32 vectors themselves |
| `index` | the HNSW graph for that vector (≈0 for `late_interaction`, which has `m=0` and no graph) |
| `payload` | `doc_id` + `text`, reported as its own component rather than estimated |
| `payload_indexes` | one entry per indexed payload field; the `text` entry is BM25's index and is reported as the `bm25` representation |
| `id_tracker` | the remaining accounted-for structure |
| `n_token_vectors` | late-interaction token-vector total, since "per million points" is ambiguous by ~100× on a multivector collection |

Each is reported in four dimensions: `disk_bytes`, `ram_bytes` (non-evictable),
`cached_bytes` (page cache, Unix only), and `expected_cache_bytes` (what Qdrant wants
resident). `working_set_bytes` = `ram_bytes` + `expected_cache_bytes` is the "how much
RAM does this actually need" figure — neither input answers it alone, since
memory-mapped vectors report ~0 `ram_bytes` while still needing to be resident to be
fast.

Qdrant's numbers are estimates — its docs put `ram_bytes` 10–15% low, since allocations
inside third-party libraries aren't tracked — but they describe the actual collection the
arms query rather than a stand-in for it, and they attribute payload directly rather
than by subtracting one measurement from another.

**Quality**, via `ranx`: **NDCG@10** (primary) and **Recall@10**. Computed from
`run_idx 0` only — retrieval is deterministic given a fixed index, so counting repeated
passes would inflate a significance test's sample size. Pairwise significance uses
`ranx.compare` with both raw and Bonferroni-corrected verdicts.

Metrics are always recomputed post-hoc from the stored ranking, never baked in at
capture time, so the reported set can change without re-running anything.

**`recall_at_pool_depth`**, the ceiling on Recall@10: what fraction of a query's
relevant documents were even present in the prefetch pool at that depth, before
rescoring ran. It separates "the reranker failed to promote a relevant document it
had" from "the pool never contained it, so go deeper". Captured at query time (the
pool itself is never persisted, only this count), and `null` for `dense`, which has no
separate pool. At depth 10 it equals Recall@10 exactly — there the pool *is* the
output.

---

## Execution order and warmup

There is no hardcoded run order and no arm-by-arm block execution. Per dataset:

```
run_dataset(dataset):
    units = build_plan(arms, dataset)        # every (arm, depth) pair; sweeping arms get
                                              # one unit per configured depth, others get one
    warmup(units)                            # once, before any measurement — see below

    for run_idx in 1..repeats:                # 3 by default
        query_order = seeded_shuffle(eval_queries)     # NEW random order every repeat
        for query in query_order:
            unit_order = seeded_shuffle(units)          # NEW random order every query
            for unit in unit_order:
                result = unit.arm(query, unit.depth)     # embed -> Qdrant round trip -> rerank
                append_jsonl(arm, depth, query, run_idx, result)
```

- **`units`** — the flat list of (arm, depth) pairs to run. Non-sweeping arms (e.g.
  `dense`) contribute one unit with `depth=None`; sweeping arms (`late_interaction_rescore`,
  `muvera_late_interaction_rescore`, `cross_encoder`) contribute one unit per
  configured depth. `build_plan` also applies an initial shuffle, but that ordering is
  superseded by the per-query reshuffle below — it only matters for the plan digest
  used to validate a resume.
- **`unit_order`** — reshuffled fresh for **every individual query**, seeded
  deterministically from `(plan seed, run_idx, query index)`. Without this, whichever
  arm happened to run first for a query would always pay that query's cold
  index-region cost, while every arm after it benefited from pages the first one just
  warmed — a systematic, always-same-direction bias favoring later arms. Reshuffling
  per query means that advantage lands on a different arm each time and washes out
  across the run.
- **`query_order`** — reshuffled fresh for **every repeat** (seeded independently
  from the unit shuffle). This doesn't affect arm-vs-arm comparisons — those are
  already protected by the unit reshuffle — but without it, whichever queries sort
  last in a fixed order would be measured under the same point in the run's drift
  (thermal, memory pressure) on every single repeat, biasing per-query and percentile
  latency statistics.
- Both reshuffles are why interleaving arms rather than running them block-by-block
  actually holds in practice: any slow-moving confound
  during the run gets smeared thinly across every arm and every query, rather than
  concentrating on whichever arm/depth happened to run during a bad stretch.

**Warmup**, compactly: a discarded pass — every unit run once against a small held-out
query pool, results thrown away — executed once per dataset before any repeat begins.
It exists so the *measured* run doesn't pay one-time costs that have nothing to do
with the arm being tested: model lazy-loading, connection setup, unloaded index
pages, CPU frequency ramp-up. Running it once, before repeats start, and against
every arm/depth equally, means no arm starts the measured run at a warmth
disadvantage relative to another.

The **50 warmup queries are a fixed, seeded holdout, persisted per dataset** —
deterministic across runs, and explicitly separate from the queries actually
evaluated, so no evaluated query's latency is ever measured from a cache that same
query already warmed. The tradeoff worth knowing: because warmup queries differ from
eval queries, they don't necessarily touch the exact same index regions eval queries
will hit — the benefit is general system warmth more than "this exact eval query is
pre-warmed." The alternative (warm on the eval queries themselves, discarding only the
first pass) would warm the exact regions measured, at the cost of spending real
eval-query budget on it. This benchmark takes the holdout.

---

## Running the benchmark

All commands below use `python -m src.cli <command>` — swap in `benchmark <command>`
if you installed the console script. Every command is safe to re-run; `run` skips
records already on disk unless you pass `--no-resume`.

### `run` — index, measure memory, execute, aggregate

```bash
python -m src.cli run --dataset all              # the four-dataset default set, ~10.4h
python -m src.cli run --dataset quora            # Quora on its own, capped at 500 queries
python -m src.cli run --dataset nfcorpus
python -m src.cli run --dataset all --arms dense,late_interaction_rescore
python -m src.cli run --dataset fiqa --dense-model BAAI/bge-base-en-v1.5
```

`--dataset all` runs the default set only. Quora is configured but excluded from it
(`enabled: false`), because uncapped it is ~80% of the benchmark's runtime. **Naming a
dataset explicitly runs it regardless of that flag**, which is how Quora is run — on its
own, without redoing the other four. It carries `query_cap: 500` in
`config/datasets.yaml`; delete that line to run all 10,000.

Running Quora separately does not affect any other dataset's numbers: each dataset gets
its own collection, its own shuffled execution plan, and its own aggregation. Metrics
are never pooled across datasets, so there is nothing for a separate run to bias.

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `nfcorpus` | dataset name, or `all` for every enabled dataset |
| `--arms` | `all` | comma-separated arm names, or `all` for the configured default set — `arms.enabled` in `config/run.yaml`, not literally every registered arm (see [Arms](#arms)) |
| `--depths` | config | override the depth sweep, e.g. `10,50,100` — wins over every dataset's own override too |
| `--resume` / `--no-resume` | resume | skip records already on disk |
| `--overwrite` | off | with `--no-resume`, delete this dataset's existing raw records first, instead of refusing to run |
| `--measure-memory` / `--no-measure-memory` | on | read Qdrant's per-vector memory accounting off the built collection |
| `--resume-ingestion` | off | reuse a matching existing collection and re-send only missing points, instead of wiping storage and rebuilding — use after a crash mid-upload |
| `--allow-unverified-baseline` | off | proceed when the dense model's Gate 1 reference is marked `verified: false` (it is verified by default, so this is only needed after overriding the model) |
| `--dense-model` | config | override the dense first-stage model the dense-family arms rerank |
| `--dense-model-dim` | auto | dimension of `--dense-model`, if FastEmbed's registry can't resolve it |
| `--analyse` / `--no-analyse` | on | aggregate after the run completes |
| `--allow-disk-bound` | off | run even when a collection's working set exceeds 80% of Qdrant's memory; its large-vector arms are then measured on disk |
| `--results-dir` | `results` | write raw records, aggregates, query samples and the manifest under another directory, so a new run never touches a recorded one; the embedding cache is shared |
| `--allow-partial` | off | aggregate even if some `(arm, depth)` units have incomplete records — for inspecting an interrupted run, never for a result you intend to trust |

`aggregate` accepts the same `--allow-partial` flag.

### `aggregate` — re-aggregate without re-querying

```bash
python -m src.cli aggregate --dataset nfcorpus
```

Rebuilds every CSV from the raw JSONL already on disk. Use this after tweaking
analysis code, or to regenerate output that was deleted. Takes the same
`--results-dir` as `run`.

### `memory` — memory phase only

```bash
python -m src.cli memory --dataset nfcorpus
```

Builds (or reuses) the dataset's collection, reads Qdrant's per-vector memory
accounting off it, and prints the record as JSON.

### `calibrate-muvera` — validate the FDE on real embeddings

```bash
python -m src.cli calibrate-muvera --dataset nfcorpus --n-docs 2000 --n-queries 25
```

Required before trusting any `muvera_late_interaction_rescore` number — see
[Known limitations](#known-limitations). Compares FDE-prefetch recall against exact
`max_sim` and against a trivial bag-of-tokens baseline on real late-interaction embeddings.

### `plan` — print the execution plan, touch nothing

```bash
python -m src.cli plan --dataset nfcorpus
python -m src.cli plan --dataset quora --arms late_interaction_rescore,muvera_late_interaction_rescore
```

Prints the shuffled `(arm, depth)` unit list and every arm exclusion with its reason.
No Qdrant connection, no embeddings, no side effects — the fastest way to check what a
`run` invocation would actually do.

### `doctor` — environment check

```bash
python -m src.cli doctor
```

Reports platform info, ONNX providers, Qdrant reachability and current memory, plus
warnings for anything unverified before a real run (Gate 1 reference, MUVERA
calibration).

---

## Interpreting the numbers

Read this before quoting anything — these are measurement choices, not incidental
details, and each changes what a number means.

- **Latency is unloaded serial latency.** One client, one query in flight, no
  pipelining — a lower bound on real-world latency, not a capacity measurement.
  Throughput under load is out of scope.
- **Client and server share one machine.** Qdrant runs in the Docker VM and the client
  on the host, so the client/server split compares two processes on the same hardware,
  not two machines. Every collection fits in the server's memory, so no arm is
  measured on disk.
- **Memory and latency describe the same collection, but memory is not additive per
  arm.** Both come from the one collection per dataset, so they describe a single
  deployment. But several arms share a vector (`late_interaction_rescore` and `cross_encoder`
  both read `dense`; both dense- and BM25-first-stage late-interaction arms read `late_interaction`), so summing per-arm memory
  columns would double-count. The
  per-vector figures are the real quantities; the per-arm columns are a convenience view.
- **The absolute `dense` latency is higher than a dense-only deployment would see.**
  Every arm queries the collection that holds all four representations, which is what
  makes arm-to-arm comparison clean, but it means the baseline's absolute number carries
  the cost of sharing a collection with the late-interaction and MUVERA vectors.
- **Everything runs on a single shard.** Multi-shard behaviour, including fusion
  divergence across shards, is out of scope.
- **Non-monotonic depth curves are preserved, not smoothed.** Reranking is not
  guaranteed to improve monotonically with candidate depth, and `*__gain_vs_depth.csv`
  reports the knee (shallowest depth within 1% of the best score) rather than masking
  a plateau or a regression.
- **Results from different machines aren't directly comparable.** No fixed target
  architecture (see [Prerequisites](#prerequisites)) means absolute latency — and to a
  lesser extent quality, since ONNX floating-point results can differ in low-order
  bits across instruction sets — is tied to whatever machine produced it. The
  embedding cache refuses a cross-architecture cache hit outright rather than risk a
  silent mismatch.

---

## Validation gates

Pre-run correctness checks, not part of a run. Each guards one assumption that, if
broken, would make every downstream number wrong while still looking plausible. Run them
once before trusting results, and again after changing models, datasets, or indexing.
They are ordinary pytest tests marked `gate` and `slow`, so they need Docker, the models,
and a live Qdrant.

| Gate | Checks | Status |
|---|---|---|
| **1** `test_gate1_baseline` | Each baseline reproduces a published NDCG@10 on NFCorpus over all 323 judged queries: `dense` the model card's figure, `bm25` the BEIR paper's. Catches broken qrels parsing, ID mapping, document construction, encoder or text-index setup, or metric calls. | ✅ `dense` passes; `bm25` pending its first run on 1.20 |
| **2** `test_gate2_arms_distinct` | No two arms return identical rankings, and every registered arm either runs or is excluded with a stated reason. Catches an arm silently collapsing into its own prefetch. | ✅ passes |
| **3** `test_gate3_hnsw_variance` | Two independent index builds of the same vectors agree within 0.005 NDCG@10. Establishes the noise floor below which no arm-to-arm difference is interpretable. | ✅ passes |
| **4** `test_gate4_unattended` | The pipeline runs end to end on a 5-query, 2-depth subset, resumes idempotently, and aggregates into its artefacts. Catches resume bugs (double-counting) and missing output. | ✅ scoped to a subset |
| **MUVERA** `test_muvera_calibration` | The FDE prefetch beats a bag-of-tokens baseline and recovers ≥80% of the exact late-interaction top-10 at the deepest sweep depth. Without it, `muvera_late_interaction_rescore` would measure its own recall ceiling rather than reranking quality. | ✅ passes |

**On Gate 1's published-vs-measured gap.** The gate passes with measured NDCG@10 =
**0.3314** against the published **0.34305** (Δ0.0117), inside a deliberately widened
0.02 tolerance. The gap has one identified cause: FastEmbed distributes this model only
as a quantized ONNX build (`qdrant/bge-small-en-v1.5-onnx-q`, 66.5 MB against ~133 MB
for full precision, with no fp32 variant offered), while the published figure was
produced with the full-precision model. Document construction was ruled out as a cause —
the loader already follows BEIR's `{title} {text}` convention. A −3.4% relative shortfall
is what a quantized encoder predicts; a genuine qrels or ID-mapping fault would drop NDCG
far more, which is what keeps the widened tolerance meaningful rather than vacuous.

**On Gate 1 for `bm25`.** The reference is BM25 on NFCorpus from the BEIR paper
(0.325, Elasticsearch), tolerance 0.02, measured through Qdrant's BM25 text index over
all 323 judged queries.

MUVERA's calibration is in place: it uses fastembed's official `Muvera` postprocessor and
is validated on the full NFCorpus corpus (see
[Known limitations](#known-limitations)). `muvera_late_interaction_rescore` is still gated per dataset —
`resolve_arms` excludes it on any dataset without a passing calibration artifact or an
explicit `muvera.calibrated: true` override, so it cannot silently run uncalibrated even
if you add a new dataset later.

Run the test suite:

```bash
pytest -m "not slow"      # unit tests — no Docker, no model downloads, seconds
pytest -m gate            # validation gates — needs the live stack, ~18 min
```

The gates share one NFCorpus collection across the session and reuse cached embeddings,
so the first run pays the index build and later ones do not.

---

## Output

```
results/
├── raw/       one JSONL per (dataset, arm, depth); one line per query per repeat
├── agg/       aggregate CSV, gain-vs-depth curve, significance table, memory JSON
├── queries/   the persisted query sample, its warmup holdout, and the ID map
└── manifest.json
```

Raw records are appended per query with a periodic fsync — not checkpointed per unit —
so a crash costs at most a handful of queries, never a whole `(arm, depth)` unit's
worth of work. A resume is refused outright if the plan digest (arms × depths ×
dataset × seeds × model config) doesn't match the existing run directory, so an
in-between config edit can't silently produce a half-old, half-new result set.

Aggregation happens only at analysis time, reading from the raw records — nothing is
averaged before it hits disk; storing raw per-query measurements rather than
aggregates is a hard requirement.

`results/manifest.json` is the reproducibility record: pinned versions, seeds, model
file digests, hardware description, every arm exclusion with its reason, and the
caveats above. Report it alongside any published number.

---

## Known limitations

Open items worth knowing before you lean on a number:

- **MUVERA is calibrated, but the margin is real and worth knowing.** `src/muvera.py`
  wraps fastembed's official `Muvera` postprocessor (`fastembed>=0.7.2`), the reference
  implementation. Calibrated on the full NFCorpus corpus (3,633 docs,
  80 queries): the FDE beats a bag-of-tokens baseline at every depth (recall@10: 0.352
  vs 0.161; recall@500: 0.852 vs 0.709). `muvera_late_interaction_rescore` is gated per dataset —
  `resolve_arms` excludes it unless `muvera.calibrated: true` or a passing
  `{dataset}__muvera_calibration.json` artifact exists — so adding a new dataset
  doesn't silently inherit NFCorpus's calibration. Re-run `calibrate-muvera` for any
  dataset whose token/document distribution might differ meaningfully. Also worth
  noting: at 10,240 float32 dimensions, MUVERA's storage footprint is not dramatically
  smaller than the ColBERT multivectors it prefetches for — report that from the
  memory table rather than tuning the FDE down until the comparison flatters.
- **Learning-to-rank is deliberately not an arm.** It would need per-dataset trained
  models, breaking the zero-training/drop-in-evaluable property every other arm has,
  and it wants engineered features (BM25 score among them) that this benchmark does
  not compute. Worth a separate study with its own methodology, not a fifth row
  in this table.
- **Memory figures are Qdrant's own estimates, read once.** Its docs put `ram_bytes`
  10–15% low, because allocations inside third-party libraries are not tracked, and
  `cached_bytes` is populated on Unix hosts only. `disk_bytes` is the most reliable
  column; treat `working_set_bytes` as an informed estimate rather than a measurement.
  Note also that on a small collection the per-vector disk figures are dominated by
  segment pre-allocation and are not meaningful — they only separate cleanly once a
  corpus is large enough to fill segments.

Arms where reranking loses, or a depth where quality plateaus, are the deliverable —
not a result to tune away.
