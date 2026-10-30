# Resolved Decisions

Every ambiguity, contradiction, and blank in `spec.md`, and how this implementation
resolves it. Each decision is recorded in `results/manifest.json` at run time so a
published result can be traced back to the choice that produced it.

Sections map to `spec.md`. `[C]` = resolved contradiction, `[U]` = filled-in blank,
`[D]` = design change from the spec.

---

## A. Architecture

### A1 `[C]` Collection layout: singletons measure memory, one combined collection serves all queries

`spec.md` §0.2 mandates one collection per representation; §4 recommends Qdrant-native
`prefetch` within one collection. Both cannot hold, because `colbert_rescore` and
`muvera_colbert` need two representations co-resident.

Resolution — two collection families per dataset, with disjoint purposes:

| Family | Name | Purpose | Queried? |
|---|---|---|---|
| Singleton | `{dataset}__{representation}` | memory attribution only | no |
| Combined | `{dataset}__combined` | every arm at query time | yes |

The singletons are built one at a time on a clean Qdrant, measured, and dropped
(§5's "freshly started Qdrant with only that collection loaded"). They are never
queried, so no cache warming can corrupt their numbers. The combined collection holds
`dense`, `colbert`, and `muvera` as named vectors and serves all arms. The `dense`
vector additionally carries a TurboQuant config so arms 1 and 2 can share it (K2).

Two consequences, both deliberate:

1. **Every arm's prefetch reads the same HNSW graph.** The `dense` baseline and the
   dense prefetch inside `colbert_rescore` hit byte-identical index structures, so
   arm-to-arm differences cannot be graph-construction noise. This removes a whole
   class of confound that per-arm collections would introduce.
2. **Memory and latency are measured on different objects.** The published memory
   figure is per-representation from the singletons; latency is from the combined
   collection. Stated explicitly in the README and manifest — they are not two views
   of one deployment, and a reader must not add them up.

The memory comparison is *not* corrupted by the combined collection, because the combined
collection is never measured and the singletons are never queried. But two limits follow,
and both are measured rather than asserted:

- **Absolute latency is inflated.** The `dense` p50 in the combined collection is not what
  a dense-only deployment would see: the ColBERT and MUVERA data share its segments. Arm-to-arm comparison is unaffected — that is the apples-to-apples the layout
  buys — but the absolute figure is pessimistic. Gate 4 now measures the dense p50 in the
  singleton versus the combined collection, after identical warmup, and writes the ratio to
  `results/agg/{dataset}__latency_inflation.json` to be published beside the latency table.
- **Singleton memory does not sum to deployment memory.** Payload is stored once per
  singleton and once more in the combined collection, so summing the singletons
  overstates. Subtract the payload baseline (B4) before adding anything up.

Indexing cost is one upsert pass per singleton plus one combined pass. Embeddings are
computed once and cached, so the extra cost is upsert and index-build only, not
inference. Per-dataset `measure_memory: false` skips the singleton phase where that
cost is not worth paying.

### A2 `[C]` Multi-round-trip arms: `t_qdrant_ms` is the sum of serial round trips

§0.3 says "Qdrant round trip", singular, but an arm may need more than one call. Rule:
`t_qdrant_ms` is the summed wall time of **all** Qdrant calls an arm makes, issued
**serially** from a single client. `n_qdrant_calls` is stored on every raw record so a
reader can tell a one-call arm from a two-call arm instead of guessing. No arm
parallelises its calls; a concurrent variant would measure the client's event loop as
much as Qdrant.

### A3 `[U]` Query concurrency: strictly serial, one client, one connection

Nothing in the spec pins this, and it changes what every latency number means.
Decision: one client, one query in flight, no pipelining. All reported latencies are
**unloaded serial latencies** — a lower bound on what a real deployment sees, not a
capacity measurement. §5's "cost per 1,000 queries" is therefore derived from serial
latency and assumes one query at a time, which **understates** achievable throughput
on an 8-vCPU box. The manifest records this and `README.md` states it beside the cost
column. Throughput-under-load is explicitly out of scope for v1.

---

## B. What gets measured

### B1 `[superseded by L1]` All arms evaluate on the same validation half

Was: the query sample split 50/50 into train/validation, with only the validation half
reported, so that `rrf_weighted`'s tuned weights and any other trained arm could not
leak. Now: **nothing trains**, so the split cost half the statistical power for no
benefit and was removed (L1). All arms still evaluate on one identical query set —
which is what made the paired significance test valid in the first place — but that set
is now everything except a small warmup holdout.

### B2 `[C]` Recall@100 is reported only where depth admits it

At depth 10 an arm can return at most 10 documents, so Recall@100 is structurally
capped rather than measured. Decision: an arm returns `min(100, depth)` results and
stores all of them. `ndcg@10`, `recall@10`, `mrr@10` are computed at every depth;
`recall@100` is emitted as `null` wherever `depth < 100` rather than as a misleadingly
small number. The `prefetch.limit >= limit + offset` guard is asserted in
`src/arms.py` before every call.

### B3 `[U]` Repetitions: 3 timing runs per query; quality from run 0 only

`run_idx` appears in the spec's raw schema and is never defined. Decision:
`repeats: 3` (configurable). Retrieval is deterministic given a fixed index, so
quality metrics are computed from `run_idx == 0` alone; all three runs feed the
latency percentiles. At a few hundred queries × 3 runs the p99 is drawn from over a
thousand samples, which is a defensible p99. Quality from one run avoids triple-counting
identical result lists into a significance test.

### B4 `[U]` Memory: three numbers, none of them RSS alone

RSS under-reports mmap'd segments, so a single number would be wrong in a direction
that flatters large collections. Each singleton records:

- `rss_delta_bytes` — container RSS after indexing minus a clean-Qdrant baseline
- `disk_bytes` — `du -sk` of the collection's storage directory
- `anon_delta_bytes` / `file_delta_bytes` — the cgroup anon vs. page-cache split

An earlier draft of this section also promised a `telemetry_bytes` field from Qdrant's
own reported sizes. It was never implemented — no such field existed on `MemoryRecord`
and the `telemetry()` helper was never called — so the claim was removed and the dead
helper deleted (DECISIONS L3) rather than left as a documented metric that does not
exist.

All of these are captured **after** the optimizer reports `green` and before any query.
Normalisation is **per million documents**, not per million stored vectors. For
`colbert` the token-vector count is recorded separately as `n_token_vectors`, because
per-million-points on a multivector collection is otherwise ambiguous by a factor of
~100.

**A payload-only baseline is measured first, and subtracted.** Every singleton stores the
same payload (`doc_id`, `text`), so payload inflates every representation by the same
absolute amount — which flatters the wide representations and
distorts every ratio in the memory table. On a small corpus with a 384-dim dense vector
the text payload can be a comparable share of the total. So the memory phase first indexes
a `{dataset}__payload_only` collection with `vectors_config=None` and no vectors at all,
then reports both the raw figure and `vector_only_rss_bytes` /
`vector_only_disk_bytes`. The latter pair is what answers "what does this representation
cost"; the raw figure is what a deployment actually pays.

### B5 `[U]` Warmup queries come from the train half

§6.4 asks for 50 discarded warmup queries but not where they come from. Drawing them
from the evaluation set would leave those queries cache-warm for the real run.
Decision: 50 queries sampled from `train`, so no evaluated query is ever pre-warmed.
Warmup runs once per Qdrant process per dataset, after the memory phase.

### B6 `[U]` Significance testing: Fisher randomisation plus explicit Bonferroni

`ranx.compare(max_p=0.05)` is used for the pairwise table. With 9 arms the pair count
makes uncorrected p-values misleading, so `src/metrics.py` also applies a Bonferroni
correction over the number of pairs actually compared and reports both the raw and
corrected verdict. Where the installed `ranx` accepts a `correction` argument it is
passed through; otherwise the correction is applied to the extracted p-values.

---

## C. Arms

**Sections C1, C3, C4 and C5 are superseded by K1 and K4.** They resolved contradictions
about arms that no longer exist — `formula_rrf`, `rrf_weighted`, `rrf`, `dbsf` — and
about a fusion prefetch limit that no longer exists. The reasoning is preserved in K1/K4
rather than duplicated here. C2 still stands.

### C1 `[superseded by K1, K4]` `formula_rrf`

Was: removed as a reported quality arm because its `boost_field` is hash-derived noise,
kept only for the sharding experiment. Now: the sharding experiment is gone (K4), so
the formula machinery and `boost_field` are deleted outright and the arm is not
declared anywhere.

### C2 `[superseded by N1]` `relevance_feedback` is declared but excluded by default

Was: the arm needed trained weights per (retriever, feedback model, collection)
triplet, and §4 says untrained weights give arbitrary results and the arm should be
omitted explicitly rather than shipped untrained. It was implemented, gated behind
`--include-relevance-feedback`, and by default recorded in `manifest.exclusions` with
that reason. Now: removed entirely (N1).

### C3 `[superseded by K1]` `rrf_weighted` is client-side

Was: weighted RRF fused in Python because Qdrant's `Fusion.RRF` exposes no per-input
weights, with its own train-split weight tuning. Now: the arm, its tuning
(`tune_rrf_weights`), and the `client_side_fusion` flag are all deleted (K1).

### C4 `[superseded by K1, K4]` Fusion arms use a fixed prefetch limit of 100

Was: `fusion_prefetch_limit: 100` for the non-sweeping fusion arms. Now: no fusion
arms exist and the config field is deleted. Every remaining arm except the baseline
sweeps candidate depth (K5).

### C5 `[superseded by K1]` Gate 2 is scoped

Was: Gate 2's arm-distinctness check needed an exemption for `(rrf, rrf_weighted)`
because tuning could legitimately select unit weights and make them identical. Now:
both arms are gone, the exemption list is removed, and Gate 2 checks every arm pair
unconditionally.

---

## D. Models and representations

### D1 `[superseded by M1]` MUVERA: our own FDE, pinned and seeded

The custom implementation below is no longer in use — replaced by fastembed's
official `Muvera` (M1), because this section's own finding (the FDE cannot be
validated on synthetic data at any setting) was a standing, unresolved doubt about a
hand-built implementation specifically. The finding itself, and the discipline it
forced (calibrate on real data, never trust synthetic tuning), still hold and are
preserved below as the reasoning that led to M1.

The spec gives no MUVERA configuration at all, and no pinnable library implementation
is assumed. `src/muvera.py` implements Fixed Dimensional Encoding from the MUVERA
paper over ColBERT token embeddings, seeded from `run.yaml`:

| Parameter | Value | Meaning |
|---|---|---|
| `k_sim` | 5 | SimHash bits → 32 space partitions |
| `n_reps` | 20 | independent repetitions |
| `d_proj` | 16 | per-partition projection dim |
| output dim | 10240 | `n_reps × 2^k_sim × d_proj` |
| distance | dot | FDE approximates Chamfer/`max_sim` under dot product |

Documents average within a partition, queries sum, and empty document partitions are
filled from the nearest occupied partition by Hamming distance. The projection and
SimHash matrices are generated from a fixed seed and their SHA-256 is recorded in the
manifest — a changed matrix silently changes every MUVERA number, so it is pinned
like a model revision. **This is an independent implementation, not a reference one**;
`README.md` says so where MUVERA results are reported.

**These are the paper's stated parameters, not a tuned setting, and they are marked
`calibrated: false`.** That flag is not boilerplate — it records a finding from building
this:

> The FDE **cannot be validated or tuned on synthetic vectors.** On isotropic random
> unit vectors the FDE fails to beat a plain summed-token dot product
> (`bag_of_tokens_similarity`) at *any* parameter setting tried — k_sim 4→8, n_reps 4→20,
> d_proj 16→128, mean and sum aggregation, up to 131,072 dimensions, where the FDE
> reached a separation of z=1.30 against the bag baseline's z=1.29. The reason is
> structural, not a bug: SimHash has no cluster structure to exploit in isotropic data,
> and partition averaging *dilutes* an exactly-matching token by its partition occupancy
> while a plain sum captures it undiluted. FDE's advantage comes from real ColBERT
> embeddings being strongly clustered.

Consequences, all enforced in code rather than left as a caveat:

- `tests/test_muvera.py` asserts only what synthetic data can legitimately establish —
  determinism, dimension, no empty document blocks, and monotone tracking of graded token
  overlap (ρ > 0.9). It does **not** assert recall of the exact `max_sim` top-k, which
  would be a meaningless assertion on synthetic input.
- `tests/test_muvera_calibration.py` is the real gate. On actual ColBERT embeddings it
  measures FDE recall of the exact `max_sim` top-10 at every sweep depth and requires
  both that the FDE beat bag-of-tokens and that recall at the deepest sweep depth reach
  0.8. Below that, `muvera_colbert` would measure the FDE's recall ceiling rather than
  the value of ColBERT rescoring, and its depth curve would be uninterpretable.
- `benchmark calibrate-muvera` runs the same measurement from the CLI and writes
  `results/agg/muvera_calibration.json`. `benchmark doctor` warns while
  `calibrated: false`.

One further honest note for the write-up: at 10,240 float32 dimensions MUVERA's storage
advantage over ColBERT multivectors is modest rather than dramatic. That is a result the
memory table should report, not a number to bury by quietly shrinking the FDE until the
comparison flatters it.

### D2 `[U]` ColBERT storage: float32, `on_disk` above 100K documents

Truncation at 300 document tokens and 32 query tokens (ColBERT convention). `datatype:
float32` and `on_disk: false` for corpora under 100K documents; `on_disk: true` above,
since the spec's own 350 GB estimate for MSMARCO does not fit 32 GB RAM. Both are
per-dataset config, and both are recorded — the storage figure is meaningless without
them.

### D3 `[superseded by K4]` BM25: FastEmbed defaults, stated

Was: `k = 1.2`, `b = 0.75`, `language: english`, `Modifier.IDF` enabled server-side.
Now: BM25 and every sparse vector are deleted entirely (K4) — there is no BM25 config
left to state.

### D7 `[U]` TurboQuant defaults: 4-bit, pinned memory, rescore on

`bits: "4"` — Qdrant's own default, 8× compression, and the one the arm's docstring
and README lead with. `memory: pinned` keeps the quantized codes resident in RAM
rather than letting the OS page them, which is the point of quantizing at all.
`rescore: true` is non-negotiable in practice: without it the arm would report
quantized-only scores, which is a different (and, per TurboQuant's own asymmetric
design, strictly worse) thing than what "quantized_rescore" is named for. All three
are overridable per run (`--quant-bits`; the other two are config-only for now).

### D4 `[U]` Document text is `title + " " + text`

BEIR convention, and it materially moves the Gate 1 number, so it is pinned in config
as `text_template` rather than hardcoded. Empty titles collapse cleanly.

### D5 `[U]` Model revision pinning is honest about its limit

FastEmbed does not expose Hugging Face revision SHAs through its public API. What is
pinned: the `fastembed` version via the lockfile, the model name, and the SHA-256 of
every downloaded model file as resolved in the local cache. That last hash is the real
guarantee — recorded in the manifest as `model_file_digests`. Where FastEmbed's model
source URL is available it is recorded too.

### D6 `[U]` Embedding cache key includes the platform

Per §1: `sha256(model_name | model_digest | platform_tag | text_template | dataset |
corpus_digest)`, where `platform_tag` is `{system}-{machine}-{python}-{onnx_provider}`.
Loading a cache entry whose platform tag differs from the current platform is a hard
error, not a warning.

---

## E. Run protocol

### E1 `[C]` Durability is per-query append, not per-unit checkpoint

§6.5 (queries outer, units inner) and §6.6 (persist after each unit) are incompatible:
under query-outer ordering no unit finishes until the last query, so a crash near the
end loses everything. Resolution: keep the interleaving, and make the raw JSONL
append-per-query with an fsync every `flush_every: 25` records. Resume reads the
completed `(unit, query_id, run_idx)` triples back off disk and skips them. This is
strictly stronger than the spec's checkpoint — a crash costs at most 25 queries, not
one unit.

A resume is refused if the plan digest (arms × depths × dataset × seed × model
digests) differs from the one in the existing run directory, so a config edit cannot
silently produce a half-old, half-new result set.

### E2 `[C]` Two indexing phases per dataset, and the cost is acknowledged

§6.1's memory protocol needs one collection loaded at a time; querying needs several.
So each dataset is indexed twice: once per singleton in isolation for memory, then
once into the combined collection for querying. Embeddings are cached across both, so
the marginal cost is upsert plus index build. `measure_memory: false` opts a dataset
out.

### E3 `[U]` Non-sweeping arms record `depth = null`

Arms 1–5 and 9 do not sweep. Their raw filename uses `depth-none` and the CSV carries
an empty `depth` cell, rather than a fake `0` or a duplicated `10` that would collide
with a real swept value.

---

## F. Datasets

### F1 `[D]` CLERC is excluded from v1

Corpus size, query count, and qrels format are all `TBD` in the spec, and CLERC does
not ship in BEIR layout. Recorded in `manifest.exclusions` with reason
`unscoped: corpus size and qrels format undetermined`. A `config/datasets.yaml` stub
is present so adding it later is config-only.

### F2 `[D]` MSMARCO is configured but disabled by default

8.84M documents through `bge-small` on 8 CPU vCPUs is tens of hours of inference per
representation before a single query runs, and the spec pins the hardware without
budgeting the compute. `enabled: false` with reason recorded; `--include-msmarco`
turns it on deliberately. Its arm eligibility already excludes `colbert` and `muvera`
per §2.

### F3 `[U]` Graded relevance where it exists

Of the enabled datasets only NFCorpus ships graded qrels; SCIDOCS, FiQA, and Quora are
binary. The loader never collapses grades and asserts that the observed grade set
matches `expected_grades` in config, which catches a parsing bug that flattens
NFCorpus's 0/1/2 to binary. §2's absolute phrasing is read as "never collapse grades
where they exist."

### F4 `[U]` Query subsample: 500, then halved

`cap = 500` for every dataset, sampled once with `seed: 20250813` and persisted to
`results/queries/{dataset}.json`. After the 50/50 split (B1) that is ~250 evaluated
queries per dataset, and 323-query NFCorpus yields ~161. Small, and stated plainly
beside every significance claim.

---

## G. Validation gates

### G1 `[U]` Gate 1's reference number is pinned in config, not in code

`config/models.yaml` carries `reference_ndcg@10` for `bge-small-en-v1.5` on NFCorpus
as **0.3164** with its source URL and a tolerance of 0.01.

Two caveats the gate enforces rather than hides:

- The published figure is over all 323 NFCorpus queries. Gate 1 therefore runs on the
  **full** query set, not the 500-cap/50%-split evaluation set, since a subsample
  cannot be compared to a full-set number. This is the one place a metric is computed
  outside the `validation` split, and it is labelled `gate1_full_queryset`.
- **The pinned value must be confirmed against the MTEB leaderboard before publishing.**
  It is recorded as `verified: false` in config, and the gate prints a warning while
  that flag is false. A wrong reference number makes Gate 1 either vacuous or falsely
  failing, and Gate 1 is what licenses trust in everything downstream.

### G2 `[D]` Gate 4 measures what it actually measures

The spec's procedure (index `dense` twice, compare NDCG@10) quantifies **HNSW build
nondeterminism**, not "the cost of separate collections versus one shared collection" —
that conclusion needs the shared-collection variant in the comparison. The gate keeps
the spec's procedure and threshold (gap < 0.005) under its true name,
`test_gate4_hnsw_build_variance`, and adds a second comparison —
`{dataset}__dense` singleton versus the `dense` vector inside `{dataset}__combined` —
which is the number A1 actually needs. Both are reported.

---

## H. Sharding experiment — REMOVED

### H1 `[superseded by K4]` The sharding experiment is out of scope

This section previously specified a corrected A/B design for measuring per-shard
fusion divergence (spec.md §7). The whole experiment was removed as out of scope
(K4), along with `src/sharding.py`, its CLI command, its config block, and the
fusion/formula machinery that existed only to serve it. Kept as a stub so the
section letters stay stable and the removal is visible rather than silent.

---

## I. Cost model

### I1 `[U]` Stated assumption, single instance, serial latency

`config/run.yaml`: AWS `m6i.2xlarge` (8 vCPU / 32 GB), on-demand `us-east-1`, USD
**0.384**/hour, as a pinned and clearly-labelled assumption with a `priced_on` date.
Cost per 1,000 queries is computed from total serial latency (A3), and the client-side
reranker is assumed to run on the *same* instance — so `cross_encoder` costs are
charged CPU time on the same box, not free. Any reader substituting their own price
needs only this one config value.

This pricing assumption is unrelated to whatever machine actually produces the
latency numbers it's multiplied against (J1) — it prices a *hypothetical* deployment
instance, not the dev machine. Keep that distinction in any write-up: the $/1k-queries
figure is only as meaningful as the latency it's derived from, and per J1 that latency
is now this machine's, not the priced instance's.

---

## J. Environment

### J1 `[D]` Run native to whatever machine executes it — no forced `linux/amd64`, no emulation

`spec.md` section 1 is explicit and repeated: *"Qdrant via Docker, pinned tag ...
linux/amd64"*, and section 11's reminders restate it — *"Build for linux/amd64 from
the start, even when developing on ARM."* The original implementation followed this
literally: `docker-compose.yml` pinned `platform: linux/amd64` unconditionally, and the
`docker run` fallback in `qdrant_control.py` did the same.

That is now overridden, by explicit user instruction: **latency measurements must be
true for the machine actually running the benchmark, not an emulated stand-in for a
different one.** The override was not theoretical — running the spec's pinned amd64
image on an Apple Silicon Mac produced empirically severe emulation overhead: a single
ColBERT-multivector upsert batch during a smoke test exceeded the client's 120s gRPC
timeout mid-write, landing only 1,098 of 3,633 points before the call was aborted.
Emulation here isn't a small tax on absolute numbers; it changes which requests even
complete.

**What changed:**

- `docker-compose.yml` no longer sets `platform`. Qdrant ships a multi-arch manifest,
  so Docker resolves it to the host's own architecture automatically — arm64 on
  Apple Silicon, amd64 on an x86 host, with no emulation layer either way.
- `qdrant_control.py`'s `_docker_run()` fallback no longer passes `--platform`, for
  the same reason.
- `manifest.py`'s hardware block drops the fixed `target_environment` /
  `is_target_environment` framing (which treated anything non-Linux-x86 as a
  deviation to warn about). No replacement field was added in its place: an earlier
  draft carried a manifest flag asserting "no emulation," first as a hardcoded
  constant (wrong — it would keep claiming true even under the amd64 override below),
  then as a live `docker exec uname -m` check. Both were removed. The only way
  emulation can occur is `docker-compose.amd64.yml` being applied deliberately, which
  requires typing that filename on the command line — self-evident to whoever does
  it, so a tripwire for it is checking something that can't happen by accident.
- `docker-compose.amd64.yml` is a new, opt-in override
  (`docker compose -f docker-compose.yml -f docker-compose.amd64.yml up -d`) for
  anyone who later wants to reproduce spec.md's originally-intended Linux x86 target
  specifically. It is expected to be slow off Linux x86, per the finding above, and
  is not the default.

**What this means for interpreting results:** there is no longer one fixed target
hardware description for this project. Whatever machine ran a given result is
authoritative for it, and that machine's description lives in `manifest.hardware` for
that run — not a spec-mandated assumption. Absolute latency not being poolable
across different hardware is ordinary benchmarking hygiene, not particular to this
project — the portable claims are the relative ones (which arm wins, the
gain-vs-depth curve shape), and those are expected to hold across machines, same as
any benchmark. What's specific here is narrower: quality (NDCG), not just latency,
can also shift slightly across architectures, because ONNX embeddings differ in
low-order floating-point bits between instruction sets, enough to flip near-tied
rankings (spec.md section 1) — that's a second-order effect most latency-only
benchmarks don't have to account for. The platform-keyed embedding cache
(`platform_info.py`) already refuses a cross-architecture cache hit outright, which
is what actually enforces this rather than a comparability warning in prose.

**What did not change:** §1's other pins (image tag, model revisions, dataset
versions, lockfile) are untouched. `spec.md` itself is never edited — this decision
records the deviation, it does not retroactively rewrite the spec's stated intent.

---

## K. Reranking-on-a-unified-baseline redesign (2026-08-25)

### K1 `[D]` The arm set is redefined around one principle: every arm reranks a unified dense baseline

A user-directed redesign, not a spec-following decision — this section documents a
deliberate departure from `spec.md` §4's original ten-arm list, agreed in chat on
2026-08-25. The reasoning, and what it changes:

**Hybrid-fusion arms are removed entirely: `bm25`, `rrf`, `rrf_weighted`, `dbsf`.**
These combine retrievers' rank lists rather than rescore a fixed candidate set with a
richer relevance model, so they are not reranking strategies under this benchmark's
current scope (this echoes the earlier "what counts as reranking by industry
standard" discussion in this chat — fusion is generally categorised as hybrid search,
not reranking, by the vendors that build both). All associated machinery is removed
from the arm registry: `weighted_rrf`, `_fusion_arm`, the `rrf_k`/`rrf_weight_grid`
config fields, and `runner.py`'s `tune_rrf_weights` (which grid-searched weights on
the train split — see B1, C3 — now dead code, deleted rather than left unreachable).
`ArmContext.tuned_weights` and `ArmSpec.client_side_fusion` are removed for the same
reason: both were always going to be `None`/`False` for every remaining arm, the same
"always-true dead field" problem already caught once this session for a manifest flag
(see J1's history) — caught here before shipping instead of after.

What's **not** touched: `sharding.py` (spec §7) still needs RRF and formula-rescore
machinery to reproduce per-shard fusion divergence, which is a different question
from "is this a reranking arm." `formula_query`, `formula_expression`, and
`rrf_prefetch_formula` stay in `arms.py`; `models.Fusion.RRF`, `_sparse()`, and
`fusion_prefetch_limit` stay wired for that experiment alone.

**Sparse baseline (`bm25`) removed; dense is now the sole baseline.** Falls out
cleanly once no arm needs a sparse vector by default: `config/datasets.yaml`'s
default `representations` list drops `sparse` for every dataset (nfcorpus, scidocs,
fiqa, quora: `[dense, colbert, muvera]`; msmarco: `[dense]`, since its `sparse` entry
existed only to feed the now-removed `bm25` arm). `sharding.py` is unaffected —
it always builds its own dedicated collection with `representations=["dense",
"sparse"]` explicitly, independent of a dataset's default list.

**Every remaining/planned arm reranks the dense baseline, or a prefetch derived from
it:**

| Arm | Status | Shape |
|---|---|---|
| `dense` | default | the baseline itself |
| `colbert_rescore` | default | dense prefetch → ColBERT `max_sim` |
| `muvera_colbert` | default | MUVERA prefetch → exact `max_sim` (m=0, so exact not approximate) |
| `cross_encoder` | registered, off-default | dense prefetch → client-side rerank |
| `quantized_rescore` | **not implemented** | quantized dense prefetch → full-precision rescore |
| `relevance_feedback` | registered, gated (unchanged from C2) | dense seeds → feedback query |

`colbert_rescore` and `muvera_colbert` needed no logic change — they already
prefetched from `dense`/`muvera` respectively before this redesign; only their
framing and surrounding documentation changed.

**The baseline dense model is now a runtime parameter, not just a config edit.**
`benchmark run --dense-model NAME` overrides `config/models.yaml`'s `dense.name` for
that run; `--dense-model-dim N` supplies the vector dimension if it can't be
auto-resolved. Auto-resolution (`resolve_dense_model_dim` in `embedding.py`) queries
FastEmbed's own `TextEmbedding.list_supported_models()` registry, which carries a
`dim` field per model — this is the source of truth used rather than asking the user
to know it, since a wrong dimension would silently corrupt collection creation.
**Consequence:** Gate 1's reference NDCG@10 (G1) is pinned to the *default* model
(`bge-small-en-v1.5`); overriding the baseline invalidates it, and the CLI prints a
note to that effect. Gate 1 is not automatically re-derived for an overridden model —
that would need its own reference number from a source like the MTEB leaderboard,
same caveat as G1 itself.

### Two decisions deliberately deferred, not guessed at

**K2 — `quantized_rescore`'s exact mechanism is undecided.** The stated default,
"TurboQuant, 4-bit," does not map onto Qdrant's native quantization types (binary =
1-bit, scalar = 8-bit; no built-in 4-bit scalar type exists). Building it requires
choosing between:
  - (a) binary quantization as the closest native fit — simplest, uses Qdrant's
    built-in `quantization_config` + `oversampling`/`rescore` search params, which
    map naturally onto this benchmark's existing depth-sweep pattern (oversampling
    factor ≈ prefetch depth), but is 1-bit, not 4-bit;
  - (b) a custom 4-bit codec (e.g. two 4-bit codes packed per byte) stored via
    Qdrant's `datatype: uint8` raw-vector support, giving genuine 4-bit width at the
    cost of building and maintaining the quantization/dequantization code ourselves.

Rather than silently picking one, the arm is declared in `resolve_arms()`'s
exclusion list (same discipline as `formula_rrf`) with this reason, and no query
logic is registered for it. `benchmark plan` and every raw run report it as excluded.
Un-blocking this needs a decision, not code.

**K3 `[superseded by N1 for relevance_feedback]` — `cross_encoder`'s and
`relevance_feedback`'s place in the arm set is unconfirmed.** The redesign's arm list
(spec.md is never edited; this is a chat-only redefinition) is exhaustive and neither
appears in it. Both were already "legitimate reranking" by the industry-standard
analysis this chat did earlier (unlike the removed fusion arms), so neither was
deleted on an inference — `cross_encoder` is registered but removed from the default
`arms.enabled` list; `relevance_feedback` is unchanged from its prior gated-by-default
state (C2). Both remain runnable via `--arms cross_encoder` /
`--include-relevance-feedback`. Confirm or reverse explicitly rather than treating
omission from the 4-arm list as a silent deletion order.


### K4 `[D]` The sharding experiment is removed

`spec.md` §7 specified a separate experiment quantifying how Qdrant's per-shard
fusion changes rankings on a multi-shard collection. Removed as out of scope by
user decision.

Consequences, all of them cascading from the fact that fusion is now used nowhere:

- `src/sharding.py` deleted, along with the `benchmark sharding` CLI command and the
  `sharding:` block in `config/run.yaml`.
- The formula-rescore machinery (`formula_query`, `formula_expression`,
  `rrf_prefetch_formula`) existed **only** to serve this experiment — the
  `formula_rrf` arm itself was already excluded from reported results (C1). With the
  experiment gone it has no consumer, so it is deleted rather than left as unreachable
  code. The `formula_rrf` exclusion entry is removed too: an exclusion note for an arm
  that no longer exists anywhere is noise, not transparency.
- `boost_field` was a payload field written on every point solely for that formula
  rescore. It is no longer written or indexed. `boost_field_for()` is deleted.
- **Sparse vectors are gone entirely.** BM25 was already removed as a baseline arm
  (K1) and fusion arms with it; the sharding experiment was the last remaining
  consumer of a sparse vector. So `SparseModel` config, `SparseVec`, the sparse
  encoder property, `embed_docs_sparse`/`embed_query_sparse`, the sparse cache
  read/write path, and `sparse_vectors_config` on collection creation are all deleted.
  Nothing in the benchmark builds a sparse vector any more.
- `execution.fusion_prefetch_limit` deleted — it only ever sized fusion prefetches.
- `tests/test_gate3_equivalences.py` deleted. Its remaining checks were the formula
  rescore guard (no longer applicable) and a depth-sweep check that
  `test_gate2_arms_distinct.py` already covers structurally.

**Not removed:** the `shard_number: 1` field in the manifest. Everything still runs
single-shard, and recording that explicitly is worth one line — a reader should not
have to infer it from an absence.

### K5 `[U]` Candidate depths are grounded in measured relevant-docs-per-query, not chosen by feel

> Extended by **O12**: the ladder now reaches 1000 on every dataset, for MUVERA's
> crossover rather than for quality headroom, and the per-dataset split below is replaced
> by a per-arm one. The relevance measurements here still set the first five rungs.

Depths were **per dataset**: NFCorpus sweeping `10, 25, 50, 100, 200, 500` and every
other dataset stopping at `200`. Set from measured qrels, not by feel.

The measured relevance density that drove it (test split, positive judgments only):

| Dataset | avg | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 38.2 | 16 | 40 | 71 | 475 |
| SciFact | 1.1 | 1 | 1 | 1 | 5 |
| SCIDOCS | 4.9 | 5 | 5 | 5 | 5 |
| FiQA | 2.6 | 2 | 3 | 5 | 15 |
| Quora | 1.6 | 1 | 1 | 2 | 75 |

**Why only NFCorpus goes to 500.** It is the only dataset whose judgments are dense
enough to still be gaining at that depth (p75 of 40, max 475); it is also the cheapest
place to sweep deep, at 323 queries and the smallest corpus. For the other four, every
relevant document sits within depth 50, so depth 200 is already ~4× past saturation —
enough to demonstrate the plateau, which is the actual deliverable. A 500 point there
would add a second flat measurement for 56% of the compute.

**The cost asymmetry that makes this the right cut:** runtime scales with the *sum* of
the depths, not their count. `10+25+50+100+200+500 = 885`, of which 500 alone is 56%.
Dropping it makes a dataset's sweep 2.3× cheaper. By contrast dropping the `25` point
saves ~6% while removing resolution from the steepest part of the curve — the 10→50
region where datasets with 1–5 relevant docs do all their moving — so `25` is kept.

Superseded detail from the original version of this decision, kept because the qrels
table it rests on is still the evidence base:

| Dataset | rel/query mean | median | p90 | p99 | max | grades |
|---|---|---|---|---|---|---|
| NFCorpus | 38.2 | 16 | 71 | 421 | 475 | 1, 2 |
| SCIDOCS | 4.9 | 5 | 5 | 5 | 5 | 0, 1 |
| FiQA | 2.6 | 2 | 5 | 10 | 15 | 1 |
| Quora | 1.6 | 1 | 2 | 9 | 75 | 1 |

Percentage of queries whose relevant documents all fit within depth D:

| Dataset | 10 | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| NFCorpus | 39.9% | 62.5% | 81.1% | 93.2% | 96.0% | 100% |
| SCIDOCS | 100% | 100% | 100% | 100% | 100% | 100% |
| FiQA | 99.1% | 100% | 100% | 100% | 100% | 100% |
| Quora | 99.2% | 99.9% | 100% | 100% | 100% | 100% |

What this changes:

- **NFCorpus is the only dataset with real quality headroom past depth ~100.** Its
  qrels are an order of magnitude denser than the others'. It is also the Gate 1
  reference dataset and the smoke-test target, so it is doing a lot of work here.
- **For SCIDOCS, FiQA and Quora the deeper points measure latency and cost against a
  quality plateau.** They are kept deliberately: locating where an approach stops
  paying off is the primary deliverable, and you cannot show a plateau without
  sampling past its start.
- Depths are now overridable per dataset (`depths` in `config/datasets.yaml`, null =
  use the global sweep), resolved by `Config.depths_for()`, so a dataset whose shape
  demands a different range does not force a global change.

Two caveats that belong in any write-up rather than in a footnote:

1. **Depth 500 on NFCorpus is ~14% of that 3.6K corpus.** Reranking an eighth of the
   entire collection is not a realistic production ratio; that data point measures
   something closer to exhaustive rescoring than to reranking. The same depth is
   0.87% of FiQA and 0.096% of Quora.
2. **This table is a capacity ceiling, not first-stage recall.** It says how many
   relevant documents *exist*, so it bounds what any depth could possibly retrieve.
   Whether the dense baseline actually places those documents inside the candidate set
   is an empirical question the benchmark measures — it is not assumed here.


### K6 `[U]` TurboQuant's memory cost is measured against an unquantized twin

`quantized_rescore` shares the `dense` vector with the baseline (K2), so the combined
collection has quantization attached to the vector arm 1 also reads. That is correct for
latency — both arms read byte-identical data, so their difference is the quantization and
nothing else — but it left the memory phase unable to answer "what does TurboQuant cost",
because the only `dense` number available already included it.

Resolution: the memory phase measures **two** dense singletons.

| Singleton | Quantization | What it isolates |
|---|---|---|
| `{dataset}__dense` | none | the unquantized dense vector's cost |
| `{dataset}__dense_tq` | TurboQuant, configured bits | the same vectors *plus* quantization |

`dense_tq` is a measurement-only pseudo-representation (`QUANTIZED_DENSE` in
`indexing.py`): it writes to the same `dense` vector name, reuses the same cached
embeddings, is never part of a dataset's `representations`, and is never queried. It is
inserted automatically after `dense` whenever the memory phase runs. Its `datatype` field
records the bit width (e.g. `turboquant-4bit`) so the row is self-describing.

**A measured result worth stating up front, because the intuitive expectation is
backwards:** on disk, TurboQuant makes a collection *larger*, not smaller. Measured on
5,000 × 384-dim vectors, 4-bit: 16,204 KB unquantized vs 17,020 KB quantized, a delta of
~816 KB that matches the predicted size of the 4-bit codes (5000 × 384 × 0.5 B ≈ 960 KB).
The original vectors have to stay — rescoring reads them — so the quantized codes are
strictly additive to storage. The benefit is that the structure actually *searched* is 8×
smaller and can be pinned in RAM (`memory: pinned`), which buys speed and resident-set
size, not disk. Any write-up that reports quantization as a storage saving without this
distinction is reporting it wrong.


---

## L. Cleanup pass (2026-09-03)

### L1 `[D]` The train/validation split is removed — nothing in this benchmark trains

The 50/50 train/validation split existed for `rrf_weighted`, which grid-searched fusion
weights on the train half. That arm was deleted in K1. What remained was a split whose
`train_qrels()` accessor **was never called by anything**, and whose only real consumer
was `warmup_ids = train_ids[:50]`.

So the benchmark was discarding half of every dataset's queries to reserve a pool that
needs 50 of them. That is pure loss of statistical power: the reported metric is a mean
over queries, and its confidence interval shrinks as 1/√n.

Replaced by `_hold_out_warmup()`, which takes a fixed count instead of a fraction:

| Dataset | Evaluated before | Evaluated now |
|---|---:|---:|
| NFCorpus | 161 | 273 |
| SciFact | — | 250 |
| SCIDOCS | 250 | 950 |
| FiQA | 250 | 598 |
| Quora | 250 | 450 |

The warmup pool shrinks automatically on small query sets (capped at a third of the
sample) rather than starving evaluation — a dataset with few queries needs them
reported more than it needs a large warmup.

Renames that follow: `train_ids`/`validation_ids` → `warmup_ids`/`eval_ids`;
`queries.train_fraction` deleted; the `train_val_split` seed becomes `warmup_holdout`;
manifest fields become `n_warmup_queries`/`n_evaluated_queries`.

**What warmup is still for:** keeping evaluated queries out of a cache the warmup pass
warmed (B5). That is a measurement-hygiene concern, not a training one.

### L2 `[D]` No query cap: every dataset runs every judged query

`spec.md` §2 caps queries at 500 per dataset. That cap is removed entirely.

The reasoning it was reconsidered against: **there is nothing methodologically wrong
with running every query — the cap was purely a compute concession.** More queries only
ever tightens confidence intervals and improves the power of the paired significance
test; it introduces no bias. So the cap was trading precision for hours, and precision
was chosen (user decision, 2026-09-03).

`queries.cap` survives as a nullable field, set to `null`. It is not a
benchmark-configuration knob any more — it exists so `scripts/smoke.py` and quick
debugging runs can limit query counts. `_subsample()` returns every judged query
(sorted, so HF row order cannot perturb anything) when the cap is null.

Resulting evaluated-query counts, after the warmup holdout:

| Dataset | Judged | Evaluated |
|---|---:|---:|
| NFCorpus | 323 | 273 |
| SciFact | 300 | 250 |
| SCIDOCS | 1,000 | 950 |
| FiQA | 648 | 598 |
| Quora | 10,000 | 9,950 |
| **Total** | | **12,021** |

**The cost, stated plainly, because it is concentrated and large:** Quora contributes
9,950 of those 12,021 queries, which makes it roughly 90% of the benchmark's
`cross_encoder` time. Measured at 310 cross-encoder pairs/sec on the development
machine, `cross_encoder` goes from ~3h (with Quora capped at 500) to **~12.8h**
uncapped, and total querying from ~4h to ~16h. That was accepted deliberately in
exchange for a ~4.5× tighter confidence interval on the only large-corpus dataset —
which matters most for `quantized_rescore`, whose expected effect is small enough
(near-lossless rescoring) that a few hundred queries could not resolve it from zero.

Capping only the expensive arm is not an option: `ranx.compare`'s paired test requires
every arm to see an identical query set, so giving `cross_encoder` fewer queries than
`dense` would stop the comparison being paired.

### L3 `[D]` Dead code removed

Deleted, each confirmed to have zero call sites:

| Removed | Was for |
|---|---|
| `datasets.train_qrels()` | the train split (L1) |
| `datasets._split()` | the train split (L1) |
| `embedding.embed_query_muvera()` | superseded — the arm calls `encoders.muvera.encode_query` directly |
| `embedding.total_token_vectors()` | never used |
| `indexing._vector_name()`, `_source_representation()` | written for `dense_tq` (K6), then handled inline instead |
| `indexing.REPRESENTATIONS` | never referenced |
| `indexing.MemoryRecord.measured_before_any_query` | a field hardcoded `True` — the same always-true-flag problem as J1's, caught before shipping |
| `metrics.per_query_scores()` | the deleted weight tuning |
| `metrics.summarise_exclusions()` | never used |
| `muvera.bag_of_tokens_similarity()` | duplicated logic `calibrate()` already computes inline |
| `qdrant_control.drop_caches_hint()` | never called |
| `qdrant_control.telemetry()` | never called — and it backed a **documented metric that did not exist** (see B4) |
| `timing.QueryTimers._extras` | declared, never written or read |
| `runner.Unit.key` | the runner builds the key tuple inline |

Also fixed in the same pass: **`benchmark aggregate` was entirely broken** — it
referenced `quant_bits` without declaring it as a parameter, so every invocation raised
`NameError`. Fallout from a bulk edit that pattern-matched across both `run()` and
`aggregate()`. The block made no sense in `aggregate` regardless: bit width only
matters when querying, not when re-aggregating records already on disk.

### L4 `[D]` MSMARCO and CLERC are dropped from the dataset set

Both were configured but disabled. MSMARCO (8.84M docs) was disabled for compute
budget (F2); it is now removed outright — the target machine does not have the RAM, and
at fp32 its embeddings alone would run ~25 GB dense (768-dim) plus ~350 GB ColBERT.
CLERC was never scoped (F1) and had no HF source. Keeping disabled stubs for datasets
nobody intends to run is clutter, so the entries, the `--include-msmarco` flag, and
their tests are gone.

The dataset set is now exactly five, all enabled by default: NFCorpus, SciFact,
SCIDOCS, FiQA-2018, Quora — spanning 144× in corpus size (3.6K→523K) and 1.1→38.2 in
relevance density.

Datasets rejected during this selection, and why, so the reasoning is not relitigated:

| Dataset | Rejected because |
|---|---|
| TREC-COVID | 50 test queries — too few for a stable mean or a meaningful significance test |
| Touché-2020 | 49 test queries — same problem |
| ArguAna | queries are argument passages, not short questions; `colbert.query_max_tokens: 32` would truncate them hard, degrading arms 3 and 4 on that dataset specifically |
| WANDS | not a BEIR dataset — needs a bespoke loader, its own label mapping (Exact/Partial/Irrelevant), and product-text assembly from several fields |

---

## M. External review pass (codex), 2026-09-03

An independent review flagged 16 issues. Each was verified against the actual code
before acting — several were confirmed exactly as described, one was declined with
reasoning, and the verification process itself surfaced two further live bugs codex's
static review did not catch. Full assessment given in chat; this section is the
technical record.

### M1 `[D]` MUVERA now uses fastembed's official `Muvera`, not a custom implementation

The custom FDE in `src/muvera.py` (SimHash + projection + partition-fill, built by
hand) could not be validated on synthetic data at any parameter setting — a real,
standing doubt (D1). `fastembed>=0.7.2` ships `fastembed.postprocess.Muvera`, the
reference implementation the MUVERA/Qdrant ecosystem itself uses. `src/muvera.py` is
now a thin wrapper: `build_muvera()` constructs it, `Encoders.muvera` returns it
directly, arms call `.process_document()`/`.process_query()` instead of the old
`encode_document`/`encode_query`. Config fields (`k_sim`, `n_reps`, `d_proj`) map
directly onto fastembed's own defaults (`k_sim=5, r_reps=20, dim_proj=16`), which also
match the Qdrant MUVERA article's own NFCorpus/ColBERTv2 benchmark config.

**Calibration was re-run for real, not just re-specified.** On the full NFCorpus
corpus (3,633 docs) and 80 queries, at the dataset's actual configured depths
(10–500): the FDE beats a bag-of-tokens baseline at every depth (recall@10: 0.352 vs
0.161; recall@500: 0.852 vs 0.709). `muvera.calibrated` is now `true`, with the
numbers recorded in `calibration_note` and the full artifact at
`results/agg/nfcorpus__muvera_calibration.json`.

`matrices_digest()` was rewritten to hash fastembed's actual matrices
(`simhash_projections[i].simhash_vectors`, `dim_reduction_projections`) rather than an
assumed `random_seed` attribute the class doesn't expose — the first version silently
produced identical digests for different seeds, caught by a test that checked
different seeds give different encodings, before the digest could ship broken.

### M2 `[D]` `muvera_colbert` is gated on a real calibration artifact, not a config flag alone (codex #1)

`arms.muvera_calibration_status(cfg, dataset)` in `resolve_arms()`: the arm is
excluded (not silently run) unless EITHER `models.yaml muvera.calibrated: true`
(a manual "I've verified this" override) OR a `{dataset}__muvera_calibration.json`
artifact exists, shows `beats_bag_of_tokens: true`, and recall at its deepest depth is
≥ 0.80. Verified live on both branches: a dataset with a passing artifact runs the
arm; a dataset with neither runs everything except `muvera_colbert`, with the
exclusion reason naming the exact command to fix it.

### M3 `[D]` Gate 1 blocks `run` by default; dataset revisions are pinned (codex #2, #9)

`--allow-unverified-baseline` is now required to run with the default dense model
while `models.yaml`'s Gate 1 reference is `verified: false` — previously this was
only ever a warning printed by `doctor`, never enforced by `run` or `aggregate`. The
check is skipped when `--dense-model` overrides the baseline (a separate warning
already covers that case; Gate 1's reference does not apply to a different model).

Every dataset's `hf_corpus`/`hf_qrels` now pins the exact HF Hub commit it was loaded
from (`hf_corpus_revision`/`hf_qrels_revision` in `datasets.yaml`), fetched from the
live HF API and spot-checked directly against two of the five before trusting the
batch fetch. Recorded per-run in the manifest. Previously a rerun months later could
silently retrieve a different corpus or qrels from the same repo name with no signal.

### M4 `[D]` Aggregation refuses incomplete data; significance no longer tests the depth it just picked (codex #3, #4)

**Completeness.** `analysis.check_completeness()` compares every `(arm, depth)` unit
present on disk against the expected `eval_queries × repeats` count. A partial unit
(crash, corrupt line, interrupted resume) now raises `IncompleteRunError` naming
exactly which units are short and by how many records, rather than being aggregated
into a normal-looking CSV. `--allow-partial` opts out explicitly, for inspecting an
in-progress run.

**Significance.** The previous design picked each arm's best-scoring depth by NDCG@10
on the eval set, then significance-tested those cherry-picked runs on that SAME eval
set — textbook selection bias, and Bonferroni was applied only across arm pairs, not
across the implicit depth search. There is deliberately no separate tuning split to
select a depth on instead (L1 removed it as pure loss of power for an arm that no
longer exists). Fix: every depth actually swept gets its own comparison; non-sweeping
arms participate in all of them; Bonferroni is applied once across every
`(arm_a, arm_b, depth)` row actually produced, not per depth.

**Found only by running this live, not by codex's static review:** `compare_arms`'s
p-value extraction was completely broken — `report.comparisons` is ranx's
`FrozensetDict`, not a plain `dict`, so `isinstance(table, dict)` was always `False`
and every p-value silently came back `None`, with no error. The significance table
would have shipped empty. Fixed to read `report.comparisons[frozenset({a, b})]`
directly, verified against the installed ranx version's actual return shape, and
covered by a regression test using real ranx (no network needed).

### M5 `[D]` `quantized_rescore`'s memory was attributed to the wrong singleton (codex #7)

`ArmSpec.requires` answers "what must a dataset index for this arm to be eligible" —
correctly `["dense"]` for `quantized_rescore`, since eligibility only needs a dense
vector to exist. The aggregate CSV's memory columns reused the same field to decide
"what does this arm's memory cost," which attributed `quantized_rescore`'s row to
`dense`'s (unquantized) memory instead of `dense_tq`'s. Fixed with a distinct
`ArmSpec.memory_reps` (defaults to `requires`, explicitly overridden to `dense_tq` for
this one arm). Verified live: `quantized_rescore`'s aggregate row now carries
`mem_dense_tq_*` populated and `mem_dense_*` empty; `dense`'s row is the reverse.
Gross AND payload-subtracted (`vector_only_*`) figures are now both exported — only
gross was before.

### M6 `[D]` Resume/storage/lifecycle bugs (codex #10, #11, #12)

- **`--no-resume` no longer silently duplicates.** It previously reused the same
  append-mode JSONL files with an empty `done` set, so every record got rewritten
  alongside the old ones — quality silently kept the last duplicate per query while
  latency averaged over all of them. `run` now refuses `--no-resume` against existing
  raw records unless `--overwrite` is also passed, which deletes them first.
- **A corrupt line anywhere but the true final line now raises**, in both
  `_iter_valid_lines` and `repair_truncated`. Previously any `JSONDecodeError` was
  silently skipped, anywhere in the file — but only the last line of a crash-truncated
  file can legitimately be partial; a bad line earlier is unexplained corruption that
  was being lost with no signal, contradicting the code's own comment.
- **`client.close()` is now in a `finally`.** An arm exception previously skipped it
  entirely, leaking the connection.

### M7 `[D]` Dead surface removed rather than left reachable-looking (codex #13, #14)

- `relevance_feedback` was permanently unreachable — `rf_weights` was never set by
  anything. `--rf-weights-path` now loads real trained weights from a JSON file and is
  required alongside `--include-relevance-feedback`; the arm's own refusal-without-
  weights behavior (spec.md §4, C2) is unchanged for the untrained case.
- The optional cross-encoder "quality ceiling" (Jina reranker v2, CC-BY-NC) was
  configured but had no CLI path to enable it (`use_ceiling_model` was read from
  `ArmContext.extras`, which nothing ever set) — removed entirely: the config block,
  `Encoders.cross_ceiling`, `ArmContext.extras`, and the branch in `cross_encoder_arm`.
  Re-add properly, with a real flag, if the ceiling comparison is wanted later.
- `pytest.ini`'s `pythonpath = ["."]` fixes bare `pytest` failing import collection
  (it needed `-m pytest` or a manual `PYTHONPATH` before). Renaming the `src` package,
  codex's proposed fix, was declined — it would touch every import, the CLI entry
  point, and every doc path for a one-line config fix.
- `tests/conftest.py`'s `nfcorpus` fixture now catches a network/HF-Hub failure
  during the actual corpus load and skips with a clear reason, instead of surfacing
  as a fixture ERROR when offline.

### M8 Declined, with reasoning

- **A full `ExperimentPlan`/`RunContract` type-system layer, and splitting each arm
  into embed→retrieve→rerank→normalize stages.** Codex's own conclusion agreed a full
  rewrite isn't warranted; M4's completeness gate delivers the actual substance of
  the ask (no incomplete data can be aggregated as a finished result) as a targeted
  addition instead of a new architectural layer.
- **Persisting full candidate-ID sets as first-class artifacts.** A real feature, not
  a bug fix — and one that would meaningfully bloat raw output at 12,000+ queries per
  dataset. `_guard_prefetch` already asserts every call respects the requested depth.
- **Repeated, independent memory measurements per representation.** The single-sample
  noise concern is real, but defaulting to 3× the memory-phase runtime was not judged
  worth imposing without being asked. Left as a known limitation instead.

## N. Arm set reduced to five, 2026-09-06

### N1 `[D]` `relevance_feedback` removed entirely

Was: registered but gated off by default (C2), reachable only via
`--include-relevance-feedback` plus `--rf-weights-path`, needing the external
`qdrant-relevance-feedback` package. Never enabled by default and never run in this
benchmark's own validation passes. Decided this arm isn't wanted in the set at all —
not just off by default — so it's deleted rather than left declared-but-unused:
`relevance_feedback_arm` and its `@register` block, `ArmContext.rf_weights`,
`run_dataset`'s `rf_weights` param, the `--include-relevance-feedback` /
`--rf-weights-path` CLI flags, `ArmsConfig.rf_seeds` / `rf_seed_source`, the
`relevance-feedback` optional dependency extra, and every README/test reference.
`ArmsConfig.gated` stays (now an empty dict) as a generic mechanism — it isn't
specific to this one arm and costs nothing to keep. The registered arm set is now
five: `dense`, `quantized_rescore`, `colbert_rescore`, `muvera_colbert`,
`cross_encoder`.

---

## O. BM25 returns as a second first stage, 2026-10-27

### O1 `[D]` `bm25` is a baseline, and the rerankers run on its candidates too

K1 removed BM25 together with the fusion arms. Fusion stays out of scope, but BM25 as a
*first stage that rerankers rescore* is reranking in the strictest sense, and BM25 →
cross-encoder is the canonical BEIR reranking setup. Against `dense` alone (bge-small),
both rerankers lose; against BM25 they are expected to win. Publishing both shows when
reranking pays off rather than only one side of it.

Shape, chosen over a `--baseline dense|sparse` run mode:

- **The first stage is part of the arm's identity**: `bm25`, `bm25_late_interaction_rescore`,
  `bm25_cross_encoder`, alongside the unchanged dense arms. Raw files, resume keys,
  aggregates and the significance table are all keyed by arm name, so nothing
  downstream needed a new dimension, and the two families can't overwrite each other.
- **Both families run in one interleaved pass** against one collection, so the two
  baselines are measured under identical conditions. A run-mode flag would put them in
  separate runs under whatever conditions each one happened to get. One family alone is
  still `--arms bm25,bm25_late_interaction_rescore,bm25_cross_encoder`.
- **Every arm declares its `baseline`**, carried into the aggregate CSV, so a gain is
  always computed against the baseline that produced the arm's candidates.
  `muvera_late_interaction_rescore` generates its own candidates and is measured against `dense`; it has
  no BM25 counterpart.
- *(Mechanism superseded by O3.)* **IDF is applied by Qdrant** (`modifier: idf` on the sparse vector) and
  **`avg_len` is measured per corpus** with FastEmbed's own tokenizer instead of its
  fixed default of 256. An offline exact-scoring check gave NDCG@10 0.3275 on NFCorpus
  (BEIR's published 0.325) and 0.6886 on SciFact (0.665). With 256 it was 0.3234 and
  0.6830. Gate 1 now checks `bm25` against the BEIR figure too.

### O2 `[D]` "ColBERT" renamed to "late interaction" everywhere it names a slot, not a model

The late-interaction model is still being chosen (ColBERTv2, answerai-colbert-small,
LateOn), so arm names, the named vector, the config section, the query-embedding timer
and the memory columns describe the mechanism instead: `late_interaction_rescore`,
`bm25_late_interaction_rescore`, `muvera_late_interaction_rescore`, vector and
representation `late_interaction`, `models.yaml late_interaction:`,
`t_embed_late_interaction_ms`, `mem_late_interaction_*`. The embedding-cache namespace
stays `"colbert"` (and its `.colbert.npz` files): cache keys already carry the model
name, and renaming the namespace would orphan every cached corpus. Recorded runs keep
their original arm names; their aggregates were built at the time and need no re-run.

### O3 `[D]` BM25 moves from sparse vectors to Qdrant 1.20's BM25-scored text index

Qdrant 1.20 (2026-10-22) scores BM25 directly on a full-text payload index and derives
every statistic (document count, IDF, average length) from the indexed data. The
sparse-vector setup from O1 is replaced by it, and Qdrant is pinned to v1.20.0 (client
1.20.x):

- **It is the method Qdrant now presents**, and this is a Qdrant guide.
- **Exact statistics, no client code.** With one shard, IDF and average length are the
  corpus's own. The FastEmbed BM25 encoder, the per-corpus `avg_len` measurement through
  FastEmbed's private helpers, and the sparse vector are deleted.
- **No re-upload:** the index sits on the `text` payload every point already stores for
  the cross-encoder.
- **Text processing matches standard English BM25**: word tokenizer, lowercasing,
  English Snowball stemmer and English stopwords. Qdrant's default is lowercasing only.
  `k` and `b` are sent with every query.
- **`enable_hnsw: false`** on the index, so it adds no payload links to the dense HNSW
  graph and the dense arms search the same graph as without BM25.
- **No query-embedding bucket for BM25.** Tokenization happens in Qdrant, inside
  `t_qdrant_server_ms`.
- **Memory** is read from the text index's own entry and reported as the `bm25`
  representation. Qdrant 1.20 nests each payload index's bytes under `usage`; the parser
  now reads that shape and fails on any other, rather than reading zero.

Every number is re-measured on 1.20, dense included. The re-run for the new
late-interaction model needed that anyway.

### O4 `[D]` FastEmbed 0.9.1, for LateOn and mLateOn on the same runtime as every other model

FastEmbed 0.9.0 added `lightonai/LateOn` and `lightonai/mLateOn`, exported to ONNX from
LightOn's own repositories. Running them through FastEmbed keeps every encoder on ONNX
Runtime, so query-embedding latency compares models, not inference libraries. A PyTorch
(PyLate) encoder would have confounded `t_embed_late_interaction_ms` and every
late-interaction arm's total latency.

- **No change for the existing models.** On 40 NFCorpus documents and 20 queries, 0.7.4
  and 0.9.1 produce bit-identical bge-small, ColBERTv2 and answerai-colbert-small
  vectors, MS MARCO MiniLM scores and MUVERA encodings, with the same MUVERA matrix
  digest. The embedding cache, keyed without the FastEmbed version, therefore stays valid.
- **MUVERA calibration names its model** (`muvera.calibrated_model`, and
  `late_interaction_model` in the artifact). A calibration never carries over to another
  model's tokens.
- **The collection records its models in Qdrant's collection metadata**, and a resume
  compares them: ColBERTv2, LateOn and mLateOn are all 128-dim, so vector sizes alone
  would let a resume mix two models' vectors in one collection.

### O5 `[D]` No benchmark-imposed token caps: every model keeps its own context window

The 300-token document and 32-token query limits on late-interaction models are removed.
Each encoder truncates where its own tokenizer does: bge-small and answerai at 512 tokens,
LateOn at 300, mLateOn at 8,192, and the cross-encoder at its own pair limit.

- **Capping one model to match another measures the cap, not the model.** The dense
  baseline already used its full window, so a late-interaction cap was also asymmetric.
- **Memory follows honestly.** A multivector stores one vector per kept token, so a model
  with a longer window shows that cost in the memory table instead of having it hidden.
- **Cache keys record the truncation** (`truncation: model_context_window`) for both
  late-interaction tokens and MUVERA encodings, so vectors built under a cap are never
  reused. Capped entries already on disk are simply not read.
- **On NFCorpus it does not change the model ranking:** answerai at 512 tokens scores
  within 0.002 of answerai at 300, and LateOn's own window is 300.

### O6 `[D]` LateOn is the late-interaction model

`lightonai/LateOn` replaces `colbert-ir/colbertv2.0`. On NFCorpus (all 273 evaluated
queries, dense first stage, depths 10/25/100, each model at its own context window):

| | answerai-colbert-small-v1 | LateOn | mLateOn |
|---|---|---|---|
| NDCG@10 (dense ~0.338) | 0.348 / 0.362 / 0.366 | 0.355 / 0.372 / 0.381 | 0.351 / 0.368 / 0.377 |
| Query embedding, p50 | 4.5 ms | 11 ms | 9.5 ms |
| Late-interaction memory | 608 MB | 608 MB | 776 MB |

- **LateOn beats answerai** at every depth (paired Wilcoxon p = 0.02 / 0.04 / 0.05) and
  beats dense at every depth (Bonferroni-significant). FastEmbed's ONNX LateOn reproduces
  the PyLate pilot within 0.002.
- **mLateOn ties LateOn** (−0.004, p ≥ 0.19) at 28% more memory and 3.5× the embedding
  time; its multilingual, code and long-context strengths do not apply to English BEIR.
- **Query embedding is slower than answerai's** (149M vs 33M parameters), but Qdrant's
  MaxSim is cheaper: LateOn does no query expansion (~8 query tokens on NFCorpus vs
  answerai's 32), so total latency at depth 100 is about the same.
- **MUVERA does not calibrate on LateOn's tokens** at FastEmbed's defaults (recall@500
  0.68 vs the 0.80 threshold; ColBERTv2 passed at 0.85), so
  `muvera_late_interaction_rescore` stays excluded until that is resolved.

### O7 `[D]` MUVERA is tuned once, on held-out queries, and checked per dataset

LateOn's tokens fail MUVERA calibration at FastEmbed's defaults, so the FDE setting is
chosen by a pre-declared procedure rather than by hand:

- **Held-out queries only.** Tuning and calibration use each dataset's 50 warmup queries,
  which are never evaluated, over the whole corpus. No reported number comes from a query
  the FDE was tuned or checked on.
- **One setting for every dataset.** What decides how well an FDE approximates MaxSim is
  mostly the model's token geometry, which is fixed once the model is; per-dataset tuning
  would flatter MUVERA. The grid (`k_sim` 4/5/6, `d_proj` 16/32, `n_reps` 20/40) runs on
  NFCorpus and SciFact.
- **Selection rule:** the smallest FDE whose recall@deepest-depth reaches 0.80 on every
  tuning dataset. 0.80 is a project convention, not a published standard; recall is the
  diagnostic, and the arm is judged end to end against the dense baseline.
- **Per-dataset check, not tuning:** `calibrate-muvera` on each dataset's held-out queries
  with the chosen setting. The arm's gate accepts an artifact only for the configured
  late-interaction model and MUVERA settings.
- **Cost is reported, not hidden:** every setting's size per document is compared with
  the late-interaction multivectors it prefetches for. A setting larger than them defeats
  MUVERA's purpose and is flagged.

### O8 `[D]` The dense baseline runs BAAI's float32 ONNX, registered with FastEmbed

FastEmbed ships `BAAI/bge-small-en-v1.5` only as `Qdrant/bge-small-en-v1.5-onnx-Q`, whose
MatMuls run in float16. ONNX Runtime has fast float16 kernels only on CPUs with float16
arithmetic (Apple M-series, AVX-512 FP16); on an AVX2-only x86 CPU (the benchmark VM's
AMD EPYC 7B12) it falls back to a single-threaded kernel. Measured on that VM, 64
documents of ~300 tokens: float16 26 s on one core, float32 2.4 s on eight.

- **Same model, same runtime.** BAAI's own `onnx/model.onnx` is loaded through
  FastEmbed's `add_custom_model` (CLS pooling, normalized); its embeddings match the
  float16 build to five decimal places.
- **Every model is now float32**, as LateOn and the cross-encoder already were.
- **Its own identity.** The registered name carries the source file, so cache keys,
  collection metadata and the plan digest never mix float16 and float32 embeddings.

### O9 `[D]` Wall-clock waste removed without touching what is measured

Four measured inefficiencies, found after the move to the VM:

- **Cross-encoder pairs are scored once per query** (`execution.share_pointwise_scores`,
  default on). The depth sweep and both cross-encoder arms kept re-scoring the same
  (query, document) pairs — 885 pair-scorings per query against 500 distinct pairs. A
  pointwise scorer gives the same score regardless of pool, so rankings are identical;
  each pair's measured cost is attributed to every unit that uses it, keeping
  `t_rerank_ms` a measured per-pair figure (throughput is depth-independent: 28.6–29.1
  pairs/s at depths 10–500). ~45% off the cross-encoder arms, which are ~98% of a full
  run. `false` restores fully isolated executions.
- **MUVERA document encoding is parallel** (`ingestion.muvera_workers`, 0 = one per
  CPU). The FDE loop is NumPy-light per document and ran on one core; the matrices are
  deterministic in (config, dim, seed), so the split cannot change any output.
- **Only the representations the requested arms read are built.** A pilot without the
  MUVERA arm no longer pays MUVERA encoding plus a 10,240-dim HNSW build it never
  queries. The official all-arms run is unchanged.
- **Exact MaxSim in calibration is corpus-at-a-time** (blocked matmul +
  `maximum.reduceat`) instead of a per-document Python loop, and Qdrant's optimizer
  threads rise from 4 (Mac sizing) to 8 on the VM. Index building only; queries unaffected.

### O10 `[D]` MUVERA encodes mean-centered tokens

Late-interaction token embeddings are anisotropic: they collapse into a narrow cone, so
nearly every random SimHash hyperplane falls to one side of the whole corpus and the
partitions carry almost no information. LightOn report the effect for LateOn directly
(https://huggingface.co/blog/lightonai/lateon-regularization): with MUVERA at R=20,
k_sim=5, d_proj=16, uncentered scores 2.89 NDCG@10 without rescoring against 32.66
centered, and 15.87 against 50.01 with rescoring at depth 200.

- **The corpus mean is subtracted on both sides** before FDE encoding, and nowhere else:
  the multivectors Qdrant rescores with are stored uncentered, so `late_interaction` and
  every arm reading it are untouched.
- **Computed once per corpus**, streamed over the ingestion chunks in float64 so the
  result does not depend on chunk boundaries, then cached — the query side must centre on
  the identical vector, and recomputing it would re-read every token of the corpus.
- **`center` is part of the MUVERA cache key and the plan digest**, so vectors built
  without centering can never be read back as centered ones.
- It does not explain MUVERA's result on its own: our uncentered NFCorpus numbers reached
  parity with dense at depth 500, not the collapse the blog reports, so the gap between
  centered MUVERA and the dense baseline is measured, not assumed.

### O11 `[D]` MUVERA's candidate count is its own knob, and calibration reports rather than gates

The MUVERA paper's only tuning knob is the beam width W — the number of candidates the
FDE stage retrieves before exact Chamfer reranking — and it is tuned per corpus to hit a
target recall (arXiv 2405.19504). Its Table 1 on MS MARCO needs 60 candidates to exceed
80% recall and 700-800 to exceed 95%, against the 1-nearest neighbour under Chamfer.

- **`depth` is W for this arm.** The depth sweep already varies exactly the quantity the
  paper tunes, so the arm needs no separate oversampling knob; comparing iso-quality and
  iso-latency against the other arms is a reading of the same curve. Giving MUVERA a
  larger pool at the same labelled depth would hide the extra rescoring it pays for.
- **Candidate counts in the hundreds are MUVERA's normal operating range**, so a
  crossover at depth 500 is expected rather than a defect. Reporting depth as a fraction
  of the corpus makes small corpora look pathological and is avoided.
- **This project's 0.80 bar is stricter than the paper's**: recall of the exact max_sim
  top-10, not of the single nearest neighbour.
- **`require_calibration: false`.** One FDE setting is chosen on the tuning datasets and
  applied everywhere, so a dataset whose recall falls short is a result to publish, not a
  gap to leave in the comparison. The pass/fail verdict and its reason are recorded per
  dataset in the manifest, and printed at run time.

### O12 `[D]` One depth ladder for all arms, with a per-arm ceiling

`arm_depth_caps` in `config/run.yaml` names arms that run the dataset's ladder truncated
at a ceiling. The ladder itself is now `10, 25, 50, 100, 200, 500, 1000` on every
dataset; no dataset carries its own override any more.

- **Truncation, not a separate ladder.** A paired significance test at depth *d* needs
  both arms to have run at *d*. Truncating keeps a capped arm's depths a subset of every
  other arm's, so the arms stay comparable everywhere they overlap and the Bonferroni
  factor is just the number of comparisons that actually exist.
- **The ladder reaches 1000 for MUVERA.** Its FDE prefetch buys a cheaper first stage by
  giving up prefetch accuracy, and wins that back through candidate count — the beam
  width W of the MUVERA paper, which `depth` already is for this arm (see O11). The
  break-even against dense sits near depth 100 on SciFact and between 500 and 1000 on
  NFCorpus and SciDocs. Stopping at 200 would end the curve below the crossover and
  report the arm as a loss on two datasets out of three. Server-side rescoring is
  20-50 ms/query at any depth, so the two rungs cost ~15 minutes benchmark-wide.
- **The cross-encoder arms are capped at 50.** A client-side forward pass per pair runs
  ~2.8 pairs/s for gte-reranker-modernbert-base, against 20-50 ms for an entire
  server-side rescore. The uncapped ladder would be ~155 h, ~95% of the benchmark, to
  extend a curve already flat by depth 50. With `share_pointwise_scores` on, depths 10
  and 25 are subsets of the depth-50 pass and cost nothing extra.
- **`--depths` clears the caps.** An explicit override asks for exactly those depths on
  every arm, which is what makes a deep cross-encoder diagnostic possible without editing
  config.
- **Caveat carried into the writeup:** depth 1000 is 28% of NFCorpus and 19% of SciFact.
  Those rungs locate MUVERA's crossover; they are not configurations to recommend.

### O13 `[D]` The cross-encoder is gte-reranker-modernbert-base, registered from its own ONNX export

No reranker FastEmbed ships beats the dense baseline on every dataset here.
`Xenova/ms-marco-MiniLM-L-6-v2` loses ndcg@10 on SciFact at every depth measured
(-0.005 at 10, -0.014 at 25, -0.021 at 50): MS MARCO web passages are far from
scientific claims, and the arm would have been a reranker that makes retrieval worse.

`Alibaba-NLP/gte-reranker-modernbert-base` gains on both datasets it was piloted on
(SciFact +0.032, NFCorpus +0.023) and is registered through
`TextCrossEncoder.add_custom_model`, the same mechanism the float32 dense model uses
(O8). It costs ~12x MiniLM per pair, which is what O12's cap pays for.

- **Piloted on SciFact, not only NFCorpus.** NFCorpus is MiniLM's best case here, so a
  comparison run only there would have selected the weaker model.
- **Throughput is measured on real corpus documents.** Pair throughput depends on
  document length, so a synthetic-document figure overstates it several-fold and every
  cost projection built on it would be wrong.
