"""Collection creation and upsert.

One collection per dataset, named after the dataset, holding every representation as a
named vector. All arms query it, so every arm's prefetch reads the same HNSW graph.
Per-vector memory is read off that same collection by `memory.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from .config import Config, DatasetSpec
from .datasets import LoadedDataset, iter_batches
from .embedding import Encoders, EmbeddingCache
from .qdrant_control import QdrantController


def collection_name(dataset: str) -> str:
    """One collection per dataset, named after it. Kept as a function so call sites read
    as an intent rather than as a bare string."""
    return dataset


@dataclass
class DocumentVectors:
    """Corpus embeddings for one dataset, computed once and reused across passes."""

    dense: np.ndarray | None = None
    late_interaction: list[np.ndarray] | None = None
    muvera: np.ndarray | None = None
    n_token_vectors: int | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def require(self, representation: str):
        value = getattr(self, representation)
        if value is None:
            raise RuntimeError(f"representation {representation!r} was not embedded")
        return value


# ------------------------------------------------------------------- embedding


def embed_corpus(
    cfg: Config,
    bundle: LoadedDataset,
    encoders: Encoders,
    representations: Sequence[str],
    *,
    cache: EmbeddingCache | None = None,
) -> DocumentVectors:
    """Embed the corpus for the requested representations, using the platform cache."""
    cache = cache or EmbeddingCache(cfg.paths["cache"])
    texts = bundle.texts
    digest = bundle.corpus_digest
    name = bundle.spec.name
    out = DocumentVectors()

    needs_late_interaction = "late_interaction" in representations or "muvera" in representations
    needs_dense = "dense" in representations

    if needs_dense:
        key = cache.key(
            representation="dense",
            model_name=cfg.models.dense.model_id,
            dataset=name,
            corpus_digest=digest,
        )
        cached = cache.load_dense(key)
        if cached is None:
            start = time.perf_counter()
            cached = encoders.embed_docs_dense(texts)
            out.timings["dense"] = time.perf_counter() - start
            cache.save_dense(key, cached, {"model": cfg.models.dense.model_id, "dataset": name})
        out.dense = cached

    if needs_late_interaction:
        key = late_interaction_cache_key(cfg, bundle, cache)
        cached_late_interaction = cache.load_late_interaction(key)
        if cached_late_interaction is None:
            start = time.perf_counter()
            cached_late_interaction = encoders.embed_docs_late_interaction(texts)
            out.timings["late_interaction"] = time.perf_counter() - start
            cache.save_late_interaction(
                key, cached_late_interaction,
                {"model": cfg.models.late_interaction.name, "dataset": name},
            )
        out.n_token_vectors = int(sum(m.shape[0] for m in cached_late_interaction))
        if "late_interaction" in representations:
            out.late_interaction = cached_late_interaction

    if "muvera" in representations:
        key = muvera_cache_key(cfg, encoders, cache, name, digest)
        cached_muvera = cache.load_dense(key)
        if cached_muvera is None:
            source = cached_late_interaction if needs_late_interaction else None
            if source is None:
                raise RuntimeError("muvera requires late_interaction token embeddings")
            start = time.perf_counter()
            from .muvera import encode_documents

            cached_muvera = encode_documents(
                cfg.models.muvera, cfg.models.late_interaction.dim, cfg.run.seeds.muvera,
                source, mean=muvera_token_mean(cfg, bundle, encoders, cache),
                workers=cfg.run.ingestion.muvera_workers or None,
            )
            out.timings["muvera"] = time.perf_counter() - start
            cache.save_dense(
                key,
                cached_muvera,
                {"model": "muvera_fde", "dataset": name, "key": key},
            )
        out.muvera = cached_muvera

    return out


# ---------------------------------------------------------------- collections


def _dense_params(
    cfg: Config, dim: int, distance: models.Distance
) -> models.VectorParams:
    """Params for a single-vector (non-multivector) named vector."""
    return models.VectorParams(
        size=dim,
        distance=distance,
        hnsw_config=models.HnswConfigDiff(
            m=cfg.run.qdrant.hnsw.m,
            ef_construct=cfg.run.qdrant.hnsw.ef_construct,
            full_scan_threshold=cfg.run.qdrant.hnsw.full_scan_threshold_kb,
        ),
    )


# Documents are truncated only by each model's own context window. Part of every
# late-interaction and MUVERA cache key, so vectors computed under any other truncation
# are never reused.
TRUNCATION = "model_context_window"


def late_interaction_cache_key(
    cfg: Config, bundle: LoadedDataset, cache: EmbeddingCache | None = None
) -> str:
    """The model name and truncation in the key separate entries; "colbert" is a fixed
    namespace label for the late-interaction cache."""
    cache = cache or EmbeddingCache(cfg.paths["cache"])
    return cache.key(
        representation="colbert",
        model_name=cfg.models.late_interaction.name,
        dataset=bundle.spec.name,
        corpus_digest=bundle.corpus_digest,
        extra={"truncation": TRUNCATION},
    )


def muvera_cache_key(
    cfg: Config,
    encoders: Encoders,
    cache: EmbeddingCache,
    dataset: str,
    corpus_digest: str,
    *,
    bounds: tuple[int, int] | None = None,
) -> str:
    """Cache key for FDE-encoded documents.

    Every input that changes the vectors is in it: the source model, the FDE's own
    random matrices, the truncation, and whether the tokens were mean-centered.
    """
    from .muvera import matrices_digest

    extra: dict = {
        "matrices": matrices_digest(
            encoders.muvera, cfg.models.muvera, cfg.models.late_interaction.dim
        ),
        "truncation": TRUNCATION,
        "center": cfg.models.muvera.center,
    }
    if bounds is not None:
        extra["chunk_start"], extra["chunk_end"] = bounds
    return cache.key(
        representation="muvera",
        model_name=f"fde({cfg.models.late_interaction.name})",
        dataset=dataset,
        corpus_digest=corpus_digest,
        extra=extra,
    )


def _chunk_bounds(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _chunk_late_interaction_cache_key(
    cfg: Config, bundle: LoadedDataset, cache: EmbeddingCache, chunk_start: int, chunk_end: int
) -> str:
    return cache.key(
        representation="colbert",
        model_name=cfg.models.late_interaction.name,
        dataset=bundle.spec.name,
        corpus_digest=bundle.corpus_digest,
        extra={
            "truncation": TRUNCATION,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        },
    )


def token_vector_count(cfg: Config, bundle: LoadedDataset) -> int | None:
    """Total late-interaction token vectors in this corpus, summed from per-chunk cache metadata.

    Needed to read the late-interaction and MUVERA memory figures: "bytes per million
    points" is ambiguous by ~100x on a collection whose points each hold a variable number of token
    vectors. Chunk boundaries are deterministic from corpus size and `chunk_size`, so this
    recomputes each chunk's cache key and sums its metadata rather than reloading any
    ragged array, which is gigabytes for the larger corpora. Returns None if any chunk is
    not yet cached.
    """
    cache = EmbeddingCache(cfg.paths["cache"])
    total = 0
    for chunk_start, chunk_end in _chunk_bounds(len(bundle.documents), cfg.run.ingestion.chunk_size):
        key = _chunk_late_interaction_cache_key(cfg, bundle, cache, chunk_start, chunk_end)
        meta = cache.meta(key)
        if meta is None or meta.get("n_token_vectors") is None:
            return None
        total += int(meta["n_token_vectors"])
    return total


def _late_interaction_params(cfg: Config, spec: DatasetSpec) -> models.VectorParams:
    return models.VectorParams(
        size=cfg.models.late_interaction.dim,
        distance=_distance(cfg.models.late_interaction.distance),
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        # m=0 disables the HNSW graph. Required, not optional: max_sim is not symmetric,
        # so multivectors are for rescoring and never for prefetch.
        hnsw_config=models.HnswConfigDiff(m=cfg.models.late_interaction.hnsw_m),
        on_disk=spec.late_interaction_on_disk,
    )


def _distance(name: str) -> models.Distance:
    return {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }[name]


def vectors_config(
    cfg: Config, spec: DatasetSpec, representations: Sequence[str]
) -> dict:
    """Named-vector config.

    Every vector is stored and read at full precision, so all arms sharing one collection
    read byte-identical data and their differences are their reranking alone.
    """
    dense_cfg: dict[str, models.VectorParams] = {}

    if "dense" in representations:
        dense_cfg["dense"] = _dense_params(
            cfg, cfg.models.dense.dim, _distance(cfg.models.dense.distance)
        )
    if "muvera" in representations:
        dense_cfg["muvera"] = _dense_params(
            cfg, cfg.models.muvera.dim, _distance(cfg.models.muvera.distance)
        )
    if "late_interaction" in representations:
        dense_cfg["late_interaction"] = _late_interaction_params(cfg, spec)
    return dense_cfg


def bm25_index_params(cfg: Config) -> models.TextIndexParams:
    """Full-text index with BM25 scoring over the document-text payload field.

    `enable_hnsw=False` keeps the index out of the dense HNSW graph: by default Qdrant
    adds graph links per indexed payload value for filtered search, which would change
    the graph every dense arm searches. BM25 scoring doesn't use them.
    """
    bm25 = cfg.models.bm25
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType(bm25.tokenizer),
        lowercase=bm25.lowercase,
        stopwords=models.Language(bm25.stopwords),
        stemmer=models.SnowballParams(
            type=models.Snowball.SNOWBALL, language=models.SnowballLanguage(bm25.stemmer)
        ),
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        enable_hnsw=False,
        on_disk=False,
    )


def collection_metadata(cfg: Config, representations: Sequence[str]) -> dict[str, Any]:
    """The models that produced a collection's vectors, stored on the collection.

    Vector names, sizes and distances cannot tell two models of the same dimension
    apart, so a resume checks these too.
    """
    meta: dict[str, Any] = {}
    if "dense" in representations:
        meta["dense_model"] = cfg.models.dense.model_id
    if "late_interaction" in representations or "muvera" in representations:
        meta["late_interaction_model"] = cfg.models.late_interaction.name
    if "muvera" in representations:
        muvera = cfg.models.muvera
        meta["muvera"] = {
            "k_sim": muvera.k_sim,
            "n_reps": muvera.n_reps,
            "d_proj": muvera.d_proj,
            "seed": cfg.run.seeds.muvera,
        }
    return meta


def create_collection(
    client: QdrantClient,
    cfg: Config,
    spec: DatasetSpec,
    name: str,
    representations: Sequence[str],
    *,
    shard_number: int = 1,
    recreate: bool = True,
) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    dense_cfg = vectors_config(cfg, spec, representations)
    client.create_collection(
        collection_name=name,
        vectors_config=dense_cfg or None,
        shard_number=shard_number,
        metadata=collection_metadata(cfg, representations),
    )
    # Created before the upload, so documents are indexed as they arrive and the
    # optimizer's final rebuild leaves BM25's corpus statistics complete.
    if "bm25" in representations:
        client.create_payload_index(
            collection_name=name,
            field_name=cfg.models.bm25.field,
            field_schema=bm25_index_params(cfg),
            wait=True,
        )


def estimate_point_floats(
    cfg: Config, vectors: DocumentVectors, representations: Sequence[str], n_docs: int
) -> int:
    """Approximate float count per point, used to size upsert batches."""
    total = 0
    if "dense" in representations:
        total += cfg.models.dense.dim
    if "muvera" in representations:
        total += cfg.models.muvera.dim
    if "late_interaction" in representations and vectors.n_token_vectors and n_docs:
        total += (vectors.n_token_vectors / n_docs) * cfg.models.late_interaction.dim
    return max(int(total), 1)


def safe_batch_size(
    cfg: Config, vectors: DocumentVectors, representations: Sequence[str], n_docs: int
) -> int:
    """Cap the upsert batch so a request cannot exceed Qdrant's payload limit.

    A 256-point batch of 10,240-dimensional MUVERA vectors serialises to ~48 MB as JSON,
    over Qdrant's 32 MB ceiling, so a fixed batch size silently breaks as soon as a
    wide representation is added. Sizing from the actual vector width keeps the
    configured value as a ceiling rather than a landmine.
    """
    floats = estimate_point_floats(cfg, vectors, representations, n_docs)
    # JSON renders a float as ~12-14 characters; protobuf sends 4 bytes.
    bytes_per_float = 4 if cfg.run.qdrant.prefer_grpc else 14
    budget_bytes = 16 * 1024 * 1024  # half of Qdrant's 32 MiB limit
    by_size = budget_bytes // max(floats * bytes_per_float, 1)
    return max(1, min(cfg.run.qdrant.upsert_batch_size, int(by_size)))


class IngestionError(RuntimeError):
    """The collection on disk is not a faithful copy of the corpus."""


def upsert_documents(
    client: QdrantClient,
    cfg: Config,
    bundle: LoadedDataset,
    name: str,
    representations: Sequence[str],
    vectors: DocumentVectors,
    *,
    skip_existing: bool = False,
) -> float:
    """Upsert the corpus, build the index, and verify the result. Elapsed seconds.

    Index-build ordering matters. Indexing is disabled for the duration of the upload and
    the threshold restored afterwards; that restore is what triggers a single clean HNSW
    build. Setting the threshold to 0 *after* uploading instead does the opposite: it
    leaves whichever segments happened to cross the default threshold mid-upload indexed
    and permanently freezes the rest as brute-force, so measured latency is a blend of
    ANN and full-scan segments in a ratio that shifts with upload timing.

    `skip_existing` resumes an interrupted upload by re-sending only the points that are
    missing. Safe because point ids are deterministic (corpus enumeration order), so a
    re-sent point overwrites itself rather than duplicating.
    """
    start = time.perf_counter()
    total = len(bundle.documents)

    _disable_indexing(client, name)

    indices = list(range(total))
    if skip_existing:
        indices = _missing_point_indices(client, name, indices)
        if not indices:
            log = f"  all {total} points already present"
        else:
            log = f"  resuming upload: {len(indices)} of {total} points missing"
        print(log)

    batch_size = safe_batch_size(cfg, vectors, representations, total)
    batches = list(iter_batches(indices, batch_size))
    for position, batch in enumerate(batches):
        points = [_point(bundle, vectors, representations, i) for i in batch]
        # The final batch waits, so the count check below cannot race an in-flight write.
        client.upsert(
            collection_name=name, points=points, wait=position == len(batches) - 1
        )

    _finish_indexing(client, cfg, name, expected_points=total)
    return time.perf_counter() - start


def _disable_indexing(client: QdrantClient, name: str) -> None:
    # No indexing while loading: indexing partial segments is wasted work, and it is
    # what leaves a collection half-indexed by accident of upload timing.
    client.update_collection(
        collection_name=name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def _finish_indexing(client: QdrantClient, cfg: Config, name: str, *, expected_points: int) -> None:
    # Restoring the threshold is what starts the index build; then wait for it.
    client.update_collection(
        collection_name=name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=cfg.run.qdrant.indexing_threshold_kb
        ),
    )
    wait_for_green(client, name)
    verify_ingestion(client, name, expected_points=expected_points)


def _missing_point_indices(
    client: QdrantClient, name: str, ids: Sequence[int]
) -> list[int]:
    """Which of `ids` are absent from the collection, checked in bounded-size batches."""
    present: set[int] = set()
    for batch in iter_batches(list(ids), 1000):
        found = client.retrieve(
            collection_name=name, ids=list(batch), with_payload=False, with_vectors=False
        )
        present.update(int(point.id) for point in found)
    return [i for i in ids if i not in present]


def verify_ingestion(
    client: QdrantClient, name: str, *, expected_points: int, require_indexed: bool = True
) -> None:
    """Hard check that the collection really holds the whole corpus, indexed.

    Silent under-ingestion is the failure mode that matters: it inflates latency (a
    smaller collection is faster) and deflates recall (missing documents can never be
    retrieved), while looking exactly like a valid run. An upload that times out or loses
    its connection partway leaves exactly that, and nothing downstream can detect it
    after the fact, so it has to fail here.
    """
    exact = client.count(collection_name=name, exact=True).count
    if exact != expected_points:
        raise IngestionError(
            f"{name}: {exact} points present, expected {expected_points} "
            f"({expected_points - exact} missing). The collection is not a faithful copy "
            "of the corpus: latency would be measured on a smaller index and every "
            "missing document is unretrievable, so recall would be understated. Rerun "
            "the build; it resumes from what is already there."
        )

    info = client.get_collection(name)
    if info.status != models.CollectionStatus.GREEN:
        raise IngestionError(f"{name}: status {info.status}, expected green after indexing")

    if not require_indexed:
        return

    # Green status is not evidence that an index exists: a collection frozen at
    # indexing_threshold=0, or one whose segments never reached the threshold, reports
    # green with nothing indexed. Only the count of indexed vectors settles it.
    threshold = getattr(info.config.optimizer_config, "indexing_threshold", None)
    if threshold == 0:
        raise IngestionError(
            f"{name}: indexing_threshold is 0, so the HNSW index was never built and "
            "every search would be a brute-force scan. Latency numbers would be "
            "meaningless."
        )
    wants_graph = [
        vector_name
        for vector_name, params in (info.config.params.vectors or {}).items()
        if (params.hnsw_config.m if params.hnsw_config else None) != 0
    ]
    if wants_graph and not info.indexed_vectors_count:
        raise IngestionError(
            f"{name}: {info.indexed_vectors_count} vectors indexed although "
            f"{', '.join(wants_graph)} ask for an HNSW graph, so every search is a "
            f"brute-force scan. The collection's segments did not reach "
            f"indexing_threshold ({threshold} KB); lower qdrant.indexing_threshold_kb."
        )


def _point(
    bundle: LoadedDataset,
    vectors: DocumentVectors,
    representations: Sequence[str],
    i: int,
) -> models.PointStruct:
    return _point_from_chunk(bundle, vectors, representations, i, chunk_start=0)


def _point_from_chunk(
    bundle: LoadedDataset,
    chunk_vectors: DocumentVectors,
    representations: Sequence[str],
    global_i: int,
    chunk_start: int,
) -> models.PointStruct:
    """Build the point for document `global_i` from one chunk's embeddings.

    Two different indices are in play and must not be conflated: `bundle.documents`
    and `bundle.id_map` are indexed globally over the whole corpus, while
    `chunk_vectors` holds only this chunk's embeddings, indexed from 0.
    """
    doc = bundle.documents[global_i]
    local_i = global_i - chunk_start
    vector: dict[str, object] = {}
    if "dense" in representations:
        vector["dense"] = chunk_vectors.require("dense")[local_i].tolist()
    if "muvera" in representations:
        vector["muvera"] = chunk_vectors.require("muvera")[local_i].tolist()
    if "late_interaction" in representations:
        vector["late_interaction"] = chunk_vectors.require("late_interaction")[local_i].tolist()
    return models.PointStruct(
        id=bundle.id_map.point(doc.doc_id),
        vector=vector,
        payload={"doc_id": doc.doc_id, "text": doc.text},
    )


def _embed_chunk(
    cfg: Config,
    bundle: LoadedDataset,
    encoders: Encoders,
    representations: Sequence[str],
    cache: EmbeddingCache,
    *,
    chunk_start: int,
    chunk_end: int,
    muvera_mean: np.ndarray | None = None,
) -> DocumentVectors:
    """Embed one slice of the corpus, reusing the on-disk cache per representation.

    Each chunk is cached under its own key (dataset, model, corpus digest, platform,
    and this chunk's bounds), so a chunk already on disk is loaded rather than
    re-embedded — the same reuse `embed_corpus` gives the whole corpus, scoped to a
    slice so peak memory stays bounded by chunk size rather than corpus size.
    """
    texts = bundle.texts[chunk_start:chunk_end]
    digest = bundle.corpus_digest
    name = bundle.spec.name
    out = DocumentVectors()

    needs_late_interaction = "late_interaction" in representations or "muvera" in representations
    needs_dense = "dense" in representations
    cached_late_interaction: list[np.ndarray] | None = None

    if needs_dense:
        key = cache.key(
            representation="dense",
            model_name=cfg.models.dense.model_id,
            dataset=name,
            corpus_digest=digest,
            extra={"chunk_start": chunk_start, "chunk_end": chunk_end},
        )
        cached = cache.load_dense(key)
        if cached is None:
            cached = encoders.embed_docs_dense(texts)
            cache.save_dense(key, cached, {"model": cfg.models.dense.model_id, "dataset": name})
        out.dense = cached

    if needs_late_interaction:
        key = _chunk_late_interaction_cache_key(cfg, bundle, cache, chunk_start, chunk_end)
        cached_late_interaction = cache.load_late_interaction(key)
        if cached_late_interaction is None:
            cached_late_interaction = encoders.embed_docs_late_interaction(texts)
            cache.save_late_interaction(
                key, cached_late_interaction,
                {"model": cfg.models.late_interaction.name, "dataset": name},
            )
        out.n_token_vectors = int(sum(m.shape[0] for m in cached_late_interaction))
        if "late_interaction" in representations:
            out.late_interaction = cached_late_interaction

    if "muvera" in representations:
        key = muvera_cache_key(
            cfg, encoders, cache, name, digest, bounds=(chunk_start, chunk_end)
        )
        cached_muvera = cache.load_dense(key)
        if cached_muvera is None:
            if cached_late_interaction is None:
                raise RuntimeError("muvera requires late_interaction token embeddings")
            from .muvera import encode_documents

            cached_muvera = encode_documents(
                cfg.models.muvera, cfg.models.late_interaction.dim, cfg.run.seeds.muvera,
                cached_late_interaction, mean=muvera_mean,
                workers=cfg.run.ingestion.muvera_workers or None,
            )
            cache.save_dense(
                key,
                cached_muvera,
                {"model": "muvera_fde", "dataset": name, "key": key},
            )
        out.muvera = cached_muvera

    return out


def muvera_token_mean(
    cfg: Config, bundle: LoadedDataset, encoders: Encoders, cache: EmbeddingCache | None = None
) -> np.ndarray | None:
    """The corpus mean token vector MUVERA centers on, or None when centering is off.

    Streamed over the same chunks ingestion uses, so a corpus that does not fit in
    memory still yields the exact mean, and cached: the query side needs the identical
    vector, and recomputing it would mean re-reading every token of the corpus.
    """
    if not cfg.models.muvera.center:
        return None
    from .muvera import TokenMean

    cache = cache or EmbeddingCache(cfg.paths["cache"])
    key = cache.key(
        representation="muvera_mean",
        model_name=cfg.models.late_interaction.name,
        dataset=bundle.spec.name,
        corpus_digest=bundle.corpus_digest,
        extra={"truncation": TRUNCATION},
    )
    cached = cache.load_dense(key)
    if cached is not None:
        return cached[0]

    mean = TokenMean(cfg.models.late_interaction.dim)
    for chunk_start, chunk_end in _chunk_bounds(
        len(bundle.documents), cfg.run.ingestion.chunk_size
    ):
        chunk = _embed_chunk(
            cfg, bundle, encoders, ["late_interaction"], cache,
            chunk_start=chunk_start, chunk_end=chunk_end,
        )
        mean.update(chunk.require("late_interaction"))
    result = mean.result()
    cache.save_dense(
        key,
        result[None, :],
        {"model": cfg.models.late_interaction.name, "dataset": bundle.spec.name,
         "kind": "muvera_token_mean"},
    )
    return result


def late_interaction_corpus(
    cfg: Config, bundle: LoadedDataset, encoders: Encoders
) -> list[np.ndarray]:
    """Every document's late-interaction tokens, in corpus order, from the chunk cache
    `run` writes; chunks not cached yet are embedded and cached."""
    cache = EmbeddingCache(cfg.paths["cache"])
    tokens: list[np.ndarray] = []
    for chunk_start, chunk_end in _chunk_bounds(len(bundle.documents), cfg.run.ingestion.chunk_size):
        chunk = _embed_chunk(
            cfg, bundle, encoders, ["late_interaction"], cache,
            chunk_start=chunk_start, chunk_end=chunk_end,
        )
        tokens.extend(chunk.require("late_interaction"))
    return tokens


def ingest_corpus(
    client: QdrantClient,
    cfg: Config,
    bundle: LoadedDataset,
    name: str,
    representations: Sequence[str],
    encoders: Encoders,
    *,
    cache: EmbeddingCache | None = None,
    resume: bool = False,
) -> float:
    """Embed and upload the corpus one chunk at a time, indexing once at the end.

    Peak memory is one chunk's embeddings, independent of corpus size: MUVERA's
    10,240-dim vectors alone would need ~21 GB resident for Quora's 522,931 documents
    if the whole corpus were embedded before the first upload.

    Embedding and upload are two independent resume states, since a crash can land
    between them: a chunk can be fully cached but not yet uploaded. Embedding resume
    comes from the on-disk cache (`_embed_chunk` loads whatever is already there);
    upload resume comes from `resume=True`, which checks Qdrant itself for which of a
    chunk's point ids are already present rather than trusting a separate marker.
    Every upsert in this function waits for the write to land before the next chunk's
    presence check runs, so that check is never racing an in-flight write.
    """
    cache = cache or EmbeddingCache(cfg.paths["cache"])
    total = len(bundle.documents)
    chunk_size = cfg.run.ingestion.chunk_size
    # Once, before any chunk is encoded: centering needs the whole corpus's mean, so it
    # cannot be derived chunk by chunk.
    muvera_mean = (
        muvera_token_mean(cfg, bundle, encoders, cache) if "muvera" in representations else None
    )

    start = time.perf_counter()
    _disable_indexing(client, name)

    for chunk_start, chunk_end in _chunk_bounds(total, chunk_size):
        chunk_ids = list(range(chunk_start, chunk_end))
        missing = _missing_point_indices(client, name, chunk_ids) if resume else chunk_ids
        if not missing:
            continue

        chunk_vectors = _embed_chunk(
            cfg, bundle, encoders, representations, cache,
            chunk_start=chunk_start, chunk_end=chunk_end, muvera_mean=muvera_mean,
        )
        batch_size = safe_batch_size(cfg, chunk_vectors, representations, chunk_end - chunk_start)
        for batch in iter_batches(missing, batch_size):
            points = [
                _point_from_chunk(bundle, chunk_vectors, representations, i, chunk_start)
                for i in batch
            ]
            client.upsert(collection_name=name, points=points, wait=True)

    _finish_indexing(client, cfg, name, expected_points=total)
    return time.perf_counter() - start


def wait_for_green(client: QdrantClient, name: str, timeout_s: float = 3600.0) -> None:
    """Block until indexing settles. Memory read before green is a partial index."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            return
        if info.status == models.CollectionStatus.RED:
            raise RuntimeError(f"collection {name} entered RED status during indexing")
        time.sleep(2.0)
    raise TimeoutError(f"collection {name} did not reach GREEN within {timeout_s}s")


def build_collection(
    cfg: Config,
    bundle: LoadedDataset,
    encoders: Encoders,
    controller: QdrantController,
    *,
    representations: Sequence[str] | None = None,
    shard_number: int = 1,
    name: str | None = None,
    restart: bool = True,
    resume: bool = True,
) -> str:
    """Build the collection every arm queries. Returns the collection name.

    With `resume=True` (the default) an existing collection with the right vector config
    is kept and only its missing points are re-sent, so an upload interrupted by a crash,
    a lost connection, or a dead battery costs only the part that had not landed. The
    collection is dropped and rebuilt when its config does not match what this run asks
    for, since mixing two vector configs in one collection is not resumable.

    `restart=True` wipes the whole Qdrant storage directory first, so latency is measured
    on an index built from nothing rather than on whatever a previous run left behind. It
    defeats resume by design — the two are exclusive.
    """
    reps = list(representations or bundle.spec.representations)
    if restart:
        controller.restart_clean()
    client = build_client(cfg)
    collection = name or collection_name(bundle.spec.name)

    reusable = (
        resume
        and not restart
        and client.collection_exists(collection)
        and _config_matches(client, cfg, bundle.spec, collection, reps)
    )
    if resume and not reusable and client.collection_exists(collection):
        print(f"  {collection}: existing collection does not match this config; rebuilding")

    if not reusable:
        create_collection(
            client, cfg, bundle.spec, collection, reps, shard_number=shard_number
        )
    ingest_corpus(client, cfg, bundle, collection, reps, encoders, resume=reusable)
    client.close()
    return collection


def _config_matches(
    client: QdrantClient,
    cfg: Config,
    spec: DatasetSpec,
    name: str,
    representations: Sequence[str],
) -> bool:
    """Is an existing collection's config the one this run would create?

    The vector names, sizes, and distances are compared — those are what make stored
    points reusable — along with the models recorded in the collection's metadata and,
    when BM25 is wanted, its full-text index and that index's text processing. A
    mismatch means the collection cannot be resumed into.
    """
    try:
        info = client.get_collection(name)
    except Exception:
        return False
    existing = info.config.params.vectors or {}
    wanted = vectors_config(cfg, spec, representations)
    if set(existing) != set(wanted):
        return False
    for vector_name, want in wanted.items():
        have = existing[vector_name]
        if have.size != want.size or have.distance != want.distance:
            return False
    if (info.config.metadata or {}) != collection_metadata(cfg, representations):
        return False
    if "bm25" not in representations:
        return True
    index = (info.payload_schema or {}).get(cfg.models.bm25.field)
    if index is None or index.params is None:
        return False
    wanted_index = bm25_index_params(cfg).model_dump(exclude_none=True)
    have_index = index.params.model_dump(exclude_none=True)
    return all(have_index.get(key) == value for key, value in wanted_index.items())


def build_client(cfg: Config, *, for_queries: bool = False) -> QdrantClient:
    """Build a client. Query clients always use gRPC.

    Queries read Qdrant's self-reported processing time off the gRPC response, so that
    transport is a requirement of the measurement rather than a preference. Indexing
    follows the configured transport, where gRPC batches far more points per request.
    """
    prefer_grpc = True if for_queries else cfg.run.qdrant.prefer_grpc
    return QdrantClient(
        url=cfg.run.qdrant.url,
        prefer_grpc=prefer_grpc,
        grpc_port=cfg.run.qdrant.grpc_port,
        timeout=cfg.run.qdrant.timeout_s,
    )
