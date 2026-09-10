"""Collection creation and upsert.

One collection per dataset, named after the dataset, holding every representation as a
named vector. All arms query it, so every arm's prefetch reads the same HNSW graph.
Per-vector memory is read off that same collection by `memory.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

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
    colbert: list[np.ndarray] | None = None
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

    needs_colbert = "colbert" in representations or "muvera" in representations
    needs_dense = "dense" in representations

    if needs_dense:
        key = cache.key(
            representation="dense",
            model_name=cfg.models.dense.name,
            dataset=name,
            corpus_digest=digest,
        )
        cached = cache.load_dense(key)
        if cached is None:
            start = time.perf_counter()
            cached = encoders.embed_docs_dense(texts)
            out.timings["dense"] = time.perf_counter() - start
            cache.save_dense(key, cached, {"model": cfg.models.dense.name, "dataset": name})
        out.dense = cached

    if needs_colbert:
        key = colbert_cache_key(cfg, bundle, cache)
        cached_colbert = cache.load_colbert(key)
        if cached_colbert is None:
            start = time.perf_counter()
            cached_colbert = encoders.embed_docs_colbert(texts)
            out.timings["colbert"] = time.perf_counter() - start
            cache.save_colbert(key, cached_colbert, {"model": cfg.models.colbert.name, "dataset": name})
        out.n_token_vectors = int(sum(m.shape[0] for m in cached_colbert))
        if "colbert" in representations:
            out.colbert = cached_colbert

    if "muvera" in representations:
        muvera = encoders.muvera
        from .muvera import matrices_digest as _muvera_digest

        digest_str = _muvera_digest(muvera, cfg.models.muvera, cfg.models.colbert.dim)
        key = cache.key(
            representation="muvera",
            model_name=f"fde({cfg.models.colbert.name})",
            dataset=name,
            corpus_digest=digest,
            extra={"matrices": digest_str},
        )
        cached_muvera = cache.load_dense(key)
        if cached_muvera is None:
            source = cached_colbert if needs_colbert else None
            if source is None:
                raise RuntimeError("muvera requires colbert token embeddings")
            start = time.perf_counter()
            cached_muvera = np.stack([muvera.process_document(d) for d in source])
            out.timings["muvera"] = time.perf_counter() - start
            cache.save_dense(
                key,
                cached_muvera,
                {"model": "muvera_fde", "dataset": name, "matrices": digest_str},
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


def colbert_cache_key(
    cfg: Config, bundle: LoadedDataset, cache: EmbeddingCache | None = None
) -> str:
    cache = cache or EmbeddingCache(cfg.paths["cache"])
    return cache.key(
        representation="colbert",
        model_name=cfg.models.colbert.name,
        dataset=bundle.spec.name,
        corpus_digest=bundle.corpus_digest,
        extra={"doc_max_tokens": cfg.models.colbert.doc_max_tokens},
    )


def _chunk_bounds(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _chunk_colbert_cache_key(
    cfg: Config, bundle: LoadedDataset, cache: EmbeddingCache, chunk_start: int, chunk_end: int
) -> str:
    return cache.key(
        representation="colbert",
        model_name=cfg.models.colbert.name,
        dataset=bundle.spec.name,
        corpus_digest=bundle.corpus_digest,
        extra={
            "doc_max_tokens": cfg.models.colbert.doc_max_tokens,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        },
    )


def token_vector_count(cfg: Config, bundle: LoadedDataset) -> int | None:
    """Total ColBERT token vectors in this corpus, summed from the per-chunk cache metadata.

    Needed to read the ColBERT and MUVERA memory figures: "bytes per million points" is
    ambiguous by ~100x on a collection whose points each hold a variable number of token
    vectors. Chunk boundaries are deterministic from corpus size and `chunk_size`, so this
    recomputes each chunk's cache key and sums its metadata rather than reloading any
    ragged array, which is gigabytes for the larger corpora. Returns None if any chunk is
    not yet cached.
    """
    cache = EmbeddingCache(cfg.paths["cache"])
    total = 0
    for chunk_start, chunk_end in _chunk_bounds(len(bundle.documents), cfg.run.ingestion.chunk_size):
        key = _chunk_colbert_cache_key(cfg, bundle, cache, chunk_start, chunk_end)
        meta = cache.meta(key)
        if meta is None or meta.get("n_token_vectors") is None:
            return None
        total += int(meta["n_token_vectors"])
    return total


def _colbert_params(cfg: Config, spec: DatasetSpec) -> models.VectorParams:
    return models.VectorParams(
        size=cfg.models.colbert.dim,
        distance=_distance(cfg.models.colbert.distance),
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        # m=0 disables the HNSW graph. Required, not optional: max_sim is not symmetric,
        # so multivectors are for rescoring and never for prefetch.
        hnsw_config=models.HnswConfigDiff(m=cfg.models.colbert.hnsw_m),
        on_disk=spec.colbert_on_disk,
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
    if "colbert" in representations:
        dense_cfg["colbert"] = _colbert_params(cfg, spec)
    return dense_cfg


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
    if "colbert" in representations and vectors.n_token_vectors and n_docs:
        total += (vectors.n_token_vectors / n_docs) * cfg.models.colbert.dim
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

    # A collection frozen at indexing_threshold=0 reports green with nothing indexed, so
    # a green status alone is not evidence that an index exists.
    threshold = getattr(info.config.optimizer_config, "indexing_threshold", None)
    if require_indexed and threshold == 0:
        raise IngestionError(
            f"{name}: indexing_threshold is 0, so the HNSW index was never built and "
            "every search would be a brute-force scan. Latency numbers would be "
            "meaningless."
        )


def _point(
    bundle: LoadedDataset,
    vectors: DocumentVectors,
    representations: Sequence[str],
    i: int,
) -> models.PointStruct:
    doc = bundle.documents[i]
    vector: dict[str, object] = {}
    if "dense" in representations:
        vector["dense"] = vectors.require("dense")[i].tolist()
    if "muvera" in representations:
        vector["muvera"] = vectors.require("muvera")[i].tolist()
    if "colbert" in representations:
        vector["colbert"] = vectors.require("colbert")[i].tolist()
    return models.PointStruct(
        id=bundle.id_map.point(doc.doc_id),
        vector=vector,
        payload={"doc_id": doc.doc_id, "text": doc.text},
    )


def _point_from_chunk(
    bundle: LoadedDataset,
    chunk_vectors: DocumentVectors,
    representations: Sequence[str],
    global_i: int,
    chunk_start: int,
) -> models.PointStruct:
    """`_point`'s counterpart for chunked embeddings.

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
    if "colbert" in representations:
        vector["colbert"] = chunk_vectors.require("colbert")[local_i].tolist()
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

    needs_colbert = "colbert" in representations or "muvera" in representations
    needs_dense = "dense" in representations
    cached_colbert: list[np.ndarray] | None = None

    if needs_dense:
        key = cache.key(
            representation="dense",
            model_name=cfg.models.dense.name,
            dataset=name,
            corpus_digest=digest,
            extra={"chunk_start": chunk_start, "chunk_end": chunk_end},
        )
        cached = cache.load_dense(key)
        if cached is None:
            cached = encoders.embed_docs_dense(texts)
            cache.save_dense(key, cached, {"model": cfg.models.dense.name, "dataset": name})
        out.dense = cached

    if needs_colbert:
        key = _chunk_colbert_cache_key(cfg, bundle, cache, chunk_start, chunk_end)
        cached_colbert = cache.load_colbert(key)
        if cached_colbert is None:
            cached_colbert = encoders.embed_docs_colbert(texts)
            cache.save_colbert(key, cached_colbert, {"model": cfg.models.colbert.name, "dataset": name})
        out.n_token_vectors = int(sum(m.shape[0] for m in cached_colbert))
        if "colbert" in representations:
            out.colbert = cached_colbert

    if "muvera" in representations:
        muvera = encoders.muvera
        from .muvera import matrices_digest as _muvera_digest

        digest_str = _muvera_digest(muvera, cfg.models.muvera, cfg.models.colbert.dim)
        key = cache.key(
            representation="muvera",
            model_name=f"fde({cfg.models.colbert.name})",
            dataset=name,
            corpus_digest=digest,
            extra={
                "matrices": digest_str,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            },
        )
        cached_muvera = cache.load_dense(key)
        if cached_muvera is None:
            if cached_colbert is None:
                raise RuntimeError("muvera requires colbert token embeddings")
            cached_muvera = np.stack([muvera.process_document(d) for d in cached_colbert])
            cache.save_dense(
                key,
                cached_muvera,
                {"model": "muvera_fde", "dataset": name, "matrices": digest_str},
            )
        out.muvera = cached_muvera

    return out


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

    start = time.perf_counter()
    _disable_indexing(client, name)

    for chunk_start, chunk_end in _chunk_bounds(total, chunk_size):
        chunk_ids = list(range(chunk_start, chunk_end))
        missing = _missing_point_indices(client, name, chunk_ids) if resume else chunk_ids
        if not missing:
            continue

        chunk_vectors = _embed_chunk(
            cfg, bundle, encoders, representations, cache,
            chunk_start=chunk_start, chunk_end=chunk_end,
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
    """Is an existing collection's vector config the one this run would create?

    Only the vector names, sizes, and distances are compared — those are what make
    stored points reusable. A mismatch means the points on disk cannot be resumed into.
    """
    try:
        existing = (client.get_collection(name).config.params.vectors) or {}
    except Exception:
        return False
    wanted = vectors_config(cfg, spec, representations)
    if set(existing) != set(wanted):
        return False
    for vector_name, want in wanted.items():
        have = existing[vector_name]
        if have.size != want.size or have.distance != want.distance:
            return False
    return True


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
