"""Per-representation memory, read from Qdrant's own accounting.

Qdrant 1.18+ reports memory for a collection at `GET /collections/{name}/memory`,
broken down per named vector into `storage` / `index`, with `payload`, each payload
index (by field name) and the id tracker reported separately. BM25 is a full-text
payload index rather than a vector, so its cost is read from that index's entry.

Caveat to carry into any writeup: these are Qdrant's own estimates. Its docs put
`ram_bytes` 10-15% low, because allocations inside third-party libraries are not
tracked, and `cached_bytes` is only populated on Unix hosts.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import Config


class MemoryReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Component:
    """One accounted-for piece of a collection, in Qdrant's four dimensions."""

    disk_bytes: int = 0
    ram_bytes: int = 0
    cached_bytes: int = 0
    expected_cache_bytes: int = 0

    @property
    def working_set_bytes(self) -> int:
        """RAM needed to serve this component without going to disk.

        `ram_bytes` is memory Qdrant holds that cannot be evicted; `expected_cache_bytes`
        is what it wants in page cache. Neither alone answers "how much RAM does this
        need" — memory-mapped vectors report ~0 `ram_bytes` while still needing to be
        resident to be fast — so the two are summed.
        """
        return self.ram_bytes + self.expected_cache_bytes

    @classmethod
    def parse(cls, raw: dict | None) -> "Component":
        raw = raw or {}
        return cls(
            disk_bytes=int(raw.get("disk_bytes") or 0),
            ram_bytes=int(raw.get("ram_bytes") or 0),
            cached_bytes=int(raw.get("cached_bytes") or 0),
            expected_cache_bytes=int(raw.get("expected_cache_bytes") or 0),
        )

    def as_dict(self) -> dict:
        return {
            "disk_bytes": self.disk_bytes,
            "ram_bytes": self.ram_bytes,
            "cached_bytes": self.cached_bytes,
            "expected_cache_bytes": self.expected_cache_bytes,
            "working_set_bytes": self.working_set_bytes,
        }


@dataclass(frozen=True)
class VectorMemory:
    """One named vector, as Qdrant reports it: the raw vectors and the HNSW graph."""

    name: str
    storage: Component
    index: Component

    @property
    def total(self) -> Component:
        parts = (self.storage, self.index)
        return Component(
            disk_bytes=sum(p.disk_bytes for p in parts),
            ram_bytes=sum(p.ram_bytes for p in parts),
            cached_bytes=sum(p.cached_bytes for p in parts),
            expected_cache_bytes=sum(p.expected_cache_bytes for p in parts),
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "storage": self.storage.as_dict(),
            "index": self.index.as_dict(),
            "total": self.total.as_dict(),
        }


@dataclass(frozen=True)
class CollectionMemory:
    """Everything Qdrant will tell us about one collection's footprint."""

    dataset: str
    collection: str
    n_documents: int
    n_token_vectors: int | None
    vectors: dict[str, VectorMemory] = field(default_factory=dict)
    payload: Component = Component()
    payload_indexes: dict[str, Component] = field(default_factory=dict)
    id_tracker: Component = Component()
    total: Component = Component()
    qdrant_version: str | None = None

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "collection": self.collection,
            "n_documents": self.n_documents,
            "n_token_vectors": self.n_token_vectors,
            "qdrant_version": self.qdrant_version,
            "vectors": {name: vec.as_dict() for name, vec in self.vectors.items()},
            "payload": self.payload.as_dict(),
            "payload_indexes": {name: c.as_dict() for name, c in self.payload_indexes.items()},
            "id_tracker": self.id_tracker.as_dict(),
            "total": self.total.as_dict(),
            "source": "qdrant /collections/{name}/memory",
            "caveat": (
                "Qdrant's own estimates: ram_bytes runs 10-15% low (untracked "
                "third-party allocations); cached_bytes is Unix-only."
            ),
        }

    def per_million_docs(self, vector: str) -> float | None:
        """Disk bytes per million documents, for comparing across corpus sizes.

        Reported per document rather than per token vector, because "per million points"
        is ambiguous by ~100x on a multivector collection.
        """
        vec = self.vectors.get(vector)
        if vec is None or not self.n_documents:
            return None
        return vec.total.disk_bytes / self.n_documents * 1_000_000


def _fetch(cfg: Config, collection: str, path: str) -> dict:
    url = f"{cfg.run.qdrant.url.rstrip('/')}/collections/{collection}{path}"
    try:
        with urllib.request.urlopen(url, timeout=cfg.run.qdrant.timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # network, 404 on an older server, malformed body
        raise MemoryReadError(
            f"could not read {url}: {exc}. The per-collection memory endpoint needs "
            "Qdrant 1.18 or newer; check the pinned image tag."
        ) from exc
    result = payload.get("result")
    if result is None:
        raise MemoryReadError(f"{url} returned no result field: {payload}")
    return result


def read_collection_memory(
    cfg: Config,
    collection: str,
    *,
    dataset: str,
    n_documents: int,
    n_token_vectors: int | None = None,
) -> CollectionMemory:
    """Read one collection's memory breakdown straight from Qdrant."""
    result = _fetch(cfg, collection, "/memory")

    vectors: dict[str, VectorMemory] = {}
    for entry in result.get("vectors") or []:
        name = entry.get("name")
        if not name:
            continue
        vectors[name] = VectorMemory(
            name=name,
            storage=Component.parse(entry.get("storage")),
            index=Component.parse(entry.get("index")),
        )

    other = result.get("other") or {}
    payload_indexes: dict[str, Component] = {}
    for entry in result.get("payload_index") or []:
        # Each entry is {"name": field, "usage": {...}}. A missing `usage` means the
        # response format changed; reading it as zero would understate memory silently.
        if "name" not in entry or "usage" not in entry:
            raise MemoryReadError(f"unexpected payload_index entry from {collection}: {entry}")
        payload_indexes[entry["name"]] = Component.parse(entry["usage"])

    return CollectionMemory(
        dataset=dataset,
        collection=collection,
        n_documents=n_documents,
        n_token_vectors=n_token_vectors,
        vectors=vectors,
        payload=Component.parse(result.get("payload")),
        payload_indexes=payload_indexes,
        id_tracker=Component.parse(other.get("id_tracker")),
        total=Component.parse(result.get("total")),
        qdrant_version=cfg.run.qdrant.image.rpartition(":")[2] or None,
    )


def representation_memory(cfg: Config, memory: CollectionMemory) -> dict[str, dict]:
    """Memory per representation, keyed by the name arms declare in `requires`.

    Named vectors map one to one. `bm25` maps to the full-text index on its payload
    field: the index is BM25's whole cost, since the text payload it indexes is stored
    for the cross-encoder arms regardless.
    """
    out = {name: vec.as_dict() for name, vec in memory.vectors.items()}
    bm25_index = memory.payload_indexes.get(cfg.models.bm25.field)
    if bm25_index is not None:
        out["bm25"] = {"name": "bm25", "index": bm25_index.as_dict(), "total": bm25_index.as_dict()}
    return out


def verify_expected_representations(
    cfg: Config, memory: CollectionMemory, expected: Sequence[str]
) -> None:
    """Every representation the dataset indexes must appear in the accounting.

    A missing one means either the upload never wrote it or the name drifted, both of
    which would silently leave that representation out of the memory table.
    """
    reported = representation_memory(cfg, memory)
    missing = [name for name in expected if name not in reported]
    if missing:
        raise MemoryReadError(
            f"{memory.collection}: Qdrant reports no memory for {missing}, but the "
            f"dataset indexes {list(expected)}. Reported: {sorted(reported)}."
        )


# Share of the Qdrant host's memory a collection's working set may occupy. The rest is
# for Qdrant's own process, the VM's OS, and page-cache churn during queries.
MAX_WORKING_SET_FRACTION = 0.8


def check_fits_in_memory(memory: CollectionMemory, server_memory_bytes: int) -> None:
    """Fail when the collection cannot stay resident on the server.

    A working set larger than the server's memory is read from disk on every query, so
    the arms that touch the large vectors (late interaction, MUVERA) would be measured on
    disk rather than on Qdrant, while `dense` and the cross-encoder's dense search, whose
    vectors are small, would not. That is a bias between arms, not just a slower run.
    """
    budget = int(server_memory_bytes * MAX_WORKING_SET_FRACTION)
    needed = memory.total.working_set_bytes
    if needed > budget:
        gib = 1024**3
        raise MemoryReadError(
            f"{memory.collection}: working set {needed / gib:.1f} GiB exceeds "
            f"{MAX_WORKING_SET_FRACTION:.0%} of Qdrant's {server_memory_bytes / gib:.1f} GiB, "
            "so the large-vector arms would be disk-bound. Give the Docker VM more memory "
            "(e.g. `colima stop && colima start --cpu 8 --memory 10`), or pass "
            "--allow-disk-bound to measure it anyway."
        )


def write_memory_record(cfg: Config, path: Path, memory: CollectionMemory) -> None:
    record = {**memory.as_dict(), "representations": representation_memory(cfg, memory)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
