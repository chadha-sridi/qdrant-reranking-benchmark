"""Per-query timers.

The split between client-side and server-side time is the central finding of the
project, so each component is accumulated separately and never summed before storage.

Query embedding is bucketed per encoder (`dense`, `late_interaction`, `muvera`) so the
encoders can be compared directly rather than only inferred from arm-vs-arm
differences; `t_embed_ms` is their sum. BM25 has no bucket: Qdrant tokenizes the query
server-side, so its cost is inside `t_qdrant_server_ms`. `t_qdrant_ms` is the sum of
all Qdrant round trips an arm makes, issued serially.

`t_qdrant_server_ms` is Qdrant's OWN reported processing time, summed over the same
calls, read from the gRPC response. It matters because a large share of the
client-observed round trip is transport and response deserialisation, not search.
Attributing all of it to Qdrant would overstate server cost and understate client cost,
which is the one comparison this whole project exists to make.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# One bucket per query encoder. `muvera` is the FDE transform applied to the
# late-interaction token vectors, not a model call of its own, so an arm that uses it
# always pays `late_interaction` as well.
EMBED_STAGES = ("dense", "late_interaction", "muvera")


@dataclass
class QueryTimers:
    t_embed_dense_ms: float = 0.0
    t_embed_late_interaction_ms: float = 0.0
    t_embed_muvera_ms: float = 0.0
    t_qdrant_ms: float = 0.0
    t_rerank_ms: float = 0.0
    # Qdrant's self-reported server-side time, accumulated across this query's calls.
    t_qdrant_server_ms: float | None = None

    @contextmanager
    def _accumulate(self, attr: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            setattr(self, attr, getattr(self, attr) + (time.perf_counter() - start) * 1000.0)

    def embed(self, stage: str):
        if stage not in EMBED_STAGES:
            raise ValueError(f"unknown embed stage {stage!r}; expected one of {EMBED_STAGES}")
        return self._accumulate(f"t_embed_{stage}_ms")

    def qdrant(self):
        return self._accumulate("t_qdrant_ms")

    def rerank(self):
        return self._accumulate("t_rerank_ms")

    def add_rerank_time(self, seconds: float) -> None:
        """Accumulate rerank cost measured outside the context manager (shared pointwise
        scoring attributes each pair's measured cost to every unit that uses the pair)."""
        self.t_rerank_ms += seconds * 1000.0

    def add_server_time(self, seconds: float | None) -> None:
        """Accumulate Qdrant's reported time for one call, in milliseconds."""
        if seconds is None:
            return
        self.t_qdrant_server_ms = (self.t_qdrant_server_ms or 0.0) + seconds * 1000.0

    @property
    def t_embed_ms(self) -> float:
        return sum(getattr(self, f"t_embed_{stage}_ms") for stage in EMBED_STAGES)

    @property
    def t_total_ms(self) -> float:
        return self.t_embed_ms + self.t_qdrant_ms + self.t_rerank_ms

    @property
    def t_qdrant_overhead_ms(self) -> float | None:
        """Round trip minus server time: transport plus response deserialisation."""
        if self.t_qdrant_server_ms is None:
            return None
        return self.t_qdrant_ms - self.t_qdrant_server_ms

    def as_dict(self) -> dict:
        def opt(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "t_embed_ms": round(self.t_embed_ms, 4),
            "t_embed_dense_ms": round(self.t_embed_dense_ms, 4),
            "t_embed_late_interaction_ms": round(self.t_embed_late_interaction_ms, 4),
            "t_embed_muvera_ms": round(self.t_embed_muvera_ms, 4),
            "t_qdrant_ms": round(self.t_qdrant_ms, 4),
            "t_rerank_ms": round(self.t_rerank_ms, 4),
            "t_total_ms": round(self.t_total_ms, 4),
            "t_qdrant_server_ms": opt(self.t_qdrant_server_ms),
            "t_qdrant_overhead_ms": opt(self.t_qdrant_overhead_ms),
        }
