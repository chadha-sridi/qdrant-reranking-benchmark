"""Three independent timers per query.

The split between client-side and server-side time is the central finding of the
project, so the three components are accumulated separately and never summed before
storage. `t_qdrant_ms` is the sum of all Qdrant round trips an arm makes, issued
serially.

`t_qdrant_server_ms` is Qdrant's OWN reported processing time, summed over the same
calls, read from the gRPC response. It matters because a large share of the
client-observed round trip is transport and response deserialisation, not search.
Attributing all of it to Qdrant would overstate server cost and understate client cost,
which is the one comparison this whole project exists to make.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class QueryTimers:
    t_embed_ms: float = 0.0
    t_qdrant_ms: float = 0.0
    t_rerank_ms: float = 0.0
    # Qdrant's self-reported server-side time, accumulated across this query's calls.
    t_qdrant_server_ms: float | None = None

    @contextmanager
    def embed(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.t_embed_ms += (time.perf_counter() - start) * 1000.0

    @contextmanager
    def qdrant(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.t_qdrant_ms += (time.perf_counter() - start) * 1000.0

    @contextmanager
    def rerank(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.t_rerank_ms += (time.perf_counter() - start) * 1000.0

    def add_server_time(self, seconds: float | None) -> None:
        """Accumulate Qdrant's reported time for one call, in milliseconds."""
        if seconds is None:
            return
        self.t_qdrant_server_ms = (self.t_qdrant_server_ms or 0.0) + seconds * 1000.0

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
        return {
            "t_embed_ms": round(self.t_embed_ms, 4),
            "t_qdrant_ms": round(self.t_qdrant_ms, 4),
            "t_rerank_ms": round(self.t_rerank_ms, 4),
            "t_total_ms": round(self.t_total_ms, 4),
            "t_qdrant_server_ms": (
                None if self.t_qdrant_server_ms is None else round(self.t_qdrant_server_ms, 4)
            ),
            "t_qdrant_overhead_ms": (
                None if self.t_qdrant_overhead_ms is None else round(self.t_qdrant_overhead_ms, 4)
            ),
        }
