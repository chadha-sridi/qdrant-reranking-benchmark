"""Three independent timers per query.

The split between client-side and server-side time is the central finding of the
project, so the three components are accumulated separately and never summed before
storage. `t_qdrant_ms` is the sum of all Qdrant round trips an arm makes, issued
serially; `n_qdrant_calls` records how many, so a one-call arm is
distinguishable from a two-call arm in the raw data.

`t_qdrant_server_ms` is Qdrant's OWN reported processing time, summed over the same
calls. It matters because 40-60% of the client-observed round trip is transport and
response deserialisation, not search: at limit=500 a measured 9.6 ms round trip was 5.8 ms
of server time and 3.9 ms of client-side overhead. Attributing all of it to Qdrant would
overstate server cost and understate client cost, which is the one comparison this whole
project exists to make.
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
    # Qdrant's self-reported server-side time. None when the transport does not expose
    # it (the gRPC path in qdrant-client discards it).
    t_qdrant_server_ms: float | None = None
    n_qdrant_calls: int = 0

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
            self.n_qdrant_calls += 1

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
            "n_qdrant_calls": self.n_qdrant_calls,
        }
