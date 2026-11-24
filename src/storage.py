"""Raw per-query record storage and resume.

Store raw per-query measurements, never aggregates.

Durability is append-per-query with a periodic fsync rather than a checkpoint at the end
of each unit. Under the interleaved execution ordering (queries outer, units inner) no unit
completes until the final query, so a per-unit checkpoint would give no crash protection
at all; this is strictly stronger — a crash costs at most `flush_every` queries.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from .timing import QueryTimers


def unit_slug(dataset: str, arm: str, depth: int | None) -> str:
    """Raw filename stem. Non-sweeping arms get `depth-none`, never a fake 0."""
    depth_part = "none" if depth is None else str(depth)
    return f"{dataset}__{arm}__depth-{depth_part}"


@dataclass(frozen=True)
class RawRecord:
    query_id: str
    arm: str
    dataset: str
    depth: int | None
    run_idx: int
    retrieved: list[tuple[str, float]]
    timers: QueryTimers
    extra: dict | None = None

    def to_json(self) -> str:
        payload = {
            "query_id": self.query_id,
            "arm": self.arm,
            "dataset": self.dataset,
            "depth": self.depth,
            "run_idx": self.run_idx,
            "retrieved": [[doc_id, score] for doc_id, score in self.retrieved],
            **self.timers.as_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if self.extra:
            payload.update(self.extra)
        return json.dumps(payload, separators=(",", ":"))


class RawWriter:
    """One JSONL file per (dataset, arm, depth), appended as results arrive."""

    def __init__(self, root: Path, *, flush_every: int = 25):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._handles: dict[str, object] = {}
        self._pending: dict[str, int] = {}

    def _handle(self, slug: str):
        if slug not in self._handles:
            self._handles[slug] = (self.root / f"{slug}.jsonl").open("a", encoding="utf-8")
            self._pending[slug] = 0
        return self._handles[slug]

    def write(self, record: RawRecord) -> None:
        slug = unit_slug(record.dataset, record.arm, record.depth)
        handle = self._handle(slug)
        handle.write(record.to_json() + "\n")
        self._pending[slug] += 1
        if self._pending[slug] >= self.flush_every:
            self._sync(slug)

    def _sync(self, slug: str) -> None:
        handle = self._handles[slug]
        handle.flush()
        os.fsync(handle.fileno())
        self._pending[slug] = 0

    def close(self) -> None:
        for slug in list(self._handles):
            try:
                self._sync(slug)
            finally:
                self._handles[slug].close()
        self._handles.clear()
        self._pending.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def completed_keys(root: Path, dataset: str) -> set[tuple[str, int | None, str, int]]:
    """Read back which (arm, depth, query_id, run_idx) tuples already exist on disk.

    A truncated final line from a hard crash is discarded rather than crashing the
    resume, and the work it represented is simply redone.
    """
    done: set[tuple[str, int | None, str, int]] = set()
    if not root.exists():
        return done
    for path in sorted(root.glob(f"{dataset}__*.jsonl")):
        for line in _iter_valid_lines(path):
            done.add(
                (line["arm"], line.get("depth"), line["query_id"], int(line.get("run_idx", 0)))
            )
    return done


class CorruptRawFileError(RuntimeError):
    pass


def _iter_valid_lines(path: Path) -> Iterator[dict]:
    """Yield each JSON record. Only the FINAL line may be silently dropped.

    A crash mid-write can only ever truncate the last line being written — every
    earlier line was already fsynced whole. So a corrupt line anywhere but the last is
    not a crash artifact; it is unexplained data corruption, and skipping it the way a
    truncated final line is skipped would lose the record with no signal at all. That
    case raises instead.
    """
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, raw in enumerate(lines):
        try:
            yield json.loads(raw)
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                continue  # a crash mid-write can only truncate the final line
            raise CorruptRawFileError(
                f"{path}: line {i + 1}/{len(lines)} is invalid JSON and is not the final "
                "line, so it cannot be a truncated-write artifact. Investigate before "
                "resuming or aggregating — this file may be silently missing data."
            ) from exc


def read_raw(root: Path, dataset: str, *, arms: Iterable[str] | None = None) -> list[dict]:
    wanted = set(arms) if arms else None
    out: list[dict] = []
    for path in sorted(root.glob(f"{dataset}__*.jsonl")):
        for record in _iter_valid_lines(path):
            if wanted is None or record["arm"] in wanted:
                out.append(record)
    return out


def repair_truncated(root: Path, dataset: str) -> list[Path]:
    """Drop a truncated FINAL line. Raises if corruption is found anywhere else.

    Mirrors `_iter_valid_lines`'s rule: only the last line of a file
    can legitimately be a crash artifact. A bad line earlier in the file is left alone
    and raised on, rather than silently rewritten away.
    """
    touched: list[Path] = []
    for path in sorted(root.glob(f"{dataset}__*.jsonl")):
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue
        for i, line in enumerate(lines[:-1]):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorruptRawFileError(
                    f"{path}: line {i + 1}/{len(lines)} is invalid JSON and is not the "
                    "final line. Refusing to auto-repair; investigate before resuming."
                ) from exc
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            touched.append(path)
    return touched
