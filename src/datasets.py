"""BEIR loading, query subsampling, warmup holdout, and ID mapping.

ID mismatches are the most common cause of silently wrong metrics,
so the string<->int map is bidirectional, persisted, and verified on load.

There is no train/test split — nothing in this benchmark trains. The
only holdout is a small warmup pool, kept out of evaluation so no reported query is
served from a cache the warmup pass warmed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config, DatasetSpec


class IdMapError(RuntimeError):
    pass


class QrelsError(RuntimeError):
    pass


# --------------------------------------------------------------------- id mapping


@dataclass
class IdMap:
    """Bidirectional BEIR doc_id <-> Qdrant point id map."""

    dataset: str
    to_point: dict[str, int]
    to_doc: dict[int, str]

    @classmethod
    def build(cls, dataset: str, doc_ids: list[str]) -> IdMap:
        if len(set(doc_ids)) != len(doc_ids):
            dupes = _first_duplicates(doc_ids)
            raise IdMapError(f"{dataset}: duplicate doc_ids in corpus, e.g. {dupes}")
        # Point ids are the corpus enumeration order, which is stable because the
        # corpus is loaded in dataset order and never shuffled.
        to_point = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        return cls(dataset=dataset, to_point=to_point, to_doc={v: k for k, v in to_point.items()})

    def point(self, doc_id: str) -> int:
        try:
            return self.to_point[doc_id]
        except KeyError:
            raise IdMapError(f"{self.dataset}: doc_id {doc_id!r} not in corpus") from None

    def doc(self, point_id: int) -> str:
        try:
            return self.to_doc[int(point_id)]
        except KeyError:
            raise IdMapError(f"{self.dataset}: point id {point_id!r} not in map") from None

    def __len__(self) -> int:
        return len(self.to_point)

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for point_id in range(len(self.to_doc)):
            h.update(self.to_doc[point_id].encode())
            h.update(b"\x00")
        return h.hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "digest": self.digest,
            "n_docs": len(self),
            # Stored as an ordered list: point id is the index, so the file is half the
            # size of a dict and cannot disagree with itself.
            "doc_ids": [self.to_doc[i] for i in range(len(self.to_doc))],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> IdMap:
        payload = json.loads(path.read_text())
        id_map = cls.build(payload["dataset"], payload["doc_ids"])
        if id_map.digest != payload["digest"]:
            raise IdMapError(
                f"{path}: id map digest mismatch — the persisted map does not match its "
                "own contents. Delete it and re-index; every metric depending on it is "
                "suspect."
            )
        return id_map


def _first_duplicates(items: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            out.append(item)
            if len(out) >= limit:
                break
        seen.add(item)
    return out


# ------------------------------------------------------------------------- corpus


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    documents: list[Document]
    queries: dict[str, str]  # subsampled: query_id -> text
    qrels: dict[str, dict[str, int]]  # graded, never collapsed
    # Warmup queries are held out from evaluation so no reported query is ever served
    # from a cache warmed by the warmup pass. No arm trains on them —
    # nothing in this benchmark trains at all.
    warmup_ids: list[str]
    eval_ids: list[str]
    id_map: IdMap
    all_queries: dict[str, str]  # full query set, for Gate 1

    @property
    def corpus_digest(self) -> str:
        h = hashlib.sha256()
        for doc in self.documents:
            h.update(doc.doc_id.encode())
            h.update(b"\x00")
            h.update(doc.text.encode())
            h.update(b"\x01")
        return h.hexdigest()

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.documents]

    def eval_queries(self) -> dict[str, str]:
        """The reported query set. Everything published comes from here."""
        return {qid: self.queries[qid] for qid in self.eval_ids}

    def eval_qrels(self) -> dict[str, dict[str, int]]:
        return {qid: self.qrels[qid] for qid in self.eval_ids if qid in self.qrels}


def _render(template: str, title: str, text: str) -> str:
    return template.format(title=(title or "").strip(), text=(text or "").strip()).strip()


def load_dataset_bundle(cfg: Config, dataset_name: str, *, cache_dir: Path | None = None) -> LoadedDataset:
    """Load corpus, queries, and qrels; subsample; split; build and persist the ID map."""
    from datasets import load_dataset as hf_load  # heavy import, deferred

    spec = cfg.datasets.get(dataset_name)
    if spec.hf_corpus is None or spec.hf_qrels is None:
        raise ValueError(
            f"{dataset_name} has no HF source configured"
            + (f": {spec.disabled_reason}" if spec.disabled_reason else "")
        )

    corpus = hf_load(spec.hf_corpus, "corpus", split="corpus", revision=spec.hf_corpus_revision)
    documents = [
        Document(
            doc_id=str(row["_id"]),
            text=_render(spec.text_template, row.get("title", ""), row.get("text", "")),
        )
        for row in corpus
    ]

    raw_queries = hf_load(spec.hf_corpus, "queries", split="queries", revision=spec.hf_corpus_revision)
    all_queries = {str(row["_id"]): str(row["text"]) for row in raw_queries}

    qrels = _load_qrels(hf_load, spec)
    _assert_grades(spec, qrels)

    # Only queries that have judgements are usable; an unjudged query silently drags
    # every arm's mean down by an identical amount, which hides nothing but wastes
    # the run.
    judged = [qid for qid in all_queries if qid in qrels and qrels[qid]]
    if not judged:
        raise QrelsError(f"{dataset_name}: no queries have judgements in split {spec.qrels_split!r}")

    cap = spec.query_cap if spec.query_cap is not None else cfg.run.queries.cap
    subsample = _subsample(judged, cap, cfg.run.seeds.query_subsample)
    warmup_ids, eval_ids = _hold_out_warmup(
        subsample, cfg.run.queries.warmup_count, cfg.run.seeds.warmup_holdout
    )

    id_map = IdMap.build(dataset_name, [d.doc_id for d in documents])
    _assert_qrels_resolve(dataset_name, qrels, subsample, id_map)

    bundle = LoadedDataset(
        spec=spec,
        documents=documents,
        queries={qid: all_queries[qid] for qid in subsample},
        qrels={qid: qrels[qid] for qid in subsample},
        warmup_ids=warmup_ids,
        eval_ids=eval_ids,
        id_map=id_map,
        all_queries=all_queries,
    )

    target = cache_dir or cfg.paths["queries"]
    id_map.save(target / f"{dataset_name}__idmap.json")
    _persist_query_split(target / f"{dataset_name}.json", bundle, cfg)
    return bundle


def _load_qrels(hf_load, spec: DatasetSpec) -> dict[str, dict[str, int]]:
    ds = hf_load(spec.hf_qrels, split=spec.qrels_split, revision=spec.hf_qrels_revision)
    out: dict[str, dict[str, int]] = {}
    for row in ds:
        qid, did = str(row["query-id"]), str(row["corpus-id"])
        # Graded judgements are preserved verbatim.
        out.setdefault(qid, {})[did] = int(row["score"])
    return out


def _assert_grades(spec: DatasetSpec, qrels: dict[str, dict[str, int]]) -> None:
    """Catch a parser that flattens graded judgements to binary."""
    if not spec.expected_grades:
        return
    observed = {grade for judgements in qrels.values() for grade in judgements.values()}
    expected = set(spec.expected_grades)
    unexpected = observed - expected
    if unexpected:
        raise QrelsError(
            f"{spec.name}: qrels contain unexpected grades {sorted(unexpected)}; "
            f"config declares {sorted(expected)}"
        )
    # NDCG needs the grades. If config says graded and we only see one positive level,
    # something collapsed them.
    positives_expected = {g for g in expected if g > 0}
    positives_observed = {g for g in observed if g > 0}
    if len(positives_expected) > 1 and len(positives_observed) <= 1:
        raise QrelsError(
            f"{spec.name}: expected graded relevance {sorted(positives_expected)} but only "
            f"observed {sorted(positives_observed)}. Grades appear to have been collapsed "
            "to binary; NDCG requires the grades."
        )


def _assert_qrels_resolve(
    dataset: str,
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
    id_map: IdMap,
) -> None:
    """Every judged document must exist in the corpus, or recall is silently wrong."""
    missing: list[str] = []
    for qid in query_ids:
        for doc_id in qrels.get(qid, {}):
            if doc_id not in id_map.to_point:
                missing.append(doc_id)
    if missing:
        sample = sorted(set(missing))[:5]
        raise IdMapError(
            f"{dataset}: {len(set(missing))} judged doc_ids are absent from the corpus, "
            f"e.g. {sample}. Recall would be understated for every arm."
        )


def _subsample(query_ids: list[str], cap: int | None, seed: int) -> list[str]:
    """Every judged query, sorted; or a fixed-seed sample when a cap is set.

    `cap=None` is the benchmark configuration — no dataset caps its queries.
    The cap path exists for smoke tests and debugging runs. Sorting
    first means HF row order cannot perturb which queries are chosen.
    """
    ordered = sorted(query_ids)
    if cap is None or len(ordered) <= cap:
        return ordered
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ordered), size=cap, replace=False)
    return [ordered[i] for i in sorted(picked.tolist())]


def _hold_out_warmup(
    query_ids: list[str], warmup_count: int, seed: int
) -> tuple[list[str], list[str]]:
    """Split off a warmup pool; everything else is evaluated.

    Warmup exists only so no evaluated query is served from a cache the warmup pass
    warmed. It is NOT a train split — nothing in this benchmark trains —
    so it takes a fixed `warmup_count` rather than a fraction, leaving the rest of the
    sample available for reporting instead of discarding half of it.

    If the sample is too small to spare a full warmup pool, warmup shrinks rather than
    starving the evaluation set: a dataset with few queries needs its queries reported
    more than it needs a big warmup.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(query_ids))
    # Never spend more than a third of a small sample on warmup.
    n_warmup = min(warmup_count, len(query_ids) // 3)
    warmup = sorted(query_ids[i] for i in order[:n_warmup])
    evaluated = sorted(query_ids[i] for i in order[n_warmup:])
    return warmup, evaluated


def _persist_query_split(path: Path, bundle: LoadedDataset, cfg: Config) -> None:
    """Persist the split so every arm and every resumed run sees identical queries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": bundle.spec.name,
        "seeds": cfg.run.seeds.model_dump(),
        "cap": bundle.spec.query_cap if bundle.spec.query_cap is not None else cfg.run.queries.cap,
        "warmup_count": cfg.run.queries.warmup_count,
        "n_corpus": len(bundle.documents),
        "corpus_digest": bundle.corpus_digest,
        "id_map_digest": bundle.id_map.digest,
        "warmup_ids": bundle.warmup_ids,
        "eval_ids": bundle.eval_ids,
        "queries": bundle.queries,
        "qrels": bundle.qrels,
    }
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("warmup_ids", "eval_ids", "corpus_digest", "id_map_digest"):
            if existing.get(key) != payload[key]:
                raise QrelsError(
                    f"{path}: persisted query split disagrees with the freshly computed one "
                    f"on {key!r}. The dataset or seeds changed under an existing run; "
                    "start a new results directory rather than mixing them."
                )
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def iter_batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
