"""Encoders and the platform-keyed embedding cache.

Corpus embeddings are expensive and reused across a dataset's indexing passes, so they
are cached. The cache key includes the platform tag, and a mismatched tag is a hard
error, never a warning.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config, CrossEncoderModel, DenseModel, ModelsConfig
from .muvera import build_muvera
from .platform_info import assert_same_platform, current_platform, file_digest, stable_digest

# ----------------------------------------------------------------------- encoders


@dataclass
class Encoders:
    """Lazily-constructed model handles. Models load on first use, not at import."""

    models: ModelsConfig
    muvera_seed: int
    _dense: object | None = field(default=None, repr=False)
    _late_interaction: object | None = field(default=None, repr=False)
    _cross: object | None = field(default=None, repr=False)
    _muvera: object | None = field(default=None, repr=False)

    @property
    def dense(self):
        if self._dense is None:
            from fastembed import TextEmbedding

            dense = self.models.dense
            if dense.onnx is not None:
                _register_onnx_source(dense)
            self._dense = TextEmbedding(model_name=dense.model_id)
        return self._dense

    @property
    def late_interaction(self):
        if self._late_interaction is None:
            from fastembed import LateInteractionTextEmbedding

            self._late_interaction = LateInteractionTextEmbedding(
                model_name=self.models.late_interaction.name
            )
        return self._late_interaction

    @property
    def cross(self):
        if self._cross is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cross = self.models.cross_encoder
            if cross.onnx is not None:
                _register_onnx_reranker(cross)
            self._cross = TextCrossEncoder(model_name=cross.model_id)
        return self._cross

    @property
    def muvera(self):
        """fastembed's Muvera postprocessor."""
        if self._muvera is None:
            self._muvera = build_muvera(
                self.models.muvera, self.models.late_interaction.dim, self.muvera_seed
            )
        return self._muvera

    # ------------------------------------------------------------------- queries

    def embed_query_dense(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self.dense.query_embed(text))), dtype=np.float32)

    def embed_query_late_interaction(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self.late_interaction.query_embed(text))), dtype=np.float32)

    # ------------------------------------------------------------------ documents

    def embed_docs_dense(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = list(self.dense.embed(list(texts), batch_size=batch_size))
        return np.asarray(out, dtype=np.float32)

    def embed_docs_late_interaction(
        self, texts: Sequence[str], batch_size: int = 16
    ) -> list[np.ndarray]:
        return [
            np.asarray(tokens, dtype=np.float32)
            for tokens in self.late_interaction.embed(list(texts), batch_size=batch_size)
        ]

    # -------------------------------------------------------------------- pinning

    def model_file_digests(self) -> dict[str, str]:
        """SHA-256 of resolved model files. The real revision pin."""
        digests: dict[str, str] = {}
        for root in _fastembed_cache_roots():
            for path in sorted(root.rglob("*.onnx")):
                try:
                    digests[str(path.relative_to(root))] = file_digest(path)
                except OSError:
                    continue
        return digests


def _register_onnx_reranker(cross: CrossEncoderModel) -> None:
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if any(entry["model"] == cross.model_id for entry in TextCrossEncoder.list_supported_models()):
        return
    TextCrossEncoder.add_custom_model(
        model=cross.model_id,
        sources=ModelSource(hf=cross.onnx.hf),
        model_file=cross.onnx.file,
        size_in_gb=cross.size_gb,
    )


def _register_onnx_source(dense: DenseModel) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    if any(entry["model"] == dense.model_id for entry in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(
        model=dense.model_id,
        pooling=PoolingType[dense.onnx.pooling.upper()],
        normalization=dense.onnx.normalize,
        sources=ModelSource(hf=dense.onnx.hf),
        dim=dense.dim,
        model_file=dense.onnx.file,
    )


def _fastembed_cache_roots() -> list[Path]:
    import os

    candidates = [
        Path(os.environ.get("FASTEMBED_CACHE_PATH", ""))
        if os.environ.get("FASTEMBED_CACHE_PATH")
        else None,
        Path.home() / ".cache" / "fastembed",
        Path("/tmp/fastembed_cache"),
    ]
    return [c for c in candidates if c and c.exists()]


# -------------------------------------------------------------------------- cache


def _atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a temp file, then publish it at `path` in one filesystem rename.

    A reader only ever sees `path` fully written or not present at all — never a
    truncated file from a write that was interrupted partway.
    """
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


class EmbeddingCache:
    """On-disk corpus embedding cache, keyed by model + platform + corpus."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        *,
        representation: str,
        model_name: str,
        dataset: str,
        corpus_digest: str,
        extra: dict | None = None,
    ) -> str:
        return stable_digest(
            {
                "representation": representation,
                "model": model_name,
                "dataset": dataset,
                "corpus_digest": corpus_digest,
                "platform": current_platform().tag,
                "extra": extra or {},
            }
        )[:32]

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{key}.meta.json"

    def _data_path(self, key: str, suffix: str) -> Path:
        return self.root / f"{key}{suffix}"

    def _write_meta(self, key: str, meta: dict) -> None:
        payload = {**meta, "platform_tag": current_platform().tag}
        text = json.dumps(payload, indent=2, sort_keys=True)
        _atomic_write(self._meta_path(key), lambda tmp: tmp.write_text(text))

    def _read_meta(self, key: str, *, context: str) -> dict | None:
        path = self._meta_path(key)
        if not path.exists():
            return None
        meta = json.loads(path.read_text())
        assert_same_platform(meta.get("platform_tag", "unknown"), context=context)
        return meta

    def meta(self, key: str) -> dict | None:
        """Cached metadata for a key, or None if nothing is cached under it."""
        return self._read_meta(key, context=f"cache {key}")

    # --- dense -------------------------------------------------------------

    def load_dense(self, key: str) -> np.ndarray | None:
        if self._read_meta(key, context=f"dense cache {key}") is None:
            return None
        path = self._data_path(key, ".dense.npy")
        return np.load(path) if path.exists() else None

    def save_dense(self, key: str, vectors: np.ndarray, meta: dict) -> None:
        def _write(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                np.save(f, vectors)

        _atomic_write(self._data_path(key, ".dense.npy"), _write)
        self._write_meta(key, {**meta, "kind": "dense", "shape": list(vectors.shape)})

    # --- late_interaction (ragged) --------------------------------------------------

    def load_late_interaction(self, key: str) -> list[np.ndarray] | None:
        if self._read_meta(key, context=f"colbert cache {key}") is None:
            return None
        path = self._data_path(key, ".colbert.npz")
        if not path.exists():
            return None
        with np.load(path) as data:
            flat, offsets, dim = data["flat"], data["offsets"], int(data["dim"])
        return [flat[offsets[i] : offsets[i + 1]].reshape(-1, dim) for i in range(len(offsets) - 1)]

    def save_late_interaction(self, key: str, matrices: list[np.ndarray], meta: dict) -> None:
        dim = matrices[0].shape[1] if matrices else 0
        offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
        for i, mat in enumerate(matrices):
            offsets[i + 1] = offsets[i] + mat.shape[0] * dim
        flat = (
            np.concatenate([m.astype(np.float32).reshape(-1) for m in matrices])
            if matrices
            else np.zeros(0, np.float32)
        )

        def _write(tmp: Path) -> None:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, flat=flat, offsets=offsets, dim=np.int64(dim))

        _atomic_write(self._data_path(key, ".colbert.npz"), _write)
        self._write_meta(
            key,
            {
                **meta,
                "kind": "colbert",
                "n": len(matrices),
                "n_token_vectors": int(sum(m.shape[0] for m in matrices)),
            },
        )


def build_encoders(cfg: Config) -> Encoders:
    return Encoders(models=cfg.models, muvera_seed=cfg.run.seeds.muvera)


class UnknownModelDimError(RuntimeError):
    pass


def resolve_dense_model_dim(model_name: str) -> int:
    """Look up a FastEmbed text-embedding model's output dimension.

    Used when the baseline dense model is overridden at runtime (--dense-model): the
    dimension has to be known before a collection can be created, and asking the user to
    supply it by hand for every override is both friction and a silent-corruption risk if
    they get it wrong. FastEmbed's own registry is the source of truth.
    """
    from fastembed import TextEmbedding

    for entry in TextEmbedding.list_supported_models():
        if entry.get("model") == model_name:
            dim = entry.get("dim")
            if isinstance(dim, int):
                return dim
            break
    raise UnknownModelDimError(
        f"could not resolve the output dimension for dense model {model_name!r} from "
        "FastEmbed's supported-model list. Pass --dense-model-dim explicitly."
    )
