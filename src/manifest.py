"""Run manifest: every pinned version, hardware description, seed, and exclusion.

The manifest is what makes a published number auditable, so it is
written incrementally and merged rather than overwritten — a crashed run still leaves
behind what it had established.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .platform_info import current_platform


def _installed_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "qdrant-client",
        "fastembed",
        "ranx",
        "datasets",
        "numpy",
        "pandas",
        "psutil",
        "pydantic",
        "typer",
        "scipy",
    ]
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _lockfile_digest(root: Path) -> dict[str, str | None]:
    from .platform_info import file_digest

    out: dict[str, str | None] = {}
    for name in ("uv.lock", "poetry.lock", "requirements.txt"):
        path = root / name
        out[name] = file_digest(path) if path.exists() else None
    return out


def _qdrant_version(url: str) -> str | None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read()).get("version")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def base_manifest(
    cfg: Config,
    *,
    image_digest: str | None = None,
    launcher: str | None = None,
    server_resources: dict | None = None,
) -> dict[str, Any]:
    info = current_platform()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "hardware": {
            **info.as_dict(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            # No fixed target: docker-compose.yml sets no `platform`, so Qdrant runs
            # native to whatever machine executes it by default.
        },
        "qdrant": {
            "image": cfg.run.qdrant.image,
            "image_digest": image_digest,
            "container_launcher": launcher,
            # What Qdrant itself had, which on macOS is the Docker VM, not the host above.
            "server_resources": server_resources,
            "reported_version": _qdrant_version(cfg.run.qdrant.url),
            "hnsw": cfg.run.qdrant.hnsw.model_dump(),
            "prefer_grpc": cfg.run.qdrant.prefer_grpc,
        },
        "models": {
            "dense": cfg.models.dense.model_dump(),
            "bm25": cfg.models.bm25.model_dump(),
            "late_interaction": cfg.models.late_interaction.model_dump(),
            "muvera": {**cfg.models.muvera.model_dump(), "dim": cfg.models.muvera.dim},
            "cross_encoder": cfg.models.cross_encoder.model_dump(),
        },
        "seeds": cfg.run.seeds.model_dump(),
        "execution": cfg.run.execution.model_dump(),
        "queries": cfg.run.queries.model_dump(),
        "dependencies": _installed_versions(),
        "lockfiles": _lockfile_digest(cfg.root),
        "git_commit": _git_commit(cfg.root),
        "measurement_caveats": [
            "Latency is unloaded serial latency: one client, one query in flight.",
            (
                "Memory is Qdrant's own per-collection, per-named-vector accounting for the "
                "same collection the arms query, so it describes one deployment. The numbers "
                "are estimates: ram_bytes runs 10-15% low because allocations inside "
                "third-party libraries are untracked, and cached_bytes is Unix-only."
            ),
            (
                "MUVERA FDEs come from fastembed's reference implementation; its random "
                "matrices are seeded and their digest is recorded below."
            ),
            (
                "Quality metrics come from run_idx 0 only; all timing runs feed the latency "
                "percentiles."
            ),
        ],
        "exclusions": {},
        "datasets": {},
    }


def _git_commit(root: Path) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or None


def merge(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge updates into the manifest on disk and return the result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    merged = _deep_merge(existing, updates)
    merged["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str))
    return merged


def _deep_merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def record_dataset(
    path: Path,
    dataset: str,
    *,
    corpus_digest: str,
    id_map_digest: str,
    n_documents: int,
    n_subsample: int,
    n_warmup: int,
    n_evaluated: int,
    representations: list[str],
    exclusions: dict[str, str],
    memory: dict | None = None,
    extra: dict | None = None,
) -> None:
    merge(
        path,
        {
            "datasets": {
                dataset: {
                    "corpus_digest": corpus_digest,
                    "id_map_digest": id_map_digest,
                    "n_documents": n_documents,
                    "n_sampled_queries": n_subsample,
                    "n_warmup_queries": n_warmup,
                    "n_evaluated_queries": n_evaluated,
                    "representations": representations,
                    "memory": memory or {},
                    **(extra or {}),
                }
            },
            "exclusions": {dataset: exclusions},
        },
    )
