"""Platform identity and hardware description.

Embeddings generated on macOS/MPS differ in low-order bits from x86
CPU inference, enough to flip near-tied rankings. The cache key includes the platform
tag so a cross-architecture reuse is impossible by accident rather than by discipline.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


class PlatformMismatchError(RuntimeError):
    """Raised when a cache entry was produced on a different platform."""


@dataclass(frozen=True)
class PlatformInfo:
    system: str
    machine: str
    python: str
    onnx_providers: tuple[str, ...]
    cpu_model: str
    logical_cpus: int
    total_ram_gb: float

    @property
    def tag(self) -> str:
        """Short identity used in cache keys. Deliberately coarse but architecture-safe."""
        provider = self.onnx_providers[0] if self.onnx_providers else "unknown"
        return f"{self.system}-{self.machine}-py{self.python}-{provider}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["onnx_providers"] = list(self.onnx_providers)
        d["tag"] = self.tag
        return d


def _onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime  # noqa: PLC0415  (optional at import time)

        return tuple(onnxruntime.get_available_providers())
    except Exception:
        return ()


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform.startswith("linux"):
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                for line in cpuinfo.read_text().splitlines():
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def current_platform() -> PlatformInfo:
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return PlatformInfo(
        system=platform.system().lower(),
        machine=platform.machine().lower(),
        python=py,
        onnx_providers=_onnx_providers(),
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count() or 0,
        total_ram_gb=round(psutil.virtual_memory().total / 1024**3, 2),
    )


def assert_same_platform(recorded_tag: str, *, context: str) -> None:
    """Hard-fail on a cross-architecture cache hit.

    A warning would be ignored on the one run where it mattered, so this raises.
    """
    actual = current_platform().tag
    if recorded_tag != actual:
        raise PlatformMismatchError(
            f"{context}: cached artifact was produced on {recorded_tag!r} but this "
            f"machine is {actual!r}. Embeddings must never be carried across "
            "architectures. Regenerate on the target machine."
        )


def file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_digest(obj: object) -> str:
    """SHA-256 of a JSON-canonicalised object. Used for plan and matrix digests."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
