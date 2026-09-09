"""Container lifecycle for the pinned Qdrant image.

A run builds its collection on a Qdrant started from wiped storage, so latency is
measured on an index built from nothing rather than on whatever a previous run left
behind. Storage is a Docker named volume; the destructive helper here refuses to touch
any volume but the one named in config.
"""

from __future__ import annotations

import functools
import subprocess
import time
import urllib.error
import urllib.request

from .config import Config


class QdrantControlError(RuntimeError):
    pass


class QdrantController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.container = cfg.run.qdrant.container_name
        self.volume = cfg.run.qdrant.storage_volume
        self.root = cfg.root.resolve()
        self.compose_file = cfg.root / "docker-compose.yml"

    # ------------------------------------------------------------- lifecycle

    @functools.cached_property
    def _launcher(self) -> str:
        """Pick a container launcher: compose plugin, compose v1, or plain docker run.

        The Compose plugin is not universally installed, so falling back matters rather
        than failing at step one. All three paths use the same pinned image and the same
        ports and mount, so the measurement is unaffected — which path was used is
        recorded in the manifest.
        """
        if _command_ok(["docker", "compose", "version"]):
            return "compose-plugin"
        if _command_ok(["docker-compose", "version"]):
            return "compose-v1"
        if _command_ok(["docker", "version"]):
            return "docker-run"
        raise QdrantControlError(
            "no usable container launcher found: need `docker compose`, `docker-compose`, "
            "or a working `docker` CLI"
        )

    def _compose(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        if self._launcher == "compose-plugin":
            cmd = ["docker", "compose", "-f", str(self.compose_file), *args]
        elif self._launcher == "compose-v1":
            cmd = ["docker-compose", "-f", str(self.compose_file), *args]
        else:
            raise QdrantControlError("_compose called without a compose launcher")
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def _docker_run(self) -> None:
        """Start the pinned image directly, mirroring docker-compose.yml.

        No `--platform` flag: this runs native to whatever host executes it, not
        emulated linux/amd64. Docker resolves the multi-arch
        manifest to the host's own architecture when none is forced.
        """
        cfg = self.cfg.run.qdrant
        subprocess.run(
            ["docker", "rm", "-f", self.container], capture_output=True, text=True, check=False
        )
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.container,
                "-p", "6333:6333",
                "-p", f"{cfg.grpc_port}:6334",
                # Same named volume as compose, so both launchers see one storage.
                "-v", f"{self.volume}:/qdrant/storage",
                "-e", "QDRANT__LOG_LEVEL=WARN",
                "-e", "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=4",
                "-e", "QDRANT__TELEMETRY_DISABLED=true",
                cfg.image,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise QdrantControlError(f"docker run failed: {result.stderr.strip()}")

    def up(self, *, wait: bool = True) -> None:
        if self._launcher == "docker-run":
            self._docker_run()
        else:
            self._compose("up", "-d")
        if wait:
            self.wait_ready()

    def down(self) -> None:
        if self._launcher == "docker-run":
            subprocess.run(
                ["docker", "rm", "-f", self.container],
                capture_output=True,
                text=True,
                check=False,
            )
            return
        self._compose("down", check=False)

    def wipe_storage(self) -> None:
        """Remove the storage volume, so the next `up` starts from an empty one.

        Guarded to one exact volume name from config: this is a destructive Docker
        operation, and a name that looks like a path or a prefix could take out
        something that is not ours. The container must be down first — Docker refuses
        to remove a volume still in use, which is the backstop if it is not.
        """
        volume = self.volume
        if not volume or "/" in volume or volume != self.cfg.run.qdrant.storage_volume:
            raise QdrantControlError(
                f"refusing to remove volume {volume!r}: must be exactly the configured "
                "run.qdrant.storage_volume and not a path"
            )
        result = subprocess.run(
            ["docker", "volume", "rm", volume], capture_output=True, text=True, check=False
        )
        stderr = result.stderr.strip()
        # A volume that does not exist yet is the expected state on a first run.
        if result.returncode != 0 and "no such volume" not in stderr.lower():
            raise QdrantControlError(f"could not remove volume {volume}: {stderr}")

    def restart_clean(self) -> None:
        """Full reset: stop, wipe, start. What a from-scratch collection build starts from."""
        self.down()
        self.wipe_storage()
        self.up()

    def wait_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.cfg.run.qdrant.url}/readyz", timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last = exc
            time.sleep(1.0)
        raise QdrantControlError(f"Qdrant not ready within {timeout_s}s: {last}")

    def image_digest(self) -> str | None:
        out = subprocess.run(
            ["docker", "image", "inspect", self.cfg.run.qdrant.image, "--format", "{{index .RepoDigests 0}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None


def _command_ok(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
