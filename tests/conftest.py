"""Shared fixtures.

Gates that need a live Qdrant and downloaded models are marked `slow` and skip cleanly
when the stack is absent, so the unit tests stay runnable on a laptop.
"""

from __future__ import annotations

import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def controller(cfg):
    from src.qdrant_control import QdrantController

    ctl = QdrantController(cfg)
    try:
        ctl.wait_ready(timeout_s=5)
    except Exception:
        pytest.skip("live Qdrant not available; start it with `docker compose up -d`")
    return ctl


@pytest.fixture(scope="session")
def encoders(cfg):
    pytest.importorskip("fastembed")
    from src.embedding import build_encoders

    return build_encoders(cfg)


@pytest.fixture(scope="session")
def nfcorpus(cfg):
    """The nfcorpus bundle, or a clean skip if it can't be fetched.

    `importorskip("datasets")` only checks the package is installed; a network failure
    while actually downloading the corpus would surface as a fixture ERROR rather than a
    SKIP, so the download is guarded too.
    """
    pytest.importorskip("datasets")
    from src.datasets import load_dataset_bundle

    try:
        return load_dataset_bundle(cfg, "nfcorpus")
    except Exception as exc:  # network/HF-hub failures land here, not just ImportError
        pytest.skip(f"could not load nfcorpus (offline, or HF Hub unreachable?): {exc}")


@pytest.fixture(scope="session")
def nfcorpus_collection(cfg, nfcorpus, encoders, controller):
    """The collection every arm queries, built once for the whole session."""
    from src.indexing import build_collection

    return build_collection(cfg, nfcorpus, encoders, controller)


@pytest.fixture
def arm_context(cfg, nfcorpus, encoders, nfcorpus_collection):
    from src.arms import ArmContext
    from src.indexing import build_client

    client = build_client(cfg, for_queries=True)
    ctx = ArmContext(
        client=client,
        cfg=cfg,
        encoders=encoders,
        id_map=nfcorpus.id_map,
        collection=nfcorpus_collection,
        dataset="nfcorpus",
    )
    yield ctx
    client.close()
