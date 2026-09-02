"""MUVERA unit tests.

The FDE construction is fastembed's `Muvera`, so its internal mechanics (SimHash,
partition fill, projection) are not this project's to test. What is: constructing it
correctly from config, and the digest that pins it in the manifest. Real validation is
`test_muvera_calibration.py`, on real embeddings — synthetic data cannot validate an FDE.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import MuveraModel
from src.muvera import build_muvera, chamfer_similarity, matrices_digest


@pytest.fixture
def model() -> MuveraModel:
    return MuveraModel(
        source_tokens="colbert",
        k_sim=5,
        n_reps=20,
        d_proj=16,
        distance="cosine",
    )


def test_dimension_matches_the_configured_formula(model):
    assert model.dim == model.n_reps * model.n_partitions * model.d_proj == 10240
    muvera = build_muvera(model, token_dim=128, seed=20250813)
    assert muvera.embedding_size == model.dim


def test_process_document_and_query_produce_the_configured_dimension(model):
    muvera = build_muvera(model, token_dim=128, seed=1)
    rng = np.random.default_rng(0)
    doc = rng.standard_normal((40, 128)).astype(np.float32)
    query = rng.standard_normal((24, 128)).astype(np.float32)
    assert muvera.process_document(doc).shape == (model.dim,)
    assert muvera.process_query(query).shape == (model.dim,)


def test_same_seed_is_deterministic_across_instances(model):
    a = build_muvera(model, token_dim=128, seed=7)
    b = build_muvera(model, token_dim=128, seed=7)
    rng = np.random.default_rng(1)
    tokens = rng.standard_normal((25, 128)).astype(np.float32)
    np.testing.assert_array_equal(a.process_document(tokens), b.process_document(tokens))
    assert matrices_digest(a, model, 128) == matrices_digest(b, model, 128)


def test_different_seeds_give_different_encodings(model):
    a = build_muvera(model, token_dim=128, seed=7)
    b = build_muvera(model, token_dim=128, seed=8)
    assert matrices_digest(a, model, 128) != matrices_digest(b, model, 128)
    rng = np.random.default_rng(1)
    tokens = rng.standard_normal((25, 128)).astype(np.float32)
    assert not np.array_equal(a.process_document(tokens), b.process_document(tokens))


def test_chamfer_similarity_matches_a_direct_max_sim_computation():
    rng = np.random.default_rng(3)
    query = rng.standard_normal((4, 8)).astype(np.float32)
    doc = rng.standard_normal((6, 8)).astype(np.float32)
    expected = sum(max(float(q @ d) for d in doc) for q in query)
    assert chamfer_similarity(query, doc) == pytest.approx(expected, rel=1e-5)
