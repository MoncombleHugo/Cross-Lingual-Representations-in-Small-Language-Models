from __future__ import annotations

import numpy as np

from cross_lingual_representations.language_subspace import (
    centroid_language_subspace,
    project_out,
    random_orthonormal_basis,
)


def test_centroid_subspace_recovers_low_rank_language_directions() -> None:
    rng = np.random.default_rng(4)
    offsets = np.asarray([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    shared = rng.normal(scale=0.01, size=(4, 20, 3))
    vectors = shared + offsets[:, None, :]

    basis, explained = centroid_language_subspace(vectors)

    assert explained[:2].sum() > 0.999
    assert abs(basis[2, 0]) < 0.02
    assert abs(basis[2, 1]) < 0.02


def test_projection_removes_only_selected_direction() -> None:
    vectors = np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)

    cleaned = project_out(vectors, basis)

    np.testing.assert_allclose(cleaned, [[0.0, 4.0, 5.0]])


def test_random_control_is_reproducible_and_orthonormal() -> None:
    first = random_orthonormal_basis(12, 3, 42)
    second = random_orthonormal_basis(12, 3, 42)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(3), atol=1e-12)
