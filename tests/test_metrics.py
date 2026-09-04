from __future__ import annotations

import numpy as np
import pytest

from cross_lingual_representations.metrics import retrieval_metrics, translation_ranks


def test_manually_constructed_ranks() -> None:
    metrics = retrieval_metrics(np.array([1, 2, 6, 11]))

    assert metrics["r1"] == 0.25
    assert metrics["r5"] == 0.5
    assert metrics["r10"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 6 + 1 / 11) / 4)


def test_translation_ranks_match_ids_instead_of_diagonal() -> None:
    similarities = np.array([[0.1, 0.9], [0.8, 0.2]])

    ranks = translation_ranks(similarities, ("a", "b"), ("b", "a"))

    np.testing.assert_array_equal(ranks, np.ones(2, dtype=np.float64))


def test_exact_ties_receive_average_rank() -> None:
    similarities = np.ones((1, 4))

    ranks = translation_ranks(similarities, ("a",), ("a", "b", "c", "d"))

    np.testing.assert_array_equal(ranks, np.array([2.5]))
    assert retrieval_metrics(ranks)["r1"] == 0.0
