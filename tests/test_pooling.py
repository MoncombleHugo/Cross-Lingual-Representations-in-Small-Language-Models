from __future__ import annotations

import pytest
import torch

from cross_lingual_representations.pooling import (
    last_token_pool,
    mean_pool,
    pool_hidden_states,
)


def test_last_token_pool_selects_final_non_padding_token() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [99.0, 99.0]],
            [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = last_token_pool(hidden, mask)

    assert pooled.shape == (2, 2)
    assert torch.equal(pooled, torch.tensor([[2.0, 20.0], [5.0, 50.0]]))


def test_mean_pool_excludes_padded_positions() -> None:
    hidden = torch.tensor([[[2.0], [4.0], [1000.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert torch.equal(mean_pool(hidden, mask), torch.tensor([[3.0]]))


def test_pool_hidden_states_stacks_layer_dimension() -> None:
    mask = torch.tensor([[1, 1]])
    states = (torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0))

    pooled = pool_hidden_states(states, mask, "mean")

    assert pooled.shape == (1, 2, 3)
    assert torch.equal(pooled[:, 0], torch.ones(1, 3))


def test_pooling_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_pool(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))

