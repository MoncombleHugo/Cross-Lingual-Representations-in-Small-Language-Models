"""Sentence pooling for token-level causal-LM hidden states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

PoolingMethod = Literal["last_token", "mean"]
SUPPORTED_POOLING = frozenset({"last_token", "mean"})


def _validate_inputs(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must have shape [batch, tokens, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, tokens]")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_state and attention_mask batch/token dimensions must match")
    if torch.any(attention_mask.sum(dim=1) <= 0):
        raise ValueError("Every sequence must contain at least one non-padding token")


def last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the final non-padding token for each right-padded sequence."""
    _validate_inputs(hidden_state, attention_mask)
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, lengths - 1]


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token states while excluding padding positions."""
    _validate_inputs(hidden_state, attention_mask)
    mask = attention_mask.to(device=hidden_state.device, dtype=hidden_state.dtype).unsqueeze(-1)
    numerator = (hidden_state * mask).sum(dim=1)
    denominator = mask.sum(dim=1)
    return numerator / denominator


def pool_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    method: PoolingMethod,
) -> torch.Tensor:
    """Pool every observed hidden state into ``[batch, states, hidden]``."""
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")
    pooler = {"last_token": last_token_pool, "mean": mean_pool}.get(method)
    if pooler is None:
        raise ValueError(f"Unsupported pooling method: {method!r}")
    pooled = [pooler(hidden_state, attention_mask) for hidden_state in hidden_states]
    if len({tuple(tensor.shape) for tensor in pooled}) != 1:
        raise ValueError("All pooled hidden states must have the same shape")
    return torch.stack(pooled, dim=1)

