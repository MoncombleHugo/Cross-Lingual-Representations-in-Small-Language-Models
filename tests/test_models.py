from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cross_lingual_representations.models import (
    LoadedModel,
    ModelLoadConfig,
    inspect_hidden_states,
    make_layer_labels,
    resolve_device,
    resolve_dtype,
)


class FakeTokenizer:
    padding_side = "right"
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        del text, kwargs
        return {
            "input_ids": torch.tensor([[1, 4, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel:
    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["output_hidden_states"] is True
        assert kwargs["use_cache"] is False
        return SimpleNamespace(
            hidden_states=(torch.zeros(1, 3, 8), torch.ones(1, 3, 8))
        )


def test_layer_labels_are_derived_from_observed_state_count() -> None:
    assert make_layer_labels(3) == ("layer_00_embedding", "layer_01", "layer_02")


def test_cpu_auto_dtype_is_float32() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
    assert resolve_device("cpu") == torch.device("cpu")


def test_unavailable_cuda_request_is_explicit() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("cuda")


def test_hidden_state_inspection_uses_observed_shapes() -> None:
    loaded = LoadedModel(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        parameter_count=123,
        config=ModelLoadConfig("fake/model", device="cpu"),
    )

    result = inspect_hidden_states(loaded, "hello")

    assert result.num_hidden_states == 2
    assert result.hidden_size == 8
    assert result.shapes == ((1, 3, 8), (1, 3, 8))
    assert result.special_tokens["pad_token_id"] == 0

