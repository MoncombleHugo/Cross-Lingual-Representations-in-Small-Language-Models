from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch

from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.extraction import extract_representations
from cross_lingual_representations.models import LoadedModel, ModelLoadConfig


class BatchTokenizer:
    padding_side = "right"
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __call__(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        del kwargs
        batch_size = 1 if isinstance(text, str) else len(text)
        return {
            "input_ids": torch.tensor([[1, 2, 0]] * batch_size),
            "attention_mask": torch.tensor([[1, 1, 0]] * batch_size),
        }


class HiddenStateModel:
    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        input_ids = kwargs["input_ids"].to(dtype=torch.float32)
        state = input_ids.unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(hidden_states=(state, state + 10))


def test_extraction_caches_only_sentence_level_shapes() -> None:
    loaded = LoadedModel(
        model=HiddenStateModel(),
        tokenizer=BatchTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        parameter_count=1,
        config=ModelLoadConfig("fake/model", device="cpu"),
    )
    dataset = ParallelSplit(
        split="devtest",
        ids=("1", "2", "3"),
        sentences={"en": ("a", "b", "c"), "ko": ("가", "나", "다")},
        language_codes={"en": "eng_Latn", "ko": "kor_Hang"},
    )

    bundles = extract_representations(
        loaded,
        dataset,
        pooling_methods=("last_token", "mean"),
        batch_size=2,
        max_length=8,
        seed=7,
        representation_dtype="float32",
        show_progress=False,
    )

    assert bundles["last_token"].vectors.shape == (2, 3, 2, 2)
    assert bundles["mean"].vectors.shape == (2, 3, 2, 2)
    assert bundles["last_token"].seed == 7
    assert bundles["last_token"].sentence_ids == dataset.ids

