from __future__ import annotations

from typing import Any

import pytest

from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.tokenization import (
    summarize_tokenization,
    tokenization_rows,
)


class CharacterTokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        add_special_tokens = kwargs["add_special_tokens"]
        extra = 2 if add_special_tokens else 0
        return {"input_ids": list(range(len(text) + extra))}


def test_tokenization_statistics_and_aligned_english_ratio() -> None:
    dataset = ParallelSplit(
        split="devtest",
        ids=("1", "2"),
        sentences={"en": ("ab", "abcd"), "ko": ("가", "가나")},
        language_codes={"en": "eng_Latn", "ko": "kor_Hang"},
    )

    rows = tokenization_rows(CharacterTokenizer(), dataset, model_name="test/model")
    korean_first = next(
        row for row in rows if row["language"] == "ko" and row["sentence_id"] == "1"
    )

    assert korean_first["characters"] == 1
    assert korean_first["bytes"] == 3
    assert korean_first["tokens"] == 3
    assert korean_first["token_ratio_to_en"] == 3 / 4
    summary = summarize_tokenization(rows)
    assert len(summary) == 2
    assert next(row for row in summary if row["language"] == "en")["mean_tokens"] == 5


def test_tokenization_requires_english_reference() -> None:
    dataset = ParallelSplit(
        split="dev",
        ids=("1",),
        sentences={"ko": ("가",)},
        language_codes={"ko": "kor_Hang"},
    )

    with pytest.raises(ValueError, match="English"):
        tokenization_rows(CharacterTokenizer(), dataset, model_name="test/model")

