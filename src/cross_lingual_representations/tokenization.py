"""Per-sentence multilingual tokenizer statistics and summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import numpy as np
import torch

from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.retrieval import ResultRow


class AnalysisTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


def _token_count(tokenizer: AnalysisTokenizer, text: str, add_special_tokens: bool) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    if "input_ids" not in encoded:
        raise ValueError("Tokenizer output is missing input_ids")
    token_ids = encoded["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        return int(token_ids.numel())
    if isinstance(token_ids, Sequence):
        if token_ids and isinstance(token_ids[0], Sequence):
            if len(token_ids) != 1:
                raise ValueError("Expected one tokenized sentence")
            return len(token_ids[0])
        return len(token_ids)
    raise TypeError("Unsupported tokenizer input_ids type")


def tokenization_rows(
    tokenizer: AnalysisTokenizer,
    dataset: ParallelSplit,
    *,
    model_name: str,
    add_special_tokens: bool = True,
    english_alias: str = "en",
) -> list[ResultRow]:
    """Measure text and token lengths, including aligned ratios to English."""
    if english_alias not in dataset.sentences:
        raise ValueError(f"English reference alias {english_alias!r} is missing")
    counts: dict[str, list[int]] = {
        language: [
            _token_count(tokenizer, text, add_special_tokens)
            for text in dataset.sentences[language]
        ]
        for language in dataset.sentences
    }
    if any(count <= 0 for language_counts in counts.values() for count in language_counts):
        raise ValueError("Every sentence must produce at least one token")

    rows: list[ResultRow] = []
    for language, texts in dataset.sentences.items():
        for index, (sentence_id, text, tokens) in enumerate(
            zip(dataset.ids, texts, counts[language], strict=True)
        ):
            characters = len(text)
            byte_count = len(text.encode("utf-8"))
            english_tokens = counts[english_alias][index]
            rows.append(
                {
                    "model": model_name,
                    "split": dataset.split,
                    "language": language,
                    "sentence_id": sentence_id,
                    "characters": characters,
                    "bytes": byte_count,
                    "tokens": tokens,
                    "tokens_per_100_characters": 100 * tokens / max(characters, 1),
                    "bytes_per_token": byte_count / tokens,
                    "characters_per_token": characters / tokens,
                    "token_ratio_to_en": tokens / english_tokens,
                }
            )
    return rows


def summarize_tokenization(rows: Sequence[Mapping[str, object]]) -> list[ResultRow]:
    """Aggregate tokenizer statistics by model and language."""
    numeric_fields = (
        "characters",
        "bytes",
        "tokens",
        "tokens_per_100_characters",
        "bytes_per_token",
        "characters_per_token",
        "token_ratio_to_en",
    )
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["language"]))].append(row)
    summary: list[ResultRow] = []
    for (model, language), group in grouped.items():
        output: ResultRow = {"model": model, "language": language, "n": len(group)}
        for field in numeric_fields:
            values = np.asarray(
                [float(cast(str | int | float, row[field])) for row in group]
            )
            output[f"mean_{field}"] = float(values.mean())
            output[f"median_{field}"] = float(np.median(values))
        summary.append(output)
    return summary
