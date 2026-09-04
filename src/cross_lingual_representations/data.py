"""FLORES loading with explicit cross-language alignment checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_DATASET_NAME = "facebook/flores"
DEFAULT_DATASET_CONFIG = "all"
DEFAULT_LANGUAGES: dict[str, str] = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
}
VALID_SPLITS = frozenset({"dev", "devtest"})

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ParallelSplit:
    """A sampled FLORES split whose rows share one ordered sentence-ID sequence."""

    split: str
    ids: tuple[str, ...]
    sentences: Mapping[str, tuple[str, ...]]
    language_codes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported FLORES split: {self.split!r}")
        if not self.ids:
            raise ValueError("A parallel split must contain at least one sentence")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("FLORES sentence IDs must be unique")
        if set(self.sentences) != set(self.language_codes):
            raise ValueError("Sentence columns and language-code keys must match")
        expected = len(self.ids)
        for language, values in self.sentences.items():
            if len(values) != expected:
                raise ValueError(
                    f"Language {language!r} has {len(values)} rows; expected {expected}"
                )
            if any(not value.strip() for value in values):
                raise ValueError(f"Language {language!r} contains an empty sentence")

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self) -> Iterable[dict[str, str]]:
        """Yield aligned rows in their deterministic sampled order."""
        for index, sentence_id in enumerate(self.ids):
            yield {
                "id": sentence_id,
                **{language: values[index] for language, values in self.sentences.items()},
            }


def _validate_languages(languages: Mapping[str, str]) -> dict[str, str]:
    normalized = dict(languages)
    if not normalized:
        raise ValueError("At least one language must be requested")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Each language alias must map to a distinct FLORES code")
    return normalized


def _sample_indices(row_count: int, n_samples: int | None, seed: int) -> NDArray[np.intp]:
    if row_count <= 0:
        raise ValueError("Cannot sample an empty FLORES split")
    if n_samples is None:
        return np.arange(row_count, dtype=np.intp)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_samples > row_count:
        raise ValueError(f"Requested {n_samples} samples from only {row_count} rows")
    selected = np.random.default_rng(seed).choice(row_count, n_samples, replace=False)
    return cast(NDArray[np.intp], np.sort(selected).astype(np.intp, copy=False))


def sample_parallel_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    languages: Mapping[str, str] = DEFAULT_LANGUAGES,
    n_samples: int | None = None,
    seed: int = 42,
    text_field_template: str = "sentence_{code}",
    id_field: str | None = "id",
) -> ParallelSplit:
    """Validate and sample records from FLORES' ``all`` configuration.

    Sampling happens once at row level, so every selected ID is shared by all
    languages. Expected text fields are ``sentence_<FLORES code>``.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    language_codes = _validate_languages(languages)
    materialized = list(records)
    indices = _sample_indices(len(materialized), n_samples, seed)

    ids: list[str] = []
    sentences: dict[str, list[str]] = {language: [] for language in language_codes}
    for index in indices:
        row = materialized[int(index)]
        if id_field is None:
            sentence_id = str(int(index))
        else:
            if id_field not in row:
                raise ValueError(f"FLORES row is missing its {id_field!r} field")
            sentence_id = str(row[id_field])
        ids.append(sentence_id)
        for language, code in language_codes.items():
            field = text_field_template.format(code=code, language=language)
            if field not in row:
                raise ValueError(f"FLORES row {sentence_id!r} is missing {field!r}")
            value = row[field]
            if not isinstance(value, str):
                raise TypeError(f"FLORES field {field!r} must contain strings")
            sentences[language].append(value)

    return ParallelSplit(
        split=split,
        ids=tuple(ids),
        sentences={language: tuple(values) for language, values in sentences.items()},
        language_codes=language_codes,
    )


def load_flores_split(
    *,
    split: str,
    languages: Mapping[str, str] = DEFAULT_LANGUAGES,
    n_samples: int | None = None,
    seed: int = 42,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    revision: str | None = None,
    text_field_template: str = "sentence_{code}",
    id_field: str | None = "id",
    load_dataset_fn: DatasetLoader | None = None,
) -> ParallelSplit:
    """Load one FLORES split and return deterministic aligned multilingual rows."""
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    loader_kwargs: dict[str, Any] = {"split": split}
    if revision is not None:
        loader_kwargs["revision"] = revision
    if dataset_config is None:
        records = load_dataset_fn(dataset_name, **loader_kwargs)
    else:
        records = load_dataset_fn(dataset_name, dataset_config, **loader_kwargs)
    return sample_parallel_rows(
        records,
        split=split,
        languages=languages,
        n_samples=n_samples,
        seed=seed,
        text_field_template=text_field_template,
        id_field=id_field,
    )


def preview_rows(dataset: ParallelSplit, count: int = 5) -> str:
    """Format a small alignment preview for manual inspection."""
    if count <= 0:
        raise ValueError("count must be positive")
    lines: list[str] = []
    for row in list(dataset.rows())[:count]:
        lines.append(f"id={row['id']}")
        lines.extend(f"  {language}: {row[language]}" for language in dataset.sentences)
    return "\n".join(lines)
