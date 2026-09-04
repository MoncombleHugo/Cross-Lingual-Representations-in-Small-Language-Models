from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from cross_lingual_representations.data import (
    DEFAULT_LANGUAGES,
    load_flores_split,
    sample_parallel_rows,
)


def flores_rows(count: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            **{
                f"sentence_{code}": f"{language}-{index}"
                for language, code in DEFAULT_LANGUAGES.items()
            },
        }
        for index in range(count)
    ]


def test_sentence_ids_and_languages_remain_aligned() -> None:
    split = sample_parallel_rows(flores_rows(), split="dev", n_samples=8, seed=42)

    assert len(split) == 8
    for row in split.rows():
        for language in DEFAULT_LANGUAGES:
            assert row[language] == f"{language}-{row['id']}"


def test_sampling_is_deterministic_and_seed_sensitive() -> None:
    rows = flores_rows(50)

    first = sample_parallel_rows(rows, split="dev", n_samples=10, seed=7)
    repeated = sample_parallel_rows(rows, split="dev", n_samples=10, seed=7)
    different = sample_parallel_rows(rows, split="dev", n_samples=10, seed=8)

    assert first.ids == repeated.ids
    assert first.ids != different.ids


def test_requested_sample_count_is_validated() -> None:
    with pytest.raises(ValueError, match="Requested 21 samples"):
        sample_parallel_rows(flores_rows(), split="dev", n_samples=21)


def test_missing_language_column_is_rejected() -> None:
    rows = flores_rows(2)
    del rows[0]["sentence_kor_Hang"]

    with pytest.raises(ValueError, match="sentence_kor_Hang"):
        sample_parallel_rows(rows, split="dev")


def test_loader_receives_requested_split_and_revision() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_loader(name: str, config: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        calls.append((name, config, kwargs))
        return iter(flores_rows())

    result = load_flores_split(
        split="devtest",
        n_samples=3,
        revision="fixed-revision",
        load_dataset_fn=fake_loader,
    )

    assert len(result) == 3
    assert result.split == "devtest"
    assert calls == [
        ("facebook/flores", "all", {"split": "devtest", "revision": "fixed-revision"})
    ]


def test_duplicate_sentence_ids_are_rejected() -> None:
    rows = flores_rows(2)
    rows[1]["id"] = rows[0]["id"]

    with pytest.raises(ValueError, match="must be unique"):
        sample_parallel_rows(rows, split="dev")


def test_explicit_index_ids_and_direct_language_columns() -> None:
    rows = [{code: f"{code}-{index}" for code in DEFAULT_LANGUAGES.values()} for index in range(4)]

    split = sample_parallel_rows(
        rows,
        split="dev",
        n_samples=2,
        seed=1,
        text_field_template="{code}",
        id_field=None,
    )

    assert len(split) == 2
    assert all(row["en"] == f"eng_Latn-{row['id']}" for row in split.rows())
