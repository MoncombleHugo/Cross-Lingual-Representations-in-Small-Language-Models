"""Layer-wise multilingual translation retrieval."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, cast

import numpy as np

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.metrics import retrieval_metrics, translation_ranks
from cross_lingual_representations.similarity import cosine_similarity_matrix, l2_normalize

ResultRow = dict[str, str | int | float]


def _number(value: object) -> float:
    return float(cast(str | int | float, value))


def evaluate_direction(
    source_vectors: np.ndarray[Any, Any],
    target_vectors: np.ndarray[Any, Any],
    source_ids: Sequence[str],
    target_ids: Sequence[str],
) -> Mapping[str, float]:
    """Evaluate one source-target direction for one layer."""
    similarities = cosine_similarity_matrix(source_vectors, target_vectors)
    ranks = translation_ranks(similarities, source_ids, target_ids)
    return retrieval_metrics(ranks)


def evaluate_all_directions(
    bundle: RepresentationBundle,
    *,
    condition: str = "raw",
) -> list[ResultRow]:
    """Evaluate every directed language pair at every observed hidden state."""
    if len(bundle.languages) < 2:
        raise ValueError("Retrieval requires at least two languages")
    num_states = len(bundle.layer_labels)
    depth_denominator = max(num_states - 1, 1)
    language_positions = {language: index for index, language in enumerate(bundle.languages)}
    rows: list[ResultRow] = []
    for layer_index, _layer_label in enumerate(bundle.layer_labels):
        normalized_depth = layer_index / depth_denominator
        for source_language, target_language in permutations(bundle.languages, 2):
            source = bundle.vectors[language_positions[source_language], :, layer_index, :]
            target = bundle.vectors[language_positions[target_language], :, layer_index, :]
            metrics = evaluate_direction(
                source,
                target,
                bundle.sentence_ids,
                bundle.sentence_ids,
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "model": bundle.model_name,
                        "layer": layer_index,
                        "normalized_depth": normalized_depth,
                        "pooling": bundle.pooling,
                        "source_language": source_language,
                        "target_language": target_language,
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                        "n": len(bundle.sentence_ids),
                    }
                )
    return rows


def pair_similarity_rows(bundle: RepresentationBundle, *, seed: int = 42) -> list[ResultRow]:
    """Return matched and randomly unmatched cosine values for sanity checks."""
    if len(bundle.sentence_ids) < 2:
        raise ValueError("At least two sentences are needed for unmatched pairs")
    rng = np.random.default_rng(seed)
    language_positions = {language: index for index, language in enumerate(bundle.languages)}
    rows: list[ResultRow] = []
    for layer_index, _layer_label in enumerate(bundle.layer_labels):
        for source_language, target_language in permutations(bundle.languages, 2):
            source = l2_normalize(
                bundle.vectors[language_positions[source_language], :, layer_index, :]
            )
            target = l2_normalize(
                bundle.vectors[language_positions[target_language], :, layer_index, :]
            )
            matched = np.sum(source * target, axis=1)
            shift = int(rng.integers(1, len(bundle.sentence_ids)))
            unmatched = np.sum(source * np.roll(target, shift=shift, axis=0), axis=1)
            for pair_type, values in (("matched", matched), ("unmatched", unmatched)):
                rows.extend(
                    {
                        "model": bundle.model_name,
                        "layer": layer_index,
                        "pooling": bundle.pooling,
                        "source_language": source_language,
                        "target_language": target_language,
                        "pair_type": pair_type,
                        "sentence_id": sentence_id,
                        "value": float(value),
                    }
                    for sentence_id, value in zip(bundle.sentence_ids, values, strict=True)
                )
    return rows


def write_tidy_csv(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    """Write homogeneous tidy rows with stable column order."""
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("All result rows must contain identical fields")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def read_tidy_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a tidy CSV so plotting can depend on saved results, not live state."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_retrieval(rows: Sequence[Mapping[str, object]]) -> list[ResultRow]:
    """Average each metric across directed language pairs by layer and pooling."""
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["layer"],
            row["normalized_depth"],
            row["pooling"],
            row["condition"],
            row["metric"],
            row["n"],
        )
        grouped[key].append(_number(row["value"]))
    return [
        {
            "model": str(key[0]),
            "layer": int(_number(key[1])),
            "normalized_depth": _number(key[2]),
            "pooling": str(key[3]),
            "condition": str(key[4]),
            "metric": str(key[5]),
            "value": float(np.mean(values)),
            "n": int(_number(key[6])),
            "directions": len(values),
        }
        for key, values in grouped.items()
    ]
