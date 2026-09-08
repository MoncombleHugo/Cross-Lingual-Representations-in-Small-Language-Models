"""Leakage-safe non-parallel centroid estimation and retrieval evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.metrics import retrieval_metrics, translation_ranks
from cross_lingual_representations.procrustes import validate_train_evaluation_bundles
from cross_lingual_representations.retrieval import ResultRow
from cross_lingual_representations.similarity import l2_normalize


class ShuffleableRows(Protocol):
    def __iter__(self) -> Any: ...

    def shuffle(self, *, seed: int, buffer_size: int) -> ShuffleableRows: ...


@dataclass(frozen=True, slots=True)
class MonolingualSplit:
    """One independently sampled language corpus with stable example IDs."""

    split: str
    language: str
    language_code: str
    ids: tuple[str, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.texts):
            raise ValueError("A monolingual split needs equally sized, non-empty IDs and texts")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Monolingual example IDs must be unique")
        if any(not text.strip() for text in self.texts):
            raise ValueError("Monolingual texts must be non-empty")

    @property
    def sentences(self) -> Mapping[str, tuple[str, ...]]:
        return {self.language: self.texts}

    @property
    def language_codes(self) -> Mapping[str, str]:
        return {self.language: self.language_code}


def sample_monolingual_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    language: str,
    language_code: str,
    n_samples: int,
    text_field: str = "text",
    id_field: str | None = "id",
) -> MonolingualSplit:
    """Take the first valid unique texts from an independently shuffled stream."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        value = row.get(text_field)
        if not isinstance(value, str) or not value.strip():
            continue
        example_id = str(index) if id_field is None else str(row.get(id_field, index))
        if example_id in seen:
            continue
        seen.add(example_id)
        ids.append(example_id)
        texts.append(value.strip())
        if len(ids) == n_samples:
            break
    if len(ids) != n_samples:
        raise ValueError(f"Requested {n_samples} valid monolingual rows, found {len(ids)}")
    return MonolingualSplit(split, language, language_code, tuple(ids), tuple(texts))


def load_monolingual_split(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    language: str,
    language_code: str,
    n_samples: int,
    seed: int,
    shuffle_buffer_size: int,
    text_field: str,
    id_field: str | None,
    revision: str | None = None,
    load_dataset_fn: Any | None = None,
) -> MonolingualSplit:
    """Stream and independently shuffle one language configuration."""
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision
    records = load_dataset_fn(dataset_name, dataset_config, **kwargs)
    if hasattr(records, "shuffle"):
        records = records.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return sample_monolingual_rows(
        records,
        split=f"monolingual:{dataset_name}:{dataset_config}:{split}",
        language=language,
        language_code=language_code,
        n_samples=n_samples,
        text_field=text_field,
        id_field=id_field,
    )


def monolingual_cache_path(
    root: str | Path, model_cache_directory: str, corpus_name: str, language: str, pooling: str
) -> Path:
    """Build a cache path that cannot collide with aligned split caches."""
    safe_corpus = corpus_name.replace("/", "__")
    return (
        Path(root)
        / model_cache_directory
        / "monolingual"
        / safe_corpus
        / language
        / f"{pooling}.npz"
    )


def centroid_sample_indices(pool_size: int, n_samples: int, seed: int) -> NDArray[np.intp]:
    """Select a deterministic subset without replacement from a cached source pool."""
    if not 0 < n_samples <= pool_size:
        raise ValueError("n_samples must be positive and no larger than pool_size")
    return np.sort(
        np.random.default_rng(seed).choice(pool_size, n_samples, replace=False)
    ).astype(np.intp, copy=False)


def _centroids(
    source_bundles: Mapping[str, RepresentationBundle], indices: Mapping[str, NDArray[np.intp]]
) -> dict[str, NDArray[np.float32]]:
    result: dict[str, NDArray[np.float32]] = {}
    for language, bundle in source_bundles.items():
        if bundle.languages != (language,):
            raise ValueError(f"Source bundle for {language!r} must contain only that language")
        result[language] = np.asarray(
            bundle.vectors[0, indices[language], :, :], dtype=np.float32
        ).mean(axis=0)
    return result


def _validate_sources(
    source_bundles: Mapping[str, RepresentationBundle], evaluation: RepresentationBundle
) -> None:
    if set(source_bundles) != set(evaluation.languages):
        raise ValueError("A monolingual source bundle is required for every evaluation language")
    for language, source in source_bundles.items():
        if (
            source.model_name != evaluation.model_name
            or source.model_revision != evaluation.model_revision
        ):
            raise ValueError(f"Model mismatch in monolingual source for {language}")
        if source.pooling != evaluation.pooling or source.layer_labels != evaluation.layer_labels:
            raise ValueError(f"Representation mismatch in monolingual source for {language}")
        if source.max_length != evaluation.max_length:
            raise ValueError(f"max_length mismatch in monolingual source for {language}")
        if source.split == evaluation.split:
            raise ValueError("Monolingual source and evaluation split labels must differ")


def evaluate_centering_sample_efficiency(
    parallel_train: RepresentationBundle,
    source_bundles: Mapping[str, RepresentationBundle],
    evaluation: RepresentationBundle,
    *,
    sample_sizes: Sequence[int],
    seeds: Sequence[int],
    include_controls: bool = True,
) -> tuple[list[ResultRow], dict[str, dict[str, list[str]]]]:
    """Compare raw, parallel, non-parallel, and global centering on evaluation data."""
    validate_train_evaluation_bundles(parallel_train, evaluation)
    _validate_sources(source_bundles, evaluation)
    positions = {language: index for index, language in enumerate(evaluation.languages)}
    state_count = len(evaluation.layer_labels)
    depth_denominator = max(state_count - 1, 1)
    rows: list[ResultRow] = []
    selections: dict[str, dict[str, list[str]]] = {}
    similarity_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def paired_metrics(
        source: NDArray[np.float32], target: NDArray[np.float32]
    ) -> tuple[Mapping[str, float], Mapping[str, float]]:
        source_tensor = torch.from_numpy(source).to(similarity_device)
        target_tensor = torch.from_numpy(target).to(similarity_device)
        similarities = (source_tensor @ target_tensor.T).cpu().numpy()
        del source_tensor, target_tensor
        forward = retrieval_metrics(
            translation_ranks(
                similarities, evaluation.sentence_ids, evaluation.sentence_ids
            )
        )
        reverse = retrieval_metrics(
            translation_ranks(
                similarities.T, evaluation.sentence_ids, evaluation.sentence_ids
            )
        )
        return forward, reverse

    def append_condition(
        condition: str,
        means: Mapping[str, NDArray[np.float32]] | None,
        n_samples: int,
        seed: int,
    ) -> None:
        for layer in range(state_count):
            for first_language, second_language in combinations(evaluation.languages, 2):
                source = np.asarray(
                    evaluation.vectors[positions[first_language], :, layer, :], dtype=np.float32
                )
                target = np.asarray(
                    evaluation.vectors[positions[second_language], :, layer, :], dtype=np.float32
                )
                status = "ok"
                metrics_by_direction: tuple[Mapping[str, float], Mapping[str, float]]
                if means is not None:
                    source = source - means[first_language][layer]
                    target = target - means[second_language][layer]
                    if np.any(np.linalg.norm(source, axis=1) <= 1e-12) or np.any(
                        np.linalg.norm(target, axis=1) <= 1e-12
                    ):
                        status = "degenerate_zero_norm"
                        metrics_by_direction = (
                            {"r1": float("nan"), "mrr": float("nan")},
                            {"r1": float("nan"), "mrr": float("nan")},
                        )
                    else:
                        source = l2_normalize(source)
                        target = l2_normalize(target)
                        metrics_by_direction = paired_metrics(source, target)
                else:
                    metrics_by_direction = paired_metrics(
                        l2_normalize(source), l2_normalize(target)
                    )
                for source_language, target_language, metrics in (
                    (first_language, second_language, metrics_by_direction[0]),
                    (second_language, first_language, metrics_by_direction[1]),
                ):
                    for metric in ("r1", "mrr"):
                        rows.append(
                            {
                                "model": evaluation.model_name,
                                "layer": layer,
                                "normalized_depth": layer / depth_denominator,
                                "pooling": evaluation.pooling,
                                "centering_condition": condition,
                                "n_centroid_samples": n_samples,
                                "seed": seed,
                                "source_language": source_language,
                                "target_language": target_language,
                                "metric": metric,
                                "status": status,
                                "value": metrics[metric],
                                "r1": metrics["r1"],
                                "mrr": metrics["mrr"],
                                "n_eval": len(evaluation.sentence_ids),
                            }
                        )

    if include_controls:
        append_condition("raw", None, 0, evaluation.seed)
        parallel_means = {
            language: np.asarray(parallel_train.vectors[position], dtype=np.float32).mean(
                axis=0
            )
            for language, position in positions.items()
        }
        append_condition(
            "parallel-centered",
            parallel_means,
            len(parallel_train.sentence_ids),
            parallel_train.seed,
        )

    for seed in seeds:
        for n_samples in sample_sizes:
            indices = {
                language: centroid_sample_indices(len(bundle.sentence_ids), n_samples, seed)
                for language, bundle in source_bundles.items()
            }
            selections[f"seed={seed},n={n_samples}"] = {
                language: [bundle.sentence_ids[int(index)] for index in indices[language]]
                for language, bundle in source_bundles.items()
            }
            language_means = _centroids(source_bundles, indices)
            append_condition("nonparallel-centered", language_means, n_samples, seed)
            global_mean = np.mean(np.stack(list(language_means.values()), axis=0), axis=0)
            append_condition(
                "global-centered",
                {language: global_mean for language in evaluation.languages},
                n_samples,
                seed,
            )
    return rows, selections
