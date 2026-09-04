"""Final cross-model summaries derived exclusively from saved tidy tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from cross_lingual_representations.retrieval import ResultRow


def _number(value: object) -> float:
    return float(cast(str | int | float, value))


def _layer_means(
    rows: Sequence[Mapping[str, object]], *, model: str, pooling: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["model"] == model
            and row["pooling"] == pooling
            and row["metric"] == "r1"
            and row["condition"] == "raw"
        ):
            grouped[int(_number(row["layer"]))].append(_number(row["value"]))
    return {layer: float(np.mean(values)) for layer, values in grouped.items()}


def summarize_models(
    retrieval_rows: Sequence[Mapping[str, object]],
    procrustes_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
    *,
    pooling: str = "last_token",
) -> list[ResultRow]:
    """Return one compact, reproducible final-analysis row per model."""
    models = sorted({str(row["model"]) for row in retrieval_rows})
    summary: list[ResultRow] = []
    for model in models:
        layer_means = _layer_means(retrieval_rows, model=model, pooling=pooling)
        if not layer_means:
            raise ValueError(f"No raw R@1 retrieval rows for {model!r} and {pooling!r}")
        best_layer = max(layer_means, key=layer_means.__getitem__)
        final_layer = max(layer_means)
        deltas = [
            _number(row["value"])
            for row in procrustes_rows
            if row["model"] == model
            and row["pooling"] == pooling
            and row["metric"] == "r1"
            and row["condition"] == "delta"
            and row["status"] == "ok"
            and np.isfinite(_number(row["value"]))
        ]
        final_probe = [
            _number(row["value"])
            for row in probe_rows
            if row["model"] == model
            and row["pooling"] == pooling
            and row["metric"] == "accuracy"
            and int(_number(row["layer"])) == final_layer
        ]
        if not deltas or len(final_probe) != 1:
            raise ValueError(f"Incomplete Procrustes or probe results for {model!r}")
        summary.append(
            {
                "model": model,
                "pooling": pooling,
                "best_layer": best_layer,
                "best_average_r1": layer_means[best_layer],
                "final_layer": final_layer,
                "final_average_r1": layer_means[final_layer],
                "mean_procrustes_delta_r1": float(np.mean(deltas)),
                "final_language_probe_accuracy": final_probe[0],
            }
        )
    return summary


def summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    """Render the compact summary as a Markdown table."""
    header = (
        "| Model | Pooling | Best layer | Best avg. R@1 | Final avg. R@1 | "
        "Mean Procrustes ΔR@1 | Final probe accuracy |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['pooling']} | {int(_number(row['best_layer']))} | "
            f"{_number(row['best_average_r1']):.4f} | "
            f"{_number(row['final_average_r1']):.4f} | "
            f"{_number(row['mean_procrustes_delta_r1']):+.4f} | "
            f"{_number(row['final_language_probe_accuracy']):.4f} |"
        )
    return "\n".join(lines) + "\n"
