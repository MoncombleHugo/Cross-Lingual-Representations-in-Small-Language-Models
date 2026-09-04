"""README-ready figures generated exclusively from tidy retrieval rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _number(value: object) -> float:
    return float(cast(str | int | float, value))


def plot_retrieval_r1(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    average_all_directions: bool,
    pooling: str | None = None,
) -> Path:
    """Plot R@1 by normalized layer depth, averaging directions as requested."""
    selected = [
        row
        for row in rows
        if row["metric"] == "r1"
        and row["condition"] == "raw"
        and (pooling is None or row["pooling"] == pooling)
    ]
    if not selected:
        raise ValueError("No raw R@1 rows were provided")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        if average_all_directions:
            label = str(row["pooling"])
        else:
            languages = sorted((str(row["source_language"]), str(row["target_language"])))
            label = " ↔ ".join(language.upper() for language in languages)
        grouped[label][_number(row["normalized_depth"])].append(_number(row["value"]))

    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for label, by_depth in sorted(grouped.items()):
        depths = np.array(sorted(by_depth))
        values = np.array([np.mean(by_depth[depth]) for depth in depths])
        axis.plot(depths, values, marker="o", markersize=3, linewidth=1.8, label=label)
    n = int(_number(selected[0]["n"]))
    axis.axhline(1.0 / n, color="0.45", linestyle="--", linewidth=1, label="random R@1")
    axis.set(title="Cross-lingual translation retrieval", xlabel="Normalized depth", ylabel="R@1")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination
