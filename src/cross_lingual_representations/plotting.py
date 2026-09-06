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


def plot_procrustes_r1(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    pooling: str,
) -> Path:
    """Plot raw, train-centered, and Procrustes-aligned direction-averaged R@1."""
    selected = [
        row
        for row in rows
        if row["metric"] == "r1"
        and row["pooling"] == pooling
        and row["condition"] in {"raw", "centered", "aligned"}
    ]
    if not selected:
        raise ValueError(f"No Procrustes R@1 rows found for pooling {pooling!r}")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[str(row["condition"])][_number(row["normalized_depth"])].append(
            _number(row["value"])
        )
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for condition, by_depth in sorted(grouped.items()):
        depths = np.array(sorted(by_depth))
        values = np.array([np.mean(by_depth[depth]) for depth in depths])
        axis.plot(depths, values, marker="o", markersize=3, linewidth=1.8, label=condition)
    axis.set(
        title=f"Raw vs centered vs Procrustes-aligned retrieval ({pooling})",
        xlabel="Normalized depth",
        ylabel="R@1",
        xlim=(0.0, 1.0),
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_language_probe_accuracy(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot language-probe accuracy by pooling and normalized depth."""
    selected = [row for row in rows if row["metric"] == "accuracy"]
    if not selected:
        raise ValueError("No language-probe accuracy rows were provided")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[str(row["pooling"])][_number(row["normalized_depth"])].append(
            _number(row["value"])
        )
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for pooling, by_depth in sorted(grouped.items()):
        depths = np.array(sorted(by_depth))
        values = np.array([np.mean(by_depth[depth]) for depth in depths])
        axis.plot(depths, values, marker="o", markersize=3, linewidth=1.8, label=pooling)
    language_count = int(_number(selected[0]["n_classes"]))
    if language_count:
        axis.axhline(
            1.0 / language_count,
            color="0.45",
            linestyle="--",
            linewidth=1,
            label="random accuracy",
        )
    axis.set(
        title="Linear language separability",
        xlabel="Normalized depth",
        ylabel="Accuracy",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_tokenization_summary(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot mean tokens per 100 characters by model and language."""
    if not rows:
        raise ValueError("No tokenization summary rows were provided")
    models = sorted({str(row["model"]) for row in rows})
    languages = sorted({str(row["language"]) for row in rows})
    lookup = {
        (str(row["model"]), str(row["language"])): _number(
            row["mean_tokens_per_100_characters"]
        )
        for row in rows
    }
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(languages))
    width = 0.8 / len(models)
    for model_index, model in enumerate(models):
        values = [lookup[(model, language)] for language in languages]
        offset = (model_index - (len(models) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=model)
    axis.set(
        title="Multilingual tokenization",
        xlabel="Language",
        ylabel="Mean tokens per 100 characters",
        xticks=x,
        xticklabels=[language.upper() for language in languages],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_alignment_vs_language(
    retrieval_rows: Sequence[Mapping[str, object]],
    probe_rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    pooling: str,
) -> Path:
    """Plot semantic retrieval and language accuracy on aligned separate axes."""
    retrieval_selected = [
        row
        for row in retrieval_rows
        if row["metric"] == "r1"
        and row["condition"] == "raw"
        and row["pooling"] == pooling
    ]
    probe_selected = [
        row
        for row in probe_rows
        if row["metric"] == "accuracy" and row["pooling"] == pooling
    ]
    if not retrieval_selected or not probe_selected:
        raise ValueError(f"Missing retrieval or probe rows for pooling {pooling!r}")
    retrieval_by_depth: dict[float, list[float]] = defaultdict(list)
    for row in retrieval_selected:
        retrieval_by_depth[_number(row["normalized_depth"])].append(_number(row["value"]))
    retrieval_depths = np.array(sorted(retrieval_by_depth))
    retrieval_values = np.array(
        [np.mean(retrieval_by_depth[depth]) for depth in retrieval_depths]
    )
    probe_depths = np.array([_number(row["normalized_depth"]) for row in probe_selected])
    probe_values = np.array([_number(row["value"]) for row in probe_selected])
    order = np.argsort(probe_depths)

    figure = Figure(figsize=(8.0, 7.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    retrieval_axis, probe_axis = figure.subplots(2, 1, sharex=True)
    retrieval_axis.plot(retrieval_depths, retrieval_values, marker="o", markersize=3)
    retrieval_axis.set(title=f"Semantic alignment ({pooling})", ylabel="Average R@1")
    retrieval_axis.grid(alpha=0.25)
    probe_axis.plot(probe_depths[order], probe_values[order], marker="o", markersize=3)
    probe_axis.axhline(
        1.0 / int(_number(probe_selected[0]["n_classes"])),
        color="0.45",
        linestyle="--",
        linewidth=1,
    )
    probe_axis.set(
        title="Linearly accessible language identity",
        xlabel="Normalized depth",
        ylabel="Accuracy",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    probe_axis.grid(alpha=0.25)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_model_retrieval_comparison(
    rows: Sequence[Mapping[str, object]], output_path: str | Path, *, pooling: str
) -> Path:
    """Compare direction-averaged raw R@1 across models at normalized depth."""
    selected = [
        row
        for row in rows
        if row["metric"] == "r1"
        and row["condition"] == "raw"
        and row["pooling"] == pooling
    ]
    if not selected:
        raise ValueError(f"No raw R@1 rows found for pooling {pooling!r}")
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[str(row["model"])][_number(row["normalized_depth"])].append(
            _number(row["value"])
        )
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for model, by_depth in sorted(grouped.items()):
        depths = np.asarray(sorted(by_depth))
        values = np.asarray([np.mean(by_depth[depth]) for depth in depths])
        axis.plot(depths, values, marker="o", markersize=3, linewidth=1.8, label=model)
    axis.axhline(
        1.0 / int(_number(selected[0]["n"])),
        color="0.45",
        linestyle="--",
        linewidth=1,
        label="random R@1",
    )
    axis.set(
        title=f"Model comparison: translation retrieval ({pooling})",
        xlabel="Normalized depth",
        ylabel="Average R@1",
        xlim=(0.0, 1.0),
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_retrieval_heatmap(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    model: str,
    pooling: str,
    layer: int,
) -> Path:
    """Plot the directed-pair raw R@1 matrix for one model and layer."""
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["pooling"] == pooling
        and row["metric"] == "r1"
        and row["condition"] == "raw"
        and int(_number(row["layer"])) == layer
    ]
    if not selected:
        raise ValueError(f"No retrieval rows found for {model!r}, layer {layer}")
    languages = sorted(
        {str(row["source_language"]) for row in selected}
        | {str(row["target_language"]) for row in selected}
    )
    positions = {language: index for index, language in enumerate(languages)}
    matrix = np.full((len(languages), len(languages)), np.nan)
    for row in selected:
        matrix[
            positions[str(row["source_language"])],
            positions[str(row["target_language"])],
        ] = _number(row["value"])
    figure = Figure(figsize=(6.2, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(0.01, np.nanmax(matrix)))
    for source_index in range(len(languages)):
        for target_index in range(len(languages)):
            value = matrix[source_index, target_index]
            if np.isfinite(value):
                axis.text(target_index, source_index, f"{value:.3f}", ha="center", va="center")
    axis.set(
        xlabel="Target language",
        ylabel="Source language",
        xticks=np.arange(len(languages)),
        yticks=np.arange(len(languages)),
        xticklabels=[language.upper() for language in languages],
        yticklabels=[language.upper() for language in languages],
    )
    display_model = model.rsplit("/", maxsplit=1)[-1]
    axis.set_title(
        f"Directed R@1 at layer {layer}\n{display_model} ({pooling})",
        fontsize=12,
        pad=10,
    )
    figure.colorbar(image, ax=axis, label="R@1")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_geometry_diagnostics(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot layer-wise anisotropy and discriminability diagnostics."""
    specifications = (
        ("cosine_margin", "Matched - unmatched cosine"),
        ("unmatched_cosine", "Unmatched-pair cosine"),
        ("centroid_norm", "Norm of mean unit vector"),
        ("stable_rank", "Stable rank after centering"),
    )
    figure = Figure(figsize=(10.0, 7.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    for axis, (metric, title) in zip(axes.flat, specifications, strict=True):
        selected = [row for row in rows if row["metric"] == metric]
        if not selected:
            raise ValueError(f"No geometry rows found for metric {metric!r}")
        grouped: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in selected:
            grouped[str(row["pooling"])][_number(row["normalized_depth"])].append(
                _number(row["value"])
            )
        for pooling, by_depth in sorted(grouped.items()):
            depths = np.asarray(sorted(by_depth))
            values = np.asarray([np.mean(by_depth[depth]) for depth in depths])
            axis.plot(depths, values, marker="o", markersize=2.5, label=pooling)
        axis.set(title=title, ylabel=metric.replace("_", " "), xlim=(0.0, 1.0))
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Normalized depth")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Representation geometry diagnostics")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_bootstrap_retrieval(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot direction-averaged R@1 with bootstrap confidence bands."""
    if not rows:
        raise ValueError("No bootstrap rows were provided")
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pooling"])].append(row)
    for pooling, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: _number(row["normalized_depth"]))
        depths = np.asarray([_number(row["normalized_depth"]) for row in ordered])
        estimates = np.asarray([_number(row["estimate"]) for row in ordered])
        lower = np.asarray([_number(row["lower"]) for row in ordered])
        upper = np.asarray([_number(row["upper"]) for row in ordered])
        (line,) = axis.plot(depths, estimates, marker="o", markersize=3, label=pooling)
        axis.fill_between(depths, lower, upper, color=line.get_color(), alpha=0.18)
    confidence = 100.0 * _number(rows[0]["confidence"])
    axis.set(
        title=f"Translation retrieval with {confidence:.0f}% bootstrap intervals",
        xlabel="Normalized depth",
        ylabel="Average R@1",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_position_retrieval(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot direction-averaged retrieval by layer and relative token position."""
    if not rows:
        raise ValueError("No position-retrieval rows were provided")
    grouped: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[_number(row["relative_position"])][int(_number(row["layer"]))].append(
            _number(row["value"])
        )
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for fraction, by_layer in sorted(grouped.items()):
        layers = np.asarray(sorted(by_layer))
        values = np.asarray([np.mean(by_layer[layer]) for layer in layers])
        axis.plot(
            layers,
            values,
            marker="o",
            linewidth=1.8,
            label=f"{fraction:.0%} of sequence",
        )
    axis.set(
        title="Translation retrieval by relative token position",
        xlabel="Hidden-state layer",
        ylabel="Average R@1",
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination


def plot_block_stage_retrieval(
    rows: Sequence[Mapping[str, object]], output_path: str | Path
) -> Path:
    """Plot retrieval before attention, after attention, and after the full block."""
    selected = [row for row in rows if row["metric"] == "r1"]
    if not selected:
        raise ValueError("No block-stage R@1 rows were provided")
    stages = ("input", "post_attention", "output")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[str(row["pooling"])][str(row["stage"])].append(_number(row["value"]))
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(stages))
    for pooling, by_stage in sorted(grouped.items()):
        axis.plot(
            x,
            [np.mean(by_stage[stage]) for stage in stages],
            marker="o",
            linewidth=2,
            label=pooling,
        )
    block_layer = int(_number(selected[0]["block_layer"]))
    axis.set(
        title=f"Residual-stream retrieval decomposition — block {block_layer}",
        xlabel="Block stage",
        ylabel="Average R@1",
        xticks=x,
        xticklabels=("block input", "after attention", "after MLP"),
        ylim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    return destination
