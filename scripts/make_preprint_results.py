"""Build the compact preprint tables from final intervention CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("results/raw")
OUTPUT = Path("results/tables/preprint_main_results.md")
EXPERIMENTS = {"tri_05b": "Tri-0.5B", "qwen_05b": "Qwen2.5-0.5B"}


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    lines = ["# Preprint — main results", "", "## Table 1 — FLORES causal intervention", ""]
    lines += [
        "| model | pooling | normal final R@1 | language final R@1 | PCA final R@1 | "
        "random final R@1 | language - PCA | CI 95 % |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for experiment, model in EXPERIMENTS.items():
        data = pd.read_csv(ROOT / f"{experiment}_intervention_basis_comparison.csv")
        metrics = data[
            (data["record_type"] == "metric")
            & (data["metric"] == "r1")
            & (data["layer"] == 24)
        ]
        bootstrap = data[
            (data["record_type"] == "bootstrap_delta")
            & (data["comparison"] == "remove_language-remove_pca")
        ]
        for pooling in ("mean", "last_token"):
            selected = metrics[metrics["pooling"] == pooling]
            values = selected.groupby("condition")["value"].mean()
            interval = bootstrap[bootstrap["pooling"] == pooling].iloc[0]
            lines.append(
                f"| {model} | {pooling} | {_fmt(values['normal'])} | "
                f"{_fmt(values['remove_language'])} | {_fmt(values['remove_pca'])} | "
                f"{_fmt(values['remove_random'])} | {_fmt(float(interval['value']))} | "
                f"[{_fmt(float(interval['ci_lower']))}; {_fmt(float(interval['ci_upper']))}] |"
            )

    lines += ["", "## Table 2 — MASSIVE", ""]
    lines += [
        "| model | pooling | normal target accuracy | language target accuracy | "
        "PCA target accuracy | random target accuracy | EN accuracy normal | "
        "EN accuracy language |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment, model in EXPERIMENTS.items():
        data = pd.read_csv(ROOT / f"{experiment}_massive_intervention.csv")
        metrics = data[(data["record_type"] == "metric") & (data["metric"] == "accuracy")]
        for pooling in ("mean", "last_token"):
            selected = metrics[metrics["pooling"] == pooling]
            target = selected[selected["test_language"].isin(("ko", "ja", "zh"))]
            target_values = target.groupby("condition")["value"].mean()
            english = selected[selected["test_language"] == "en"].groupby("condition")[
                "value"
            ].mean()
            lines.append(
                f"| {model} | {pooling} | {_fmt(target_values['normal'])} | "
                f"{_fmt(target_values['language'])} | {_fmt(target_values['pca'])} | "
                f"{_fmt(target_values['random'])} | {_fmt(english['normal'])} | "
                f"{_fmt(english['language'])} |"
            )

    lines += ["", "## Table 3 — Language modeling cost", ""]
    lines += [
        "| model | language | normal NLL | language NLL | PCA NLL | delta language | delta PCA |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for experiment, model in EXPERIMENTS.items():
        data = pd.read_csv(ROOT / f"{experiment}_intervention_lm_eval.csv")
        for language in ("en", "ko", "ja", "zh"):
            selected = data[data["language"] == language].set_index("condition")
            lines.append(
                f"| {model} | {language.upper()} | {_fmt(selected.loc['normal', 'value'])} | "
                f"{_fmt(selected.loc['language', 'value'])} | "
                f"{_fmt(selected.loc['pca', 'value'])} | "
                f"{_fmt(selected.loc['language', 'delta_nll'])} | "
                f"{_fmt(selected.loc['pca', 'delta_nll'])} |"
            )

    lines += [
        "",
        "Target accuracy is the unweighted mean over KO, JA, and ZH. Random values are "
        "means over five reproducible orthonormal bases. All confidence intervals use 1,000 "
        "paired bootstrap resamples.",
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved preprint table: {OUTPUT}")


if __name__ == "__main__":
    main()
