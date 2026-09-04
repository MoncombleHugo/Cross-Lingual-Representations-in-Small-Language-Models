from __future__ import annotations

from pathlib import Path

from cross_lingual_representations.plotting import (
    plot_alignment_vs_language,
    plot_language_probe_accuracy,
    plot_procrustes_r1,
    plot_tokenization_summary,
)


def test_phase_7_to_9_figures_render_headlessly(tmp_path: Path) -> None:
    procrustes_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    for layer, depth in enumerate((0.0, 1.0)):
        for condition, value in (("raw", 0.25 + layer * 0.1), ("aligned", 0.5 + layer * 0.1)):
            procrustes_rows.append(
                {
                    "metric": "r1",
                    "pooling": "last_token",
                    "condition": condition,
                    "normalized_depth": depth,
                    "value": value,
                }
            )
        retrieval_rows.append(
            {
                "metric": "r1",
                "pooling": "last_token",
                "condition": "raw",
                "normalized_depth": depth,
                "value": 0.3 + layer * 0.1,
            }
        )
        probe_rows.append(
            {
                "metric": "accuracy",
                "pooling": "last_token",
                "normalized_depth": depth,
                "value": 0.9,
                "n_classes": 4,
            }
        )
    tokenization_rows = [
        {"model": "model-a", "language": "en", "mean_tokens_per_100_characters": 20},
        {"model": "model-a", "language": "ko", "mean_tokens_per_100_characters": 40},
    ]

    outputs = (
        plot_procrustes_r1(
            procrustes_rows, tmp_path / "procrustes.png", pooling="last_token"
        ),
        plot_language_probe_accuracy(probe_rows, tmp_path / "probe.png"),
        plot_alignment_vs_language(
            retrieval_rows,
            probe_rows,
            tmp_path / "combined.png",
            pooling="last_token",
        ),
        plot_tokenization_summary(tokenization_rows, tmp_path / "tokenization.png"),
    )

    assert all(path.stat().st_size > 0 for path in outputs)

