import pytest

from cross_lingual_representations.analysis import summarize_models, summary_markdown


def test_summarize_models_selects_best_layer_and_finite_alignment() -> None:
    retrieval = []
    probe = []
    procrustes = []
    for layer, value in enumerate((0.1, 0.4, 0.2)):
        for direction in (("en", "ko"), ("ko", "en")):
            retrieval.append(
                {
                    "model": "model",
                    "pooling": "last_token",
                    "metric": "r1",
                    "condition": "raw",
                    "layer": layer,
                    "source_language": direction[0],
                    "target_language": direction[1],
                    "value": value,
                }
            )
        probe.append(
            {
                "model": "model",
                "pooling": "last_token",
                "metric": "accuracy",
                "layer": layer,
                "value": 0.7 + layer / 10,
            }
        )
        procrustes.append(
            {
                "model": "model",
                "pooling": "last_token",
                "metric": "r1",
                "condition": "delta",
                "status": "ok",
                "value": 0.05,
            }
        )
    result = summarize_models(retrieval, procrustes, probe)
    assert result[0]["best_layer"] == 1
    assert result[0]["best_average_r1"] == 0.4
    assert result[0]["final_average_r1"] == 0.2
    assert result[0]["final_language_probe_accuracy"] == pytest.approx(0.9)
    assert "| model | last_token | 1 |" in summary_markdown(result)
