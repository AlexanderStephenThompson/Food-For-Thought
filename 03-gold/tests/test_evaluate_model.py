"""Tests for model_pipeline.evaluate_model."""

import numpy

from model_pipeline.evaluate_model import (
    build_evaluation_payload,
    compute_fold_metrics,
    render_evaluation_markdown,
    summarize_confusion_pairs,
)
from tests.model_payload_builders import (
    MODEL_BUILD_BLOCK,
    make_cuisines_payload,
)

CUISINE_IDS = ("italian", "mexican", "thai")


def _make_fold_metrics():
    return [
        {"fold": 0, "accuracy": 0.8, "macro_f1": 0.75, "log_loss": 0.6},
        {"fold": 1, "accuracy": 0.9, "macro_f1": 0.85, "log_loss": 0.4},
    ]


def test_compute_fold_metrics_reports_three_metrics():
    true_indices = numpy.array([0, 1, 2, 0])
    probabilities = numpy.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.5, 0.2],
        ]
    )

    metrics = compute_fold_metrics(true_indices, probabilities, CUISINE_IDS)

    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert metrics["log_loss"] > 0.0


def test_confusion_pairs_annotated_with_neighbor_similarity():
    true_ids = ["italian", "italian", "thai", "thai", "thai"]
    predicted_ids = ["mexican", "mexican", "italian", "italian", "italian"]

    pairs = summarize_confusion_pairs(
        true_ids, predicted_ids, make_cuisines_payload(), top_count=5
    )

    pair_by_key = {
        (entry["true_cuisine"], entry["predicted_cuisine"]): entry
        for entry in pairs
    }
    assert pair_by_key[("italian", "mexican")]["neighbor_similarity"] == 0.45
    assert pair_by_key[("thai", "italian")]["neighbor_similarity"] is None
    assert pairs[0]["count"] == 3


def test_evaluation_payload_reports_fold_and_mean_metrics():
    payload = build_evaluation_payload(
        grid_results=[
            {"c_value": 1.0, "parent_weight": 0.3, "pooled_oof_log_loss": 0.5}
        ],
        configuration={"c_value": 1.0, "parent_weight": 0.3},
        fold_metrics=_make_fold_metrics(),
        calibration_summary={
            "temperature": 1.2,
            "ece_before": 0.08,
            "ece_after": 0.02,
        },
        per_cuisine_recall=[
            {"cuisine": cuisine, "recipe_count": 4, "recall": 0.9}
            for cuisine in CUISINE_IDS
        ],
        confusion_pairs=[],
        baseline_naive_bayes={
            "pooled_oof_log_loss": 0.9,
            "mean_accuracy": 0.7,
            "mean_macro_f1": 0.65,
        },
        fingerprint=MODEL_BUILD_BLOCK,
    )

    assert payload["mean"]["accuracy"] == 0.85
    assert payload["mean"]["log_loss"] == 0.5
    assert payload["folds"] == _make_fold_metrics()
    assert payload["baseline_naive_bayes"]["mean_accuracy"] == 0.7


def test_evaluation_markdown_lists_per_cuisine_recall():
    payload = build_evaluation_payload(
        grid_results=[],
        configuration={"c_value": 1.0, "parent_weight": 0.3},
        fold_metrics=_make_fold_metrics(),
        calibration_summary={
            "temperature": 1.2,
            "ece_before": 0.08,
            "ece_after": 0.02,
        },
        per_cuisine_recall=[
            {"cuisine": cuisine, "recipe_count": 4, "recall": 0.9}
            for cuisine in CUISINE_IDS
        ],
        confusion_pairs=[
            {
                "true_cuisine": "italian",
                "predicted_cuisine": "mexican",
                "count": 2,
                "neighbor_similarity": 0.45,
            }
        ],
        baseline_naive_bayes={
            "pooled_oof_log_loss": 0.9,
            "mean_accuracy": 0.7,
            "mean_macro_f1": 0.65,
        },
        fingerprint=MODEL_BUILD_BLOCK,
    )

    markdown = render_evaluation_markdown(payload)

    assert "italian" in markdown
    assert "thai" in markdown
    assert "0.45" in markdown
