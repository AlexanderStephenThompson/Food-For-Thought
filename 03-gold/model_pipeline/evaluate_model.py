"""Build the model evaluation report for one model build.

Condenses cross-validation metrics, calibration honesty, per-cuisine
recall, and the confusion-versus-taxonomy cross-check into a JSON artifact
plus readable Markdown (mirroring the coverage and fold-balance reports).
Confusion pairs are annotated with the silver taxonomy's neighbor
similarity, so "the model confuses what the data says are neighbors" is
checkable at a glance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy
from sklearn.metrics import accuracy_score, f1_score

from model_pipeline.calibrate_blend import LIKELIHOOD_FLOOR, METRIC_DECIMALS
from silver_pipeline.artifact_io import (
    SCHEMA_VERSION,
    write_artifact_json,
    write_text_atomically,
)

CONFUSION_PAIR_REPORT_COUNT = 15
EVALUATION_JSON_FILENAME = "evaluation.json"
EVALUATION_MARKDOWN_FILENAME = "evaluation.md"
LINE_SEPARATOR = "\n"
MISSING_SIMILARITY_LABEL = "—"
FOLD_METRIC_NAMES = ("accuracy", "macro_f1", "log_loss")


def compute_fold_metrics(
    true_indices, probabilities, cuisine_ids: Sequence[str]
) -> dict:
    """Compute one fold's accuracy, macro-F1, and log loss.

    Args:
        true_indices: True cuisine index per held-out sample.
        probabilities: Predicted probabilities (sample x cuisine).
        cuisine_ids: Sorted cuisine identifiers (label order).

    Returns:
        {"accuracy", "macro_f1", "log_loss"} rounded to METRIC_DECIMALS.
    """
    probabilities = numpy.asarray(probabilities, dtype=float)
    true_indices = numpy.asarray(true_indices)
    predictions = probabilities.argmax(axis=1)
    row_positions = numpy.arange(len(true_indices))
    likelihoods = numpy.clip(
        probabilities[row_positions, true_indices], LIKELIHOOD_FLOOR, None
    )
    return {
        "accuracy": round(float(accuracy_score(true_indices, predictions)), METRIC_DECIMALS),
        "log_loss": round(float(-numpy.log(likelihoods).mean()), METRIC_DECIMALS),
        "macro_f1": round(
            float(
                f1_score(
                    true_indices,
                    predictions,
                    average="macro",
                    labels=range(len(cuisine_ids)),
                    zero_division=0,
                )
            ),
            METRIC_DECIMALS,
        ),
    }


def summarize_confusion_pairs(
    true_ids: Sequence[str],
    predicted_ids: Sequence[str],
    cuisines_payload: dict,
    top_count: int = CONFUSION_PAIR_REPORT_COUNT,
) -> list[dict]:
    """Rank misclassification pairs and annotate them with the taxonomy.

    Args:
        true_ids: True cuisine per out-of-fold sample.
        predicted_ids: Predicted cuisine per out-of-fold sample.
        cuisines_payload: Parsed silver cuisines.json document.
        top_count: Pairs to keep, by descending count.

    Returns:
        [{true_cuisine, predicted_cuisine, count, neighbor_similarity}]
        where neighbor_similarity is the taxonomy's value when the
        predicted cuisine is a top-4 neighbor of the true one, else None.
    """
    similarity_by_pair = {
        (cuisine["id"], neighbor["id"]): neighbor["similarity"]
        for cuisine in cuisines_payload["cuisines"]
        for neighbor in cuisine["neighbors"]
    }
    confusion_counts = Counter(
        (true_id, predicted_id)
        for true_id, predicted_id in zip(true_ids, predicted_ids)
        if true_id != predicted_id
    )
    ranked_pairs = sorted(
        confusion_counts.items(), key=lambda item: (-item[1], item[0])
    )[:top_count]
    return [
        {
            "count": count,
            "neighbor_similarity": similarity_by_pair.get(pair),
            "predicted_cuisine": pair[1],
            "true_cuisine": pair[0],
        }
        for pair, count in ranked_pairs
    ]


def build_evaluation_payload(
    grid_results: Sequence[dict],
    configuration: dict,
    fold_metrics: Sequence[dict],
    calibration_summary: dict,
    per_cuisine_recall: Sequence[dict],
    confusion_pairs: Sequence[dict],
    baseline_naive_bayes: dict,
    fingerprint: dict,
) -> dict:
    """Assemble the evaluation artifact from its computed sections.

    Args:
        grid_results: One {c_value, parent_weight, pooled_oof_log_loss}
            per evaluated grid configuration.
        configuration: The winning configuration echo.
        fold_metrics: Per-fold {fold, accuracy, macro_f1, log_loss}.
        calibration_summary: {temperature, ece_before, ece_after}.
        per_cuisine_recall: Per-cuisine {cuisine, recipe_count, recall}.
        confusion_pairs: Output of summarize_confusion_pairs.
        baseline_naive_bayes: Baseline metrics for comparison.
        fingerprint: Model build block embedded in the artifact.

    Returns:
        Payload ready for write_artifact_json.
    """
    mean_metrics = {
        metric_name: round(
            sum(fold[metric_name] for fold in fold_metrics) / len(fold_metrics),
            METRIC_DECIMALS,
        )
        for metric_name in FOLD_METRIC_NAMES
    }
    return {
        "baseline_naive_bayes": dict(baseline_naive_bayes),
        "build": dict(fingerprint),
        "calibration": dict(calibration_summary),
        "configuration": dict(configuration),
        "confusion_pairs": list(confusion_pairs),
        "folds": list(fold_metrics),
        "grid_search": list(grid_results),
        "mean": mean_metrics,
        "per_cuisine": list(per_cuisine_recall),
        "schema_version": SCHEMA_VERSION,
    }


def render_evaluation_markdown(payload: dict) -> str:
    """Render the evaluation payload as a readable Markdown report."""
    configuration = payload["configuration"]
    calibration = payload["calibration"]
    baseline = payload["baseline_naive_bayes"]
    lines = [
        "# Model evaluation",
        "",
        f"- Model configuration: C={configuration['c_value']}, "
        f"parent_weight={configuration['parent_weight']}",
        f"- Temperature: {calibration['temperature']}",
        f"- ECE before -> after calibration: {calibration['ece_before']} -> "
        f"{calibration['ece_after']}",
        "",
        "## Cross-validation (per fold)",
        "",
        "| Fold | Accuracy | Macro F1 | Log loss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for fold in payload["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['accuracy']} | {fold['macro_f1']} | "
            f"{fold['log_loss']} |"
        )
    mean = payload["mean"]
    lines.extend(
        [
            f"| mean | {mean['accuracy']} | {mean['macro_f1']} | "
            f"{mean['log_loss']} |",
            "",
            "## Baseline comparison",
            "",
            "| Model | Pooled OOF log loss | Mean accuracy | Mean macro F1 |",
            "| --- | ---: | ---: | ---: |",
            f"| logistic regression | {_find_winner_loss(payload)} | "
            f"{mean['accuracy']} | {mean['macro_f1']} |",
            f"| naive bayes | {baseline['pooled_oof_log_loss']} | "
            f"{baseline['mean_accuracy']} | {baseline['mean_macro_f1']} |",
            "",
            "## Per-cuisine recall",
            "",
            "| Cuisine | Recipes | Recall |",
            "| --- | ---: | ---: |",
        ]
    )
    for entry in payload["per_cuisine"]:
        lines.append(
            f"| {entry['cuisine']} | {entry['recipe_count']} | {entry['recall']} |"
        )
    lines.extend(
        [
            "",
            "## Top confusion pairs (vs taxonomy neighbors)",
            "",
            "| True | Predicted | Count | Neighbor similarity |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for pair in payload["confusion_pairs"]:
        similarity = pair["neighbor_similarity"]
        similarity_cell = (
            MISSING_SIMILARITY_LABEL if similarity is None else str(similarity)
        )
        lines.append(
            f"| {pair['true_cuisine']} | {pair['predicted_cuisine']} | "
            f"{pair['count']} | {similarity_cell} |"
        )
    lines.append("")
    return LINE_SEPARATOR.join(lines)


def _find_winner_loss(payload: dict) -> float | str:
    """Pull the winning grid entry's pooled loss for the comparison table."""
    configuration = payload["configuration"]
    for entry in payload["grid_search"]:
        if (
            entry["c_value"] == configuration["c_value"]
            and entry["parent_weight"] == configuration["parent_weight"]
        ):
            return entry["pooled_oof_log_loss"]
    return MISSING_SIMILARITY_LABEL


def write_evaluation_reports(payload: dict, reports_directory: Path) -> None:
    """Atomically write the evaluation JSON artifact and Markdown report.

    Args:
        payload: Evaluation payload from build_evaluation_payload.
        reports_directory: Destination directory; must exist.
    """
    write_artifact_json(payload, reports_directory / EVALUATION_JSON_FILENAME)
    write_text_atomically(
        render_evaluation_markdown(payload),
        reports_directory / EVALUATION_MARKDOWN_FILENAME,
    )
