"""Score blends from rounded parameters and calibrate them with temperature.

This module owns the single scoring path (logits -> softmax blend) that the
build, the validator, and the predict CLI all share, so every artifact on
disk is exactly reproducible from parameters.json plus calibration.json.
The temperature is fit on pooled out-of-fold logits with a fixed-iteration
golden-section search — bit-deterministic, unlike optimizer heuristics
whose stopping rules drift between library versions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy

from silver_pipeline.artifact_io import SCHEMA_VERSION

ECE_BIN_COUNT = 10
GOLDEN_SECTION_ITERATIONS = 100
GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0
TEMPERATURE_LOG10_RANGE = (-2.0, 2.0)
TEMPERATURE_DECIMALS = 6
METRIC_DECIMALS = 6
BLEND_DECIMALS = 4
LIKELIHOOD_FLOOR = 1e-15

CALIBRATION_FILENAME = "calibration.json"
BLENDS_FILENAME = "blends_test.json"


def compute_logits_from_parameters(
    feature_matrix, coefficients: Sequence[Sequence[float]], intercepts: Sequence[float]
) -> numpy.ndarray:
    """Compute per-cuisine logits as feature_matrix @ coefficients.T + intercepts.

    Args:
        feature_matrix: Dense array or scipy sparse matrix, one row per recipe.
        coefficients: Per-cuisine coefficient rows (cuisine x feature).
        intercepts: Per-cuisine intercepts.

    Returns:
        Logit array of shape (recipe_count, cuisine_count).
    """
    coefficient_matrix = numpy.asarray(coefficients, dtype=float)
    intercept_vector = numpy.asarray(intercepts, dtype=float)
    return numpy.asarray(feature_matrix @ coefficient_matrix.T) + intercept_vector


def convert_logits_to_blend(logits, temperature: float) -> numpy.ndarray:
    """Convert logits to a calibrated blend via temperature-scaled softmax.

    Args:
        logits: Logit array of shape (recipe_count, cuisine_count).
        temperature: Softmax temperature; 1.0 leaves logits unscaled.

    Returns:
        Probability array of the same shape; every row sums to 1.
    """
    scaled = numpy.asarray(logits, dtype=float) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = numpy.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def compute_negative_log_likelihood(
    logits, label_indices, temperature: float
) -> float:
    """Mean negative log likelihood of the true labels at one temperature.

    Args:
        logits: Logit array of shape (sample_count, cuisine_count).
        label_indices: True cuisine index per sample.
        temperature: Softmax temperature to evaluate.

    Returns:
        Mean negative log likelihood (the log loss), as a float.
    """
    blend = convert_logits_to_blend(logits, temperature)
    row_positions = numpy.arange(len(label_indices))
    likelihoods = blend[row_positions, numpy.asarray(label_indices)]
    return float(-numpy.log(numpy.clip(likelihoods, LIKELIHOOD_FLOOR, None)).mean())


def fit_temperature(oof_logits, label_indices) -> float:
    """Fit the blend temperature on pooled out-of-fold logits.

    Golden-section search over log10(T) in TEMPERATURE_LOG10_RANGE for a
    fixed GOLDEN_SECTION_ITERATIONS iterations, minimizing the negative log
    likelihood — deterministic across runs by construction.

    Args:
        oof_logits: Pooled out-of-fold logits (sample x cuisine).
        label_indices: True cuisine index per sample.

    Returns:
        The fitted temperature, rounded to TEMPERATURE_DECIMALS.
    """

    def loss_at(log_temperature: float) -> float:
        return compute_negative_log_likelihood(
            oof_logits, label_indices, 10.0**log_temperature
        )

    low, high = TEMPERATURE_LOG10_RANGE
    inner_low = high - GOLDEN_RATIO * (high - low)
    inner_high = low + GOLDEN_RATIO * (high - low)
    loss_low = loss_at(inner_low)
    loss_high = loss_at(inner_high)
    for _ in range(GOLDEN_SECTION_ITERATIONS):
        if loss_low < loss_high:
            high, inner_high, loss_high = inner_high, inner_low, loss_low
            inner_low = high - GOLDEN_RATIO * (high - low)
            loss_low = loss_at(inner_low)
        else:
            low, inner_low, loss_low = inner_low, inner_high, loss_high
            inner_high = low + GOLDEN_RATIO * (high - low)
            loss_high = loss_at(inner_high)
    return round(10.0 ** ((low + high) / 2.0), TEMPERATURE_DECIMALS)


def compute_expected_calibration_error(probabilities, label_indices) -> float:
    """Expected calibration error over ECE_BIN_COUNT equal-width bins.

    Bins are taken over top-class confidence; each bin contributes its
    count-weighted |accuracy - mean confidence|.

    Args:
        probabilities: Probability array (sample x cuisine).
        label_indices: True cuisine index per sample.

    Returns:
        The ECE, rounded to METRIC_DECIMALS.
    """
    probabilities = numpy.asarray(probabilities, dtype=float)
    label_indices = numpy.asarray(label_indices)
    confidences = probabilities.max(axis=1)
    is_correct = probabilities.argmax(axis=1) == label_indices
    bin_positions = numpy.minimum(
        (confidences * ECE_BIN_COUNT).astype(int), ECE_BIN_COUNT - 1
    )
    total_count = len(label_indices)
    error = 0.0
    for bin_number in range(ECE_BIN_COUNT):
        inside = bin_positions == bin_number
        count = int(inside.sum())
        if count == 0:
            continue
        gap = abs(float(is_correct[inside].mean()) - float(confidences[inside].mean()))
        error += (count / total_count) * gap
    return round(error, METRIC_DECIMALS)


def summarize_reliability_bins(probabilities, label_indices) -> list[dict]:
    """Per-bin confidence vs accuracy table for the reliability report.

    Args:
        probabilities: Probability array (sample x cuisine).
        label_indices: True cuisine index per sample.

    Returns:
        One dict per bin: {bin, count, mean_confidence, accuracy} (zeros
        for empty bins), rounded to METRIC_DECIMALS.
    """
    probabilities = numpy.asarray(probabilities, dtype=float)
    label_indices = numpy.asarray(label_indices)
    confidences = probabilities.max(axis=1)
    is_correct = probabilities.argmax(axis=1) == label_indices
    bin_positions = numpy.minimum(
        (confidences * ECE_BIN_COUNT).astype(int), ECE_BIN_COUNT - 1
    )
    bins = []
    for bin_number in range(ECE_BIN_COUNT):
        inside = bin_positions == bin_number
        count = int(inside.sum())
        mean_confidence = (
            round(float(confidences[inside].mean()), METRIC_DECIMALS) if count else 0.0
        )
        accuracy = (
            round(float(is_correct[inside].mean()), METRIC_DECIMALS) if count else 0.0
        )
        bins.append(
            {
                "accuracy": accuracy,
                "bin": bin_number,
                "count": count,
                "mean_confidence": mean_confidence,
            }
        )
    return bins


def build_calibration_payload(
    temperature: float, oof_logits, label_indices, fingerprint: dict
) -> dict:
    """Build the calibration artifact with before/after honesty metrics.

    Args:
        temperature: Fitted blend temperature.
        oof_logits: Pooled out-of-fold logits the temperature was fit on.
        label_indices: True cuisine index per out-of-fold sample.
        fingerprint: Model build block embedded in the artifact.

    Returns:
        Payload ready for write_artifact_json.
    """
    blend_before = convert_logits_to_blend(oof_logits, 1.0)
    blend_after = convert_logits_to_blend(oof_logits, temperature)
    return {
        "build": dict(fingerprint),
        "ece_bin_count": ECE_BIN_COUNT,
        "optimizer": {
            "iterations": GOLDEN_SECTION_ITERATIONS,
            "log10_range": list(TEMPERATURE_LOG10_RANGE),
            "method": "golden_section",
        },
        "out_of_fold": {
            "ece_after": compute_expected_calibration_error(
                blend_after, label_indices
            ),
            "ece_before": compute_expected_calibration_error(
                blend_before, label_indices
            ),
            "log_loss_after": round(
                compute_negative_log_likelihood(
                    oof_logits, label_indices, temperature
                ),
                METRIC_DECIMALS,
            ),
            "log_loss_before": round(
                compute_negative_log_likelihood(oof_logits, label_indices, 1.0),
                METRIC_DECIMALS,
            ),
        },
        "reliability": {
            "after": summarize_reliability_bins(blend_after, label_indices),
            "before": summarize_reliability_bins(blend_before, label_indices),
        },
        "schema_version": SCHEMA_VERSION,
        "temperature": temperature,
    }


def build_blends_payload(
    blend_matrix, recipe_ids: Sequence[int], cuisine_ids: Sequence[str],
    fingerprint: dict,
) -> dict:
    """Build the per-recipe blends artifact for the test split.

    Empty recipes need no special case: a zero feature row's logits are the
    intercepts, so its blend is the calibrated prior automatically.

    Args:
        blend_matrix: Calibrated probabilities (recipe x cuisine).
        recipe_ids: Recipe ids aligned with the matrix rows.
        cuisine_ids: Cuisine identifiers defining the blend column order.
        fingerprint: Model build block embedded in the artifact.

    Returns:
        Payload with rows sorted by recipe id; blends rounded to
        BLEND_DECIMALS, top_cuisine from full precision (lowest index wins
        ties).
    """
    blend_matrix = numpy.asarray(blend_matrix, dtype=float)
    top_positions = blend_matrix.argmax(axis=1)
    rows = [
        {
            "blend": [
                round(float(value), BLEND_DECIMALS) for value in blend_matrix[position]
            ],
            "recipe_id": recipe_id,
            "top_cuisine": cuisine_ids[top_positions[position]],
        }
        for position, recipe_id in enumerate(recipe_ids)
    ]
    rows.sort(key=lambda row: row["recipe_id"])
    return {
        "build": dict(fingerprint),
        "cuisines": list(cuisine_ids),
        "rows": rows,
        "schema_version": SCHEMA_VERSION,
    }
