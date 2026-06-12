"""Tests for model_pipeline.calibrate_blend."""

import numpy

from model_pipeline.calibrate_blend import (
    build_blends_payload,
    compute_expected_calibration_error,
    compute_logits_from_parameters,
    compute_negative_log_likelihood,
    convert_logits_to_blend,
    fit_temperature,
)
from tests.model_payload_builders import MODEL_BUILD_BLOCK

CUISINE_IDS = ("italian", "mexican", "thai")


def test_logits_match_manual_dot_product():
    feature_matrix = numpy.array([[1.0, 0.5]])
    coefficients = [[2.0, -1.0], [0.0, 4.0]]
    intercepts = [0.5, -0.5]

    logits = compute_logits_from_parameters(feature_matrix, coefficients, intercepts)

    assert logits[0].tolist() == [2.0 - 0.5 + 0.5, 2.0 - 0.5]


def test_blend_rows_sum_to_one():
    logits = numpy.array([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0]])

    blend = convert_logits_to_blend(logits, temperature=1.5)

    assert numpy.allclose(blend.sum(axis=1), 1.0)
    assert blend[1].tolist() == [1.0 / 3.0] * 3


def test_fit_temperature_exceeds_one_for_overconfident_logits():
    generator = numpy.random.default_rng(7)
    calibrated_logits = generator.normal(size=(400, 3))
    labels = numpy.array(
        [
            generator.choice(3, p=row)
            for row in convert_logits_to_blend(calibrated_logits, 1.0)
        ]
    )
    overconfident_logits = calibrated_logits * 4.0

    temperature = fit_temperature(overconfident_logits, labels)

    assert temperature > 1.0


def test_fit_temperature_is_deterministic_and_optimal():
    generator = numpy.random.default_rng(11)
    logits = generator.normal(size=(200, 3)) * 3.0
    labels = generator.integers(0, 3, size=200)

    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)

    assert first == second
    fitted_loss = compute_negative_log_likelihood(logits, labels, first)
    assert fitted_loss <= compute_negative_log_likelihood(logits, labels, 1.0)
    assert fitted_loss <= compute_negative_log_likelihood(logits, labels, 4.0)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    confident_row = [0.75, 0.25]
    probabilities = numpy.array([confident_row] * 20)
    labels = numpy.array([0] * 15 + [1] * 5)

    error = compute_expected_calibration_error(probabilities, labels)

    assert error == 0.0


def test_blends_payload_rounds_and_carries_top_cuisine():
    logits = numpy.array([[4.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    blend = convert_logits_to_blend(logits, temperature=1.0)

    payload = build_blends_payload(
        blend, recipe_ids=[100, 102], cuisine_ids=CUISINE_IDS,
        fingerprint=MODEL_BUILD_BLOCK,
    )

    first_row = payload["rows"][0]
    assert first_row["recipe_id"] == 100
    assert first_row["top_cuisine"] == "italian"
    assert len(first_row["blend"]) == 3
    assert all(
        value == round(value, 4) for row in payload["rows"] for value in row["blend"]
    )
    assert payload["cuisines"] == list(CUISINE_IDS)
    assert payload["rows"][1]["top_cuisine"] == "italian"
