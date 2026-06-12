"""Tests for model_pipeline.train_model.

All fits run on the 12-row synthetic corpus — milliseconds per fit.
"""

import numpy

from model_pipeline.assemble_matrices import (
    assemble_design_matrix,
    encode_cuisine_labels,
)
from model_pipeline.train_model import (
    COEFFICIENT_DECIMALS,
    ConfigurationResult,
    build_parameters_payload,
    evaluate_configuration_over_folds,
    fit_logistic_model,
    select_winning_configuration,
)
from tests.model_payload_builders import (
    MODEL_BUILD_BLOCK,
    make_folds_payload,
    make_train_feature_rows,
)

SYNTHETIC_FEATURE_COUNT = 6
CUISINE_IDS = ("italian", "mexican", "thai")


def _train_matrix_and_labels(parent_weight=0.3):
    rows = make_train_feature_rows()
    matrix = assemble_design_matrix(rows, SYNTHETIC_FEATURE_COUNT, parent_weight)
    labels = encode_cuisine_labels(rows, CUISINE_IDS)
    return rows, matrix, labels


def test_fit_logistic_model_is_deterministic_across_calls():
    _, matrix, labels = _train_matrix_and_labels()

    first = fit_logistic_model(matrix, labels, c_value=1.0)
    second = fit_logistic_model(matrix, labels, c_value=1.0)

    assert numpy.array_equal(first.coef_, second.coef_)
    assert numpy.array_equal(first.intercept_, second.intercept_)


def test_evaluate_configuration_returns_pooled_loss_and_oof_logits():
    rows = make_train_feature_rows()
    folds_payload = make_folds_payload(rows)

    result = evaluate_configuration_over_folds(
        rows,
        folds_payload,
        SYNTHETIC_FEATURE_COUNT,
        CUISINE_IDS,
        c_value=1.0,
        parent_weight=0.3,
    )

    assert result.c_value == 1.0
    assert result.parent_weight == 0.3
    assert result.pooled_oof_log_loss > 0.0
    assert result.oof_logits.shape == (len(rows), len(CUISINE_IDS))


def test_select_winning_configuration_prefers_lowest_loss_then_lexicographic():
    empty_logits = numpy.zeros((1, 3))
    results = [
        ConfigurationResult(3.0, 1.0, 0.50, empty_logits),
        ConfigurationResult(0.1, 0.3, 0.40, empty_logits),
        ConfigurationResult(0.1, 0.0, 0.40, empty_logits),
        ConfigurationResult(1.0, 0.0, 0.45, empty_logits),
    ]

    winner = select_winning_configuration(results)

    assert (winner.c_value, winner.parent_weight) == (0.1, 0.0)


def test_parameters_payload_rounds_coefficients_to_six_decimals():
    _, matrix, labels = _train_matrix_and_labels()
    model = fit_logistic_model(matrix, labels, c_value=1.0)
    winner = ConfigurationResult(1.0, 0.3, 0.2, numpy.zeros((1, 3)))

    payload = build_parameters_payload(
        model, winner, CUISINE_IDS, SYNTHETIC_FEATURE_COUNT, MODEL_BUILD_BLOCK
    )

    assert payload["cuisines"] == list(CUISINE_IDS)
    assert len(payload["coefficients"]) == len(CUISINE_IDS)
    assert all(
        len(cuisine_row) == SYNTHETIC_FEATURE_COUNT
        for cuisine_row in payload["coefficients"]
    )
    assert all(
        value == round(value, COEFFICIENT_DECIMALS)
        for cuisine_row in payload["coefficients"]
        for value in cuisine_row
    )
    assert payload["configuration"]["c_value"] == 1.0
    assert payload["configuration"]["parent_weight"] == 0.3
