"""Tests for model_pipeline.validate_model.

Builds a coherent synthetic artifact set through the real builder modules,
then corrupts one field at a time and asserts the matching gate fires.
"""

import copy

import numpy
import pytest

from model_pipeline.assemble_matrices import (
    assemble_design_matrix,
    encode_cuisine_labels,
)
from model_pipeline.build_submission import (
    build_submission_rows,
    render_submission_csv,
)
from model_pipeline.calibrate_blend import (
    build_blends_payload,
    build_calibration_payload,
    compute_logits_from_parameters,
    convert_logits_to_blend,
    fit_temperature,
)
from model_pipeline.evaluate_model import build_evaluation_payload
from model_pipeline.train_model import (
    ConfigurationResult,
    build_parameters_payload,
    fit_logistic_model,
)
from model_pipeline.validate_model import (
    ValidationError,
    validate_blends_payload,
    validate_model_artifacts,
    validate_parameters_payload,
    validate_submission_csv,
)
from tests.model_payload_builders import (
    MODEL_BUILD_BLOCK,
    make_cuisines_payload,
    make_feature_space_payload,
    make_features_payload,
    make_test_feature_rows,
    make_train_feature_rows,
)

SYNTHETIC_FEATURE_COUNT = 6
SYNTHETIC_TEST_ROW_COUNT = 3
CUISINE_IDS = ("italian", "mexican", "thai")
SYNTHETIC_CONFIGURATION = ConfigurationResult(
    c_value=1.0, parent_weight=0.3, pooled_oof_log_loss=0.2,
    oof_logits=numpy.zeros((1, 3)),
)


def _build_valid_artifacts() -> dict:
    """Train on the synthetic corpus and derive every model artifact."""
    train_rows = make_train_feature_rows()
    test_rows = make_test_feature_rows()
    train_matrix = assemble_design_matrix(
        train_rows, SYNTHETIC_FEATURE_COUNT, SYNTHETIC_CONFIGURATION.parent_weight
    )
    labels = encode_cuisine_labels(train_rows, CUISINE_IDS)
    model = fit_logistic_model(
        train_matrix, labels, SYNTHETIC_CONFIGURATION.c_value
    )
    parameters = build_parameters_payload(
        model,
        SYNTHETIC_CONFIGURATION,
        CUISINE_IDS,
        SYNTHETIC_FEATURE_COUNT,
        MODEL_BUILD_BLOCK,
    )

    train_logits = compute_logits_from_parameters(
        train_matrix.toarray(), parameters["coefficients"], parameters["intercepts"]
    )
    temperature = fit_temperature(train_logits, labels)
    calibration = build_calibration_payload(
        temperature, train_logits, labels, MODEL_BUILD_BLOCK
    )

    test_matrix = assemble_design_matrix(
        test_rows, SYNTHETIC_FEATURE_COUNT, SYNTHETIC_CONFIGURATION.parent_weight
    )
    test_logits = compute_logits_from_parameters(
        test_matrix.toarray(), parameters["coefficients"], parameters["intercepts"]
    )
    blend = convert_logits_to_blend(test_logits, temperature)
    blends = build_blends_payload(
        blend,
        recipe_ids=[row["recipe_id"] for row in test_rows],
        cuisine_ids=CUISINE_IDS,
        fingerprint=MODEL_BUILD_BLOCK,
    )
    evaluation = build_evaluation_payload(
        grid_results=[
            {"c_value": 1.0, "parent_weight": 0.3, "pooled_oof_log_loss": 0.2}
        ],
        configuration=dict(parameters["configuration"]),
        fold_metrics=[
            {"fold": 0, "accuracy": 1.0, "macro_f1": 1.0, "log_loss": 0.1},
            {"fold": 1, "accuracy": 1.0, "macro_f1": 1.0, "log_loss": 0.1},
        ],
        calibration_summary={
            "temperature": calibration["temperature"],
            "ece_before": calibration["out_of_fold"]["ece_before"],
            "ece_after": calibration["out_of_fold"]["ece_after"],
        },
        per_cuisine_recall=[
            {"cuisine": cuisine, "recipe_count": 4, "recall": 1.0}
            for cuisine in CUISINE_IDS
        ],
        confusion_pairs=[],
        baseline_naive_bayes={
            "pooled_oof_log_loss": 0.5,
            "mean_accuracy": 0.9,
            "mean_macro_f1": 0.9,
        },
        fingerprint=MODEL_BUILD_BLOCK,
    )
    submission_csv = render_submission_csv(build_submission_rows(blends))
    return {
        "parameters": parameters,
        "calibration": calibration,
        "blends": blends,
        "evaluation": evaluation,
        "submission_csv": submission_csv,
        "feature_space": make_feature_space_payload(),
        "features_test": make_features_payload(test_rows),
        "cuisines": make_cuisines_payload(),
    }


def _validate(artifacts: dict, expected_fingerprint: dict | None = None) -> None:
    validate_model_artifacts(
        artifacts["parameters"],
        artifacts["calibration"],
        artifacts["blends"],
        artifacts["evaluation"],
        artifacts["submission_csv"],
        artifacts["feature_space"],
        artifacts["features_test"],
        artifacts["cuisines"],
        expected_fingerprint=expected_fingerprint or MODEL_BUILD_BLOCK,
        expected_feature_count=SYNTHETIC_FEATURE_COUNT,
        expected_test_row_count=SYNTHETIC_TEST_ROW_COUNT,
    )


def test_valid_model_artifacts_pass():
    _validate(_build_valid_artifacts())


def test_missing_fingerprint_key_raises():
    artifacts = _build_valid_artifacts()
    artifacts["parameters"] = copy.deepcopy(artifacts["parameters"])
    del artifacts["parameters"]["build"]["sklearn_version"]

    with pytest.raises(ValidationError, match="build"):
        _validate(artifacts)


def test_sklearn_version_drift_raises():
    artifacts = _build_valid_artifacts()
    drifted_fingerprint = dict(MODEL_BUILD_BLOCK, sklearn_version="0.0.1")

    with pytest.raises(ValidationError, match="fingerprint"):
        _validate(artifacts, expected_fingerprint=drifted_fingerprint)


def test_wrong_coefficient_shape_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["parameters"])
    corrupted["coefficients"][0] = corrupted["coefficients"][0][:-1]

    with pytest.raises(ValidationError, match="coefficients"):
        validate_parameters_payload(
            corrupted, artifacts["cuisines"], SYNTHETIC_FEATURE_COUNT
        )


def test_configuration_outside_pinned_grid_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["parameters"])
    corrupted["configuration"]["c_value"] = 99.0

    with pytest.raises(ValidationError, match="configuration"):
        validate_parameters_payload(
            corrupted, artifacts["cuisines"], SYNTHETIC_FEATURE_COUNT
        )


def test_blend_not_summing_to_one_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["blends"])
    # Halving keeps every entry inside [0, 1] so the sum gate is the one
    # that fires, not the range gate.
    corrupted["rows"][0]["blend"] = [
        value * 0.5 for value in corrupted["rows"][0]["blend"]
    ]

    with pytest.raises(ValidationError, match="sum"):
        validate_blends_payload(
            corrupted,
            artifacts["parameters"],
            artifacts["features_test"],
            SYNTHETIC_TEST_ROW_COUNT,
        )


def test_blends_missing_test_recipe_raises():
    artifacts = _build_valid_artifacts()
    corrupted = copy.deepcopy(artifacts["blends"])
    corrupted["rows"] = corrupted["rows"][:-1]

    with pytest.raises(ValidationError, match="row"):
        validate_blends_payload(
            corrupted,
            artifacts["parameters"],
            artifacts["features_test"],
            SYNTHETIC_TEST_ROW_COUNT,
        )


def test_submission_cuisine_disagreeing_with_argmax_raises():
    artifacts = _build_valid_artifacts()
    lines = artifacts["submission_csv"].splitlines()
    recipe_id = lines[1].split(",")[0]
    lines[1] = f"{recipe_id},mexican" if not lines[1].endswith(
        ",mexican"
    ) else f"{recipe_id},thai"
    corrupted_csv = "\n".join(lines) + "\n"

    with pytest.raises(ValidationError, match="submission"):
        validate_submission_csv(corrupted_csv, artifacts["blends"])
