"""Validation gates over the model artifacts.

Pure validators over already-loaded payloads — no scikit-learn, no file
I/O. Each gate failure raises ValidationError naming the gate and the
offending values. The umbrella validate_model_artifacts additionally
proves all artifacts share one build block and that the block matches the
fingerprint freshly computed from the inputs on disk (which includes the
installed scikit-learn version — a version drift fails here, loudly,
instead of masquerading as a determinism bug).
"""

from __future__ import annotations

from model_pipeline.build_submission import SUBMISSION_HEADER
from model_pipeline.calibrate_blend import TEMPERATURE_LOG10_RANGE
from model_pipeline.train_model import (
    C_VALUE_GRID,
    COEFFICIENT_DECIMALS,
    MODEL_NAME,
    PARENT_WEIGHT_GRID,
)
from silver_pipeline.artifact_io import SCHEMA_VERSION

EXPECTED_FEATURE_COUNT = 2_813
EXPECTED_TEST_FEATURE_ROW_COUNT = 9_944
BLEND_SUM_TOLERANCE = 0.002
LOG_LOSS_TOLERANCE = 1e-6
MEAN_METRIC_TOLERANCE = 1e-6

REQUIRED_MODEL_FINGERPRINT_KEYS = frozenset(
    (
        "cuisines_sha256",
        "feature_space_sha256",
        "features_test_sha256",
        "features_train_sha256",
        "folds_sha256",
        "random_seed",
        "sklearn_version",
    )
)


class ValidationError(ValueError):
    """A model artifact violates the pinned schema or a consistency gate."""


def _require_envelope(payload: dict, artifact_name: str) -> None:
    """Gate: schema_version and a complete build block."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"{artifact_name}: schema_version must be {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    build = payload.get("build")
    if not isinstance(build, dict) or set(build) != REQUIRED_MODEL_FINGERPRINT_KEYS:
        raise ValidationError(
            f"{artifact_name}: build block must hold exactly "
            f"{sorted(REQUIRED_MODEL_FINGERPRINT_KEYS)}"
        )


def validate_parameters_payload(
    payload: dict, cuisines_payload: dict, expected_feature_count: int
) -> None:
    """Gate the parameters artifact: shape, rounding, and pinned config.

    Args:
        payload: model/parameters.json content.
        cuisines_payload: Parsed silver cuisines.json document.
        expected_feature_count: Pinned feature-space width.

    Raises:
        ValidationError: On any schema, shape, rounding, cuisine-set, or
            configuration violation.
    """
    _require_envelope(payload, "parameters")
    if payload["model"] != MODEL_NAME:
        raise ValidationError(
            f"parameters: model must be {MODEL_NAME!r}, got {payload['model']!r}"
        )

    expected_cuisines = sorted(
        cuisine["id"] for cuisine in cuisines_payload["cuisines"]
    )
    if payload["cuisines"] != expected_cuisines:
        raise ValidationError(
            "parameters: cuisines must equal the sorted taxonomy ids"
        )

    coefficients = payload["coefficients"]
    if len(coefficients) != len(expected_cuisines):
        raise ValidationError(
            f"parameters: {len(coefficients)} coefficients rows for "
            f"{len(expected_cuisines)} cuisines"
        )
    for cuisine, cuisine_row in zip(payload["cuisines"], coefficients):
        if len(cuisine_row) != expected_feature_count:
            raise ValidationError(
                f"parameters: coefficients row for {cuisine} has "
                f"{len(cuisine_row)} columns, expected {expected_feature_count}"
            )
    if len(payload["intercepts"]) != len(expected_cuisines):
        raise ValidationError(
            f"parameters: {len(payload['intercepts'])} intercepts for "
            f"{len(expected_cuisines)} cuisines"
        )
    for value in payload["intercepts"]:
        _require_rounded_finite(value, "parameters: intercepts")
    for cuisine_row in coefficients:
        for value in cuisine_row:
            _require_rounded_finite(value, "parameters: coefficients")

    configuration = payload["configuration"]
    if configuration["c_value"] not in C_VALUE_GRID:
        raise ValidationError(
            f"parameters: configuration c_value {configuration['c_value']} "
            f"is outside the pinned grid {C_VALUE_GRID}"
        )
    if configuration["parent_weight"] not in PARENT_WEIGHT_GRID:
        raise ValidationError(
            "parameters: configuration parent_weight "
            f"{configuration['parent_weight']} is outside the pinned grid "
            f"{PARENT_WEIGHT_GRID}"
        )


def _require_rounded_finite(value: float, context: str) -> None:
    """Gate: a parameter value is finite and 6-decimal round-trippable."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValidationError(f"{context}: non-finite value {value!r}")
    if value != round(value, COEFFICIENT_DECIMALS):
        raise ValidationError(
            f"{context}: value {value!r} is not rounded to "
            f"{COEFFICIENT_DECIMALS} decimals"
        )


def validate_calibration_payload(payload: dict) -> None:
    """Gate the calibration artifact: temperature range and honesty.

    Args:
        payload: model/calibration.json content.

    Raises:
        ValidationError: If the temperature is out of range or calibration
            made the out-of-fold log loss worse.
    """
    _require_envelope(payload, "calibration")
    temperature = payload["temperature"]
    low, high = TEMPERATURE_LOG10_RANGE
    if not 10.0**low <= temperature <= 10.0**high:
        raise ValidationError(
            f"calibration: temperature {temperature} outside "
            f"[{10.0 ** low}, {10.0 ** high}]"
        )
    out_of_fold = payload["out_of_fold"]
    if (
        out_of_fold["log_loss_after"]
        > out_of_fold["log_loss_before"] + LOG_LOSS_TOLERANCE
    ):
        raise ValidationError(
            "calibration: log_loss_after "
            f"{out_of_fold['log_loss_after']} exceeds log_loss_before "
            f"{out_of_fold['log_loss_before']}"
        )


def validate_blends_payload(
    payload: dict,
    parameters_payload: dict,
    features_test_payload: dict,
    expected_row_count: int,
) -> None:
    """Gate the blends artifact against the test split and parameters.

    Args:
        payload: model/blends_test.json content.
        parameters_payload: model/parameters.json content.
        features_test_payload: Gold features_test.json content.
        expected_row_count: Pinned test recipe count.

    Raises:
        ValidationError: On any schema, count, id-coverage, range, sum, or
            argmax violation.
    """
    _require_envelope(payload, "blends")
    if payload["cuisines"] != parameters_payload["cuisines"]:
        raise ValidationError(
            "blends: cuisine column order differs from the parameters artifact"
        )
    rows = payload["rows"]
    if len(rows) != expected_row_count:
        raise ValidationError(
            f"blends: expected {expected_row_count} rows, got {len(rows)}"
        )
    expected_ids = [row["recipe_id"] for row in features_test_payload["rows"]]
    blend_ids = [row["recipe_id"] for row in rows]
    if blend_ids != expected_ids:
        raise ValidationError(
            "blends: row recipe ids do not match the gold test split"
        )

    cuisine_position = {
        cuisine: position
        for position, cuisine in enumerate(payload["cuisines"])
    }
    cuisine_count = len(payload["cuisines"])
    for row in rows:
        blend = row["blend"]
        if len(blend) != cuisine_count:
            raise ValidationError(
                f"blends: recipe {row['recipe_id']} blend has {len(blend)} "
                f"entries, expected {cuisine_count}"
            )
        if any(not 0.0 <= value <= 1.0 for value in blend):
            raise ValidationError(
                f"blends: recipe {row['recipe_id']} has probabilities "
                "outside [0, 1]"
            )
        blend_total = sum(blend)
        if abs(blend_total - 1.0) > BLEND_SUM_TOLERANCE:
            raise ValidationError(
                f"blends: recipe {row['recipe_id']} blend sum {blend_total} "
                f"deviates from 1 beyond {BLEND_SUM_TOLERANCE}"
            )
        if blend[cuisine_position[row["top_cuisine"]]] != max(blend):
            raise ValidationError(
                f"blends: recipe {row['recipe_id']} top_cuisine "
                f"{row['top_cuisine']} does not carry the maximum share"
            )


def validate_submission_csv(csv_text: str, blends_payload: dict) -> None:
    """Gate the submission text against the blends artifact.

    Args:
        csv_text: Full submission.csv content.
        blends_payload: model/blends_test.json content.

    Raises:
        ValidationError: On header, row-count, id, or argmax disagreement.
    """
    lines = csv_text.splitlines()
    if not lines or lines[0] != SUBMISSION_HEADER:
        raise ValidationError(
            f"submission: header must be {SUBMISSION_HEADER!r}"
        )
    blend_rows = blends_payload["rows"]
    data_lines = lines[1:]
    if len(data_lines) != len(blend_rows):
        raise ValidationError(
            f"submission: {len(data_lines)} rows for {len(blend_rows)} blends"
        )
    for line, blend_row in zip(data_lines, blend_rows):
        expected_line = f"{blend_row['recipe_id']},{blend_row['top_cuisine']}"
        if line != expected_line:
            raise ValidationError(
                f"submission: line {line!r} disagrees with the blend's "
                f"{expected_line!r}"
            )


def validate_evaluation_payload(
    payload: dict, parameters_payload: dict, cuisines_payload: dict
) -> None:
    """Gate the evaluation artifact's internal consistency.

    Args:
        payload: reports/evaluation.json content.
        parameters_payload: model/parameters.json content.
        cuisines_payload: Parsed silver cuisines.json document.

    Raises:
        ValidationError: On configuration mismatch, fold irregularities,
            mean drift, missing cuisines, grid inconsistencies, or bad
            confusion annotations.
    """
    _require_envelope(payload, "evaluation")
    configuration = payload["configuration"]
    parameters_configuration = parameters_payload["configuration"]
    if (
        configuration["c_value"] != parameters_configuration["c_value"]
        or configuration["parent_weight"]
        != parameters_configuration["parent_weight"]
    ):
        raise ValidationError(
            "evaluation: configuration disagrees with the parameters artifact"
        )

    folds = payload["folds"]
    fold_numbers = [fold["fold"] for fold in folds]
    if not folds or len(fold_numbers) != len(set(fold_numbers)):
        raise ValidationError(
            f"evaluation: fold numbers must be unique and non-empty, "
            f"got {fold_numbers}"
        )
    for metric_name, reported_mean in payload["mean"].items():
        recomputed = sum(fold[metric_name] for fold in folds) / len(folds)
        if abs(reported_mean - recomputed) > MEAN_METRIC_TOLERANCE:
            raise ValidationError(
                f"evaluation: mean {metric_name} {reported_mean} != "
                f"recomputed {round(recomputed, 6)}"
            )

    known_cuisines = {cuisine["id"] for cuisine in cuisines_payload["cuisines"]}
    recall_cuisines = {entry["cuisine"] for entry in payload["per_cuisine"]}
    if recall_cuisines != known_cuisines:
        raise ValidationError(
            "evaluation: per_cuisine coverage does not match the taxonomy"
        )

    for entry in payload["grid_search"]:
        if (
            entry["c_value"] not in C_VALUE_GRID
            or entry["parent_weight"] not in PARENT_WEIGHT_GRID
        ):
            raise ValidationError(
                f"evaluation: grid entry ({entry['c_value']}, "
                f"{entry['parent_weight']}) is outside the pinned grids"
            )
    if payload["grid_search"]:
        best_entry = min(
            payload["grid_search"],
            key=lambda entry: (
                entry["pooled_oof_log_loss"],
                entry["c_value"],
                entry["parent_weight"],
            ),
        )
        if (
            best_entry["c_value"] != configuration["c_value"]
            or best_entry["parent_weight"] != configuration["parent_weight"]
        ):
            raise ValidationError(
                "evaluation: configuration is not the grid_search argmin"
            )

    similarity_by_pair = {
        (cuisine["id"], neighbor["id"]): neighbor["similarity"]
        for cuisine in cuisines_payload["cuisines"]
        for neighbor in cuisine["neighbors"]
    }
    for pair in payload["confusion_pairs"]:
        if pair["true_cuisine"] == pair["predicted_cuisine"]:
            raise ValidationError(
                f"evaluation: confusion pair repeats {pair['true_cuisine']}"
            )
        if not {pair["true_cuisine"], pair["predicted_cuisine"]} <= known_cuisines:
            raise ValidationError(
                f"evaluation: confusion pair holds unknown cuisines: {pair}"
            )
        expected_similarity = similarity_by_pair.get(
            (pair["true_cuisine"], pair["predicted_cuisine"])
        )
        if pair["neighbor_similarity"] != expected_similarity:
            raise ValidationError(
                f"evaluation: confusion pair {pair['true_cuisine']} -> "
                f"{pair['predicted_cuisine']} similarity "
                f"{pair['neighbor_similarity']} != taxonomy "
                f"{expected_similarity}"
            )


def _require_shared_fresh_build_blocks(
    payload_by_name: dict[str, dict], expected_fingerprint: dict
) -> None:
    """Gate: every artifact carries the same, fresh build block."""
    blocks = list(
        {name: payload.get("build") for name, payload in payload_by_name.items()}.values()
    )
    if any(block != blocks[0] for block in blocks[1:]):
        raise ValidationError(
            "model artifacts carry differing build blocks: "
            f"{sorted(payload_by_name)}"
        )
    if blocks[0] != expected_fingerprint:
        raise ValidationError(
            "model build fingerprint is stale: artifacts were built from a "
            "different input state or scikit-learn version than the current "
            "environment"
        )


def validate_model_artifacts(
    parameters_payload: dict,
    calibration_payload: dict,
    blends_payload: dict,
    evaluation_payload: dict,
    submission_csv_text: str,
    feature_space_payload: dict,
    features_test_payload: dict,
    cuisines_payload: dict,
    *,
    expected_fingerprint: dict,
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
    expected_test_row_count: int = EXPECTED_TEST_FEATURE_ROW_COUNT,
) -> None:
    """Run every model gate plus the cross-artifact checks.

    Args:
        parameters_payload: model/parameters.json content.
        calibration_payload: model/calibration.json content.
        blends_payload: model/blends_test.json content.
        evaluation_payload: reports/evaluation.json content.
        submission_csv_text: Full submission.csv content.
        feature_space_payload: Gold feature_space.json content.
        features_test_payload: Gold features_test.json content.
        cuisines_payload: Parsed silver cuisines.json document.
        expected_fingerprint: Fingerprint freshly computed from the inputs
            on disk and the installed environment.
        expected_feature_count: Pinned feature-space width.
        expected_test_row_count: Pinned test recipe count.

    Raises:
        ValidationError: If any gate fails.
    """
    _require_shared_fresh_build_blocks(
        {
            "parameters": parameters_payload,
            "calibration": calibration_payload,
            "blends": blends_payload,
            "evaluation": evaluation_payload,
        },
        expected_fingerprint,
    )
    if feature_space_payload["feature_count"] != expected_feature_count:
        raise ValidationError(
            f"feature space on disk has {feature_space_payload['feature_count']} "
            f"features, expected {expected_feature_count}"
        )
    validate_parameters_payload(
        parameters_payload, cuisines_payload, expected_feature_count
    )
    validate_calibration_payload(calibration_payload)
    validate_blends_payload(
        blends_payload,
        parameters_payload,
        features_test_payload,
        expected_test_row_count,
    )
    validate_submission_csv(submission_csv_text, blends_payload)
    validate_evaluation_payload(
        evaluation_payload, parameters_payload, cuisines_payload
    )
    if (
        calibration_payload["temperature"]
        != evaluation_payload["calibration"]["temperature"]
    ):
        raise ValidationError(
            "calibration temperature disagrees between the calibration and "
            "evaluation artifacts"
        )
