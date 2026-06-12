"""Rebuild every model artifact from the gold datasets.

Usage:
    .venv/bin/python 03-gold/build.py                    # full rebuild (minutes:
                                                         # the grid refits 60 models)
    .venv/bin/python 03-gold/build.py --check-idempotent # rebuild in memory,
                                                         # verify disk matches

Deterministic given the environment: the same gold inputs under the same
scikit-learn version produce byte-identical artifacts (coefficients are
rounded before any downstream artifact is derived from them).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# model_pipeline reads gold data via gold_pipeline.locations and shares
# silver_pipeline's artifact I/O, so both earlier tier roots must be
# importable before the model imports below. Python already puts this
# script's own directory (03-gold) on sys.path for model_pipeline itself.
TIER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TIER_ROOT.parent / "02-silver"))
sys.path.insert(0, str(TIER_ROOT.parent / "01-bronze"))

import numpy

from model_pipeline import locations
from model_pipeline.assemble_matrices import (
    assemble_design_matrix,
    encode_cuisine_labels,
)
from model_pipeline.build_submission import (
    SUBMISSION_FILENAME,
    build_submission_rows,
    render_submission_csv,
)
from model_pipeline.calibrate_blend import (
    BLENDS_FILENAME,
    CALIBRATION_FILENAME,
    METRIC_DECIMALS,
    build_blends_payload,
    build_calibration_payload,
    compute_logits_from_parameters,
    convert_logits_to_blend,
    fit_temperature,
)
from model_pipeline.evaluate_model import (
    EVALUATION_JSON_FILENAME,
    EVALUATION_MARKDOWN_FILENAME,
    build_evaluation_payload,
    compute_fold_metrics,
    render_evaluation_markdown,
    summarize_confusion_pairs,
    write_evaluation_reports,
)
from model_pipeline.load_gold_datasets import (
    compute_model_build_fingerprint,
    load_gold_model_inputs,
)
from model_pipeline.train_model import (
    C_VALUE_GRID,
    PARAMETERS_FILENAME,
    PARENT_WEIGHT_GRID,
    build_parameters_payload,
    evaluate_configuration_over_folds,
    fit_naive_bayes_baseline,
    select_winning_configuration,
    train_final_model,
)
from model_pipeline.validate_model import validate_model_artifacts
from silver_pipeline.artifact_io import (
    find_artifact_mismatches,
    serialize_artifact_json,
    write_artifact_json,
    write_text_atomically,
)

PARAMETERS_PATH = locations.GOLD_MODEL_DIRECTORY / PARAMETERS_FILENAME
CALIBRATION_PATH = locations.GOLD_MODEL_DIRECTORY / CALIBRATION_FILENAME
BLENDS_PATH = locations.GOLD_MODEL_DIRECTORY / BLENDS_FILENAME
EVALUATION_JSON_PATH = locations.GOLD_REPORTS_DIRECTORY / EVALUATION_JSON_FILENAME
EVALUATION_MARKDOWN_PATH = (
    locations.GOLD_REPORTS_DIRECTORY / EVALUATION_MARKDOWN_FILENAME
)
SUBMISSION_PATH = locations.GOLD_SUBMISSION_DIRECTORY / SUBMISSION_FILENAME

PROGRESS_LOG_FORMAT = "%(message)s"

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Every payload one full model build produces."""

    parameters: dict
    calibration: dict
    blends_test: dict
    evaluation: dict
    submission_csv: str


def _summarize_per_cuisine_recall(
    label_indices, predicted_indices, cuisine_ids
) -> list[dict]:
    """Per-cuisine recall over the pooled out-of-fold predictions."""
    per_cuisine = []
    for position, cuisine in enumerate(cuisine_ids):
        inside = label_indices == position
        recipe_count = int(inside.sum())
        correct_count = int((predicted_indices[inside] == position).sum())
        recall = correct_count / recipe_count if recipe_count else 0.0
        per_cuisine.append(
            {
                "cuisine": cuisine,
                "recall": round(recall, METRIC_DECIMALS),
                "recipe_count": recipe_count,
            }
        )
    return per_cuisine


def build_model_artifacts() -> ModelArtifacts:
    """Run the full model build in memory: tune, fit, calibrate, evaluate.

    Returns:
        ModelArtifacts with every payload, already validated by the model
        gates against the gold inputs on disk.

    Raises:
        ValidationError: If any model gate fails.
        FileNotFoundError: If a gold or silver input file is missing.
    """
    fingerprint = compute_model_build_fingerprint()
    inputs = load_gold_model_inputs()
    cuisine_ids = sorted(cuisine["id"] for cuisine in inputs.cuisines["cuisines"])
    feature_count = inputs.feature_space["feature_count"]
    train_rows = inputs.features_train["rows"]
    test_rows = inputs.features_test["rows"]
    logger.info(
        "loaded gold inputs: %d features, %d train / %d test rows",
        feature_count,
        len(train_rows),
        len(test_rows),
    )

    grid_results = []
    for parent_weight in PARENT_WEIGHT_GRID:
        for c_value in C_VALUE_GRID:
            result = evaluate_configuration_over_folds(
                train_rows,
                inputs.folds,
                feature_count,
                cuisine_ids,
                c_value=c_value,
                parent_weight=parent_weight,
            )
            logger.info(
                "grid C=%.1f w=%.1f -> pooled OOF log loss %.4f",
                c_value,
                parent_weight,
                result.pooled_oof_log_loss,
            )
            grid_results.append(result)
    winner = select_winning_configuration(grid_results)
    logger.info(
        "winner: C=%.1f w=%.1f (pooled OOF log loss %.4f)",
        winner.c_value,
        winner.parent_weight,
        winner.pooled_oof_log_loss,
    )

    label_indices = encode_cuisine_labels(train_rows, cuisine_ids)
    temperature = fit_temperature(winner.oof_logits, label_indices)
    calibration = build_calibration_payload(
        temperature, winner.oof_logits, label_indices, fingerprint
    )
    logger.info(
        "temperature %.4f: OOF log loss %.4f -> %.4f, ECE %.4f -> %.4f",
        temperature,
        calibration["out_of_fold"]["log_loss_before"],
        calibration["out_of_fold"]["log_loss_after"],
        calibration["out_of_fold"]["ece_before"],
        calibration["out_of_fold"]["ece_after"],
    )

    final_model = train_final_model(train_rows, feature_count, cuisine_ids, winner)
    parameters = build_parameters_payload(
        final_model, winner, cuisine_ids, feature_count, fingerprint
    )

    test_matrix = assemble_design_matrix(
        test_rows, feature_count, winner.parent_weight
    )
    test_logits = compute_logits_from_parameters(
        test_matrix, parameters["coefficients"], parameters["intercepts"]
    )
    test_blend = convert_logits_to_blend(test_logits, temperature)
    blends_test = build_blends_payload(
        test_blend,
        recipe_ids=[row["recipe_id"] for row in test_rows],
        cuisine_ids=cuisine_ids,
        fingerprint=fingerprint,
    )
    submission_csv = render_submission_csv(build_submission_rows(blends_test))

    oof_blend = convert_logits_to_blend(winner.oof_logits, temperature)
    oof_predictions = oof_blend.argmax(axis=1)
    positions_by_fold: dict[int, list[int]] = {}
    fold_by_recipe_id = {
        assignment["recipe_id"]: assignment["fold"]
        for assignment in inputs.folds["assignments"]
    }
    for position, row in enumerate(train_rows):
        positions_by_fold.setdefault(
            fold_by_recipe_id[row["recipe_id"]], []
        ).append(position)
    fold_metrics = []
    for fold in sorted(positions_by_fold):
        held_out = numpy.array(positions_by_fold[fold])
        metrics = compute_fold_metrics(
            label_indices[held_out], oof_blend[held_out], cuisine_ids
        )
        fold_metrics.append({"fold": fold, **metrics})

    baseline = fit_naive_bayes_baseline(
        train_rows, inputs.folds, feature_count, cuisine_ids, winner.parent_weight
    )
    logger.info(
        "baseline NB: pooled OOF log loss %.4f (LR winner %.4f)",
        baseline["pooled_oof_log_loss"],
        winner.pooled_oof_log_loss,
    )

    evaluation = build_evaluation_payload(
        grid_results=[
            {
                "c_value": result.c_value,
                "parent_weight": result.parent_weight,
                "pooled_oof_log_loss": round(
                    result.pooled_oof_log_loss, METRIC_DECIMALS
                ),
            }
            for result in grid_results
        ],
        configuration={
            "c_value": winner.c_value,
            "parent_weight": winner.parent_weight,
        },
        fold_metrics=fold_metrics,
        calibration_summary={
            "ece_after": calibration["out_of_fold"]["ece_after"],
            "ece_before": calibration["out_of_fold"]["ece_before"],
            "temperature": temperature,
        },
        per_cuisine_recall=_summarize_per_cuisine_recall(
            label_indices, oof_predictions, cuisine_ids
        ),
        confusion_pairs=summarize_confusion_pairs(
            [cuisine_ids[index] for index in label_indices],
            [cuisine_ids[index] for index in oof_predictions],
            inputs.cuisines,
        ),
        baseline_naive_bayes=baseline,
        fingerprint=fingerprint,
    )

    validate_model_artifacts(
        parameters,
        calibration,
        blends_test,
        evaluation,
        submission_csv,
        inputs.feature_space,
        inputs.features_test,
        inputs.cuisines,
        expected_fingerprint=fingerprint,
    )
    logger.info("validation gates: PASS")

    return ModelArtifacts(
        parameters=parameters,
        calibration=calibration,
        blends_test=blends_test,
        evaluation=evaluation,
        submission_csv=submission_csv,
    )


def write_model_artifacts(artifacts: ModelArtifacts) -> None:
    """Persist every artifact atomically to model/, reports/, submission/."""
    locations.GOLD_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    locations.GOLD_REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    locations.GOLD_SUBMISSION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_artifact_json(artifacts.parameters, PARAMETERS_PATH)
    write_artifact_json(artifacts.calibration, CALIBRATION_PATH)
    write_artifact_json(artifacts.blends_test, BLENDS_PATH)
    write_evaluation_reports(artifacts.evaluation, locations.GOLD_REPORTS_DIRECTORY)
    write_text_atomically(artifacts.submission_csv, SUBMISSION_PATH)
    logger.info(
        "wrote model artifacts to %s, %s, and %s",
        locations.GOLD_MODEL_DIRECTORY,
        locations.GOLD_REPORTS_DIRECTORY,
        locations.GOLD_SUBMISSION_DIRECTORY,
    )


def verify_rebuild_matches_disk(artifacts: ModelArtifacts) -> list[str]:
    """Compare freshly built payloads against the model files on disk.

    Args:
        artifacts: Payloads from build_model_artifacts.

    Returns:
        Names of artifact files whose on-disk bytes differ from the rebuild
        (empty when the build is idempotent under this environment).
    """
    return find_artifact_mismatches(
        {
            PARAMETERS_PATH: serialize_artifact_json(artifacts.parameters),
            CALIBRATION_PATH: serialize_artifact_json(artifacts.calibration),
            BLENDS_PATH: serialize_artifact_json(artifacts.blends_test),
            EVALUATION_JSON_PATH: serialize_artifact_json(artifacts.evaluation),
            EVALUATION_MARKDOWN_PATH: render_evaluation_markdown(
                artifacts.evaluation
            ),
            SUBMISSION_PATH: artifacts.submission_csv,
        }
    )


def main() -> int:
    """Entry point: rebuild model artifacts or verify idempotency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-idempotent",
        action="store_true",
        help="rebuild in memory and verify the model files on disk match",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=PROGRESS_LOG_FORMAT)

    artifacts = build_model_artifacts()
    if not arguments.check_idempotent:
        write_model_artifacts(artifacts)
        return 0

    mismatches = verify_rebuild_matches_disk(artifacts)
    if mismatches:
        print(f"IDEMPOTENCY FAILURE: {', '.join(sorted(mismatches))}")
        return 1
    print("idempotency check: PASS (rebuild matches disk byte-for-byte)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
