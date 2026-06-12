"""Train the multinomial logistic regression and tune it on the gold folds.

Hyperparameters (inverse regularization strength and the parent back-off
weight) are selected by pooled out-of-fold log loss over the gold 5-fold
assignment — the blend-honesty metric, not accuracy. A MultinomialNB
baseline is fit on the same matrices solely for the evaluation report.
Only this module imports scikit-learn estimators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from model_pipeline.assemble_matrices import (
    assemble_design_matrix,
    encode_cuisine_labels,
    split_row_positions_by_fold,
)
from model_pipeline.calibrate_blend import (
    LIKELIHOOD_FLOOR,
    METRIC_DECIMALS,
    compute_negative_log_likelihood,
)
from silver_pipeline.artifact_io import BUILD_RANDOM_SEED, SCHEMA_VERSION

MODEL_NAME = "multinomial_logistic_regression"
C_VALUE_GRID = (0.1, 0.3, 1.0, 3.0)
PARENT_WEIGHT_GRID = (0.0, 0.3, 1.0)
LBFGS_MAX_ITERATIONS = 1000
LBFGS_TOLERANCE = 1e-4
NAIVE_BAYES_ALPHA = 1.0
COEFFICIENT_DECIMALS = 6
SELECTION_METRIC_NAME = "pooled_out_of_fold_log_loss"

PARAMETERS_FILENAME = "parameters.json"


@dataclass(frozen=True)
class ConfigurationResult:
    """One hyperparameter configuration's cross-validation outcome.

    Attributes:
        c_value: Inverse L2 regularization strength.
        parent_weight: Back-off weight given to parent features.
        pooled_oof_log_loss: Log loss over all out-of-fold predictions.
        oof_logits: Pooled out-of-fold logits in original row order.
    """

    c_value: float
    parent_weight: float
    pooled_oof_log_loss: float
    oof_logits: numpy.ndarray


def fit_logistic_model(
    matrix, label_indices, c_value: float
) -> LogisticRegression:
    """Fit one multinomial logistic regression with pinned settings.

    Args:
        matrix: Sparse design matrix (recipe x feature).
        label_indices: Encoded cuisine labels aligned with the matrix.
        c_value: Inverse L2 regularization strength.

    Returns:
        The fitted estimator (multinomial by default for lbfgs multiclass).
    """
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=LBFGS_MAX_ITERATIONS,
        tol=LBFGS_TOLERANCE,
        random_state=BUILD_RANDOM_SEED,
    )
    model.fit(matrix, label_indices)
    return model


def _training_positions_excluding_fold(
    positions_by_fold: dict[int, list[int]], held_out_fold: int
) -> numpy.ndarray:
    """Concatenate every fold's row positions except the held-out fold."""
    training_positions = [
        position
        for fold, positions in sorted(positions_by_fold.items())
        if fold != held_out_fold
        for position in positions
    ]
    return numpy.array(training_positions)


def evaluate_configuration_over_folds(
    rows: Sequence[dict],
    folds_payload: dict,
    feature_count: int,
    cuisine_ids: Sequence[str],
    c_value: float,
    parent_weight: float,
) -> ConfigurationResult:
    """Cross-validate one configuration and pool its out-of-fold logits.

    Args:
        rows: Labeled gold feature rows.
        folds_payload: Gold folds artifact.
        feature_count: Width of the feature space.
        cuisine_ids: Sorted cuisine identifiers (label order).
        c_value: Inverse L2 regularization strength to evaluate.
        parent_weight: Parent back-off weight to evaluate.

    Returns:
        ConfigurationResult with the pooled out-of-fold log loss and the
        out-of-fold logits in original row order.
    """
    matrix = assemble_design_matrix(rows, feature_count, parent_weight)
    label_indices = encode_cuisine_labels(rows, cuisine_ids)
    positions_by_fold = split_row_positions_by_fold(rows, folds_payload)
    oof_logits = numpy.zeros((len(rows), len(cuisine_ids)))
    for fold in sorted(positions_by_fold):
        held_out = numpy.array(positions_by_fold[fold])
        training = _training_positions_excluding_fold(positions_by_fold, fold)
        model = fit_logistic_model(matrix[training], label_indices[training], c_value)
        oof_logits[held_out] = model.decision_function(matrix[held_out])
    pooled_log_loss = compute_negative_log_likelihood(
        oof_logits, label_indices, temperature=1.0
    )
    return ConfigurationResult(
        c_value=c_value,
        parent_weight=parent_weight,
        pooled_oof_log_loss=pooled_log_loss,
        oof_logits=oof_logits,
    )


def select_winning_configuration(
    results: Sequence[ConfigurationResult],
) -> ConfigurationResult:
    """Pick the configuration with the lowest pooled out-of-fold log loss.

    Ties break toward the lexicographically smallest (c_value,
    parent_weight) so selection is deterministic.

    Args:
        results: One ConfigurationResult per grid configuration.

    Returns:
        The winning ConfigurationResult.
    """
    return min(
        results,
        key=lambda result: (
            result.pooled_oof_log_loss,
            result.c_value,
            result.parent_weight,
        ),
    )


def train_final_model(
    rows: Sequence[dict],
    feature_count: int,
    cuisine_ids: Sequence[str],
    configuration: ConfigurationResult,
) -> LogisticRegression:
    """Fit the winning configuration on every labeled row.

    Args:
        rows: All labeled gold feature rows.
        feature_count: Width of the feature space.
        cuisine_ids: Sorted cuisine identifiers (label order).
        configuration: The grid winner from select_winning_configuration.

    Returns:
        The final fitted estimator.
    """
    matrix = assemble_design_matrix(
        rows, feature_count, configuration.parent_weight
    )
    label_indices = encode_cuisine_labels(rows, cuisine_ids)
    return fit_logistic_model(matrix, label_indices, configuration.c_value)


def fit_naive_bayes_baseline(
    rows: Sequence[dict],
    folds_payload: dict,
    feature_count: int,
    cuisine_ids: Sequence[str],
    parent_weight: float,
) -> dict:
    """Cross-validate the MultinomialNB baseline on the same matrices.

    Args:
        rows: Labeled gold feature rows.
        folds_payload: Gold folds artifact.
        feature_count: Width of the feature space.
        cuisine_ids: Sorted cuisine identifiers (label order).
        parent_weight: The winning parent back-off weight (fair comparison).

    Returns:
        Baseline metrics: pooled_oof_log_loss, mean_accuracy,
        mean_macro_f1 — rounded for the evaluation report.
    """
    matrix = assemble_design_matrix(rows, feature_count, parent_weight)
    label_indices = encode_cuisine_labels(rows, cuisine_ids)
    positions_by_fold = split_row_positions_by_fold(rows, folds_payload)
    oof_probabilities = numpy.zeros((len(rows), len(cuisine_ids)))
    accuracies: list[float] = []
    macro_f1_scores: list[float] = []
    for fold in sorted(positions_by_fold):
        held_out = numpy.array(positions_by_fold[fold])
        training = _training_positions_excluding_fold(positions_by_fold, fold)
        model = MultinomialNB(alpha=NAIVE_BAYES_ALPHA)
        model.fit(matrix[training], label_indices[training])
        fold_probabilities = model.predict_proba(matrix[held_out])
        oof_probabilities[held_out] = fold_probabilities
        fold_predictions = fold_probabilities.argmax(axis=1)
        accuracies.append(
            float(accuracy_score(label_indices[held_out], fold_predictions))
        )
        macro_f1_scores.append(
            float(
                f1_score(
                    label_indices[held_out],
                    fold_predictions,
                    average="macro",
                    labels=range(len(cuisine_ids)),
                    zero_division=0,
                )
            )
        )
    row_positions = numpy.arange(len(label_indices))
    likelihoods = numpy.clip(
        oof_probabilities[row_positions, label_indices], LIKELIHOOD_FLOOR, None
    )
    pooled_log_loss = float(-numpy.log(likelihoods).mean())
    return {
        "mean_accuracy": round(
            sum(accuracies) / len(accuracies), METRIC_DECIMALS
        ),
        "mean_macro_f1": round(
            sum(macro_f1_scores) / len(macro_f1_scores), METRIC_DECIMALS
        ),
        "pooled_oof_log_loss": round(pooled_log_loss, METRIC_DECIMALS),
    }


def build_parameters_payload(
    model: LogisticRegression,
    configuration: ConfigurationResult,
    cuisine_ids: Sequence[str],
    feature_count: int,
    fingerprint: dict,
) -> dict:
    """Serialize the fitted model with rounded, reproducible parameters.

    Every downstream artifact (blends, submission, CLI output) is derived
    from these rounded values, so the file fully determines the model.

    Args:
        model: The final fitted estimator.
        configuration: The winning grid configuration.
        cuisine_ids: Sorted cuisine identifiers (coefficient row order).
        feature_count: Width of the feature space (coefficient columns).
        fingerprint: Model build block embedded in the artifact.

    Returns:
        Payload ready for write_artifact_json.

    Raises:
        ValueError: If the fitted model's shape disagrees with the
            cuisine list or feature count.
    """
    if model.coef_.shape != (len(cuisine_ids), feature_count):
        raise ValueError(
            f"fitted coefficients shape {model.coef_.shape} != "
            f"({len(cuisine_ids)}, {feature_count})"
        )
    coefficients = [
        [round(float(value), COEFFICIENT_DECIMALS) for value in cuisine_row]
        for cuisine_row in model.coef_
    ]
    intercepts = [
        round(float(value), COEFFICIENT_DECIMALS) for value in model.intercept_
    ]
    return {
        "build": dict(fingerprint),
        "coefficients": coefficients,
        "configuration": {
            "c_value": configuration.c_value,
            "max_iterations": LBFGS_MAX_ITERATIONS,
            "parent_weight": configuration.parent_weight,
            "tolerance": LBFGS_TOLERANCE,
        },
        "cuisines": list(cuisine_ids),
        "intercepts": intercepts,
        "model": MODEL_NAME,
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "metric": SELECTION_METRIC_NAME,
            "value": round(configuration.pooled_oof_log_loss, METRIC_DECIMALS),
        },
    }
