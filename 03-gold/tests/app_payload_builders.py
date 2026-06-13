"""Shared builders for the synthetic inputs the app-exporter tests use.

Reuses the model-test corpus (6 features, 1 parent link, 3 cuisines) and
adds hand-built parameters/calibration/evaluation/ingredients payloads so
no exporter test needs scikit-learn or real artifacts.
"""

from tests.model_payload_builders import (
    MODEL_BUILD_BLOCK,
    make_cuisines_payload,
    make_feature_space_payload,
)

APP_BUILD_BLOCK = {
    "cuisines_sha256": "1" * 64,
    "evaluation_sha256": "2" * 64,
    "feature_space_sha256": "3" * 64,
    "ingredients_sha256": "4" * 64,
    "calibration_sha256": "5" * 64,
    "parameters_sha256": "6" * 64,
}

CUISINE_IDS = ("italian", "mexican", "thai")


def make_parameters_payload() -> dict:
    """Hand-built 3-cuisine x 6-feature parameters (thai loves 1/2/5)."""
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "schema_version": 1,
        "model": "multinomial_logistic_regression",
        "configuration": {
            "c_value": 1.0,
            "max_iterations": 1000,
            "parent_weight": 1.0,
            "tolerance": 0.0001,
        },
        "selection": {"metric": "pooled_out_of_fold_log_loss", "value": 0.2},
        "cuisines": list(CUISINE_IDS),
        "intercepts": [0.25, -0.1, -0.15],
        "coefficients": [
            [2.123456, -1.0, -1.0, 2.0, -0.5, -0.654321],
            [-1.0, -1.0, -1.0, -1.0, 3.0, -0.5],
            [-2.0, 3.111111, 3.0, -2.0, -0.25, 1.0],
        ],
    }


def make_calibration_payload() -> dict:
    """Hand-built calibration with reliability bins."""
    bins = [
        {"accuracy": 0.0, "bin": bin_number, "count": 0, "mean_confidence": 0.0}
        for bin_number in range(10)
    ]
    bins[7] = {"accuracy": 0.75, "bin": 7, "count": 20, "mean_confidence": 0.76}
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "schema_version": 1,
        "temperature": 1.05,
        "ece_bin_count": 10,
        "optimizer": {
            "iterations": 100,
            "log10_range": [-2.0, 2.0],
            "method": "golden_section",
        },
        "out_of_fold": {
            "ece_after": 0.02,
            "ece_before": 0.05,
            "log_loss_after": 0.5,
            "log_loss_before": 0.52,
        },
        "reliability": {"after": bins, "before": bins},
    }


def make_evaluation_payload() -> dict:
    """Hand-built evaluation matching the 3-cuisine corpus."""
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "schema_version": 1,
        "configuration": {"c_value": 1.0, "parent_weight": 1.0},
        "grid_search": [
            {"c_value": 1.0, "parent_weight": 1.0, "pooled_oof_log_loss": 0.2}
        ],
        "folds": [
            {"fold": 0, "accuracy": 0.9, "macro_f1": 0.85, "log_loss": 0.3},
            {"fold": 1, "accuracy": 0.8, "macro_f1": 0.75, "log_loss": 0.5},
        ],
        "mean": {"accuracy": 0.85, "macro_f1": 0.8, "log_loss": 0.4},
        "calibration": {"ece_after": 0.02, "ece_before": 0.05, "temperature": 1.05},
        "per_cuisine": [
            {"cuisine": "italian", "recall": 0.9, "recipe_count": 4},
            {"cuisine": "mexican", "recall": 0.7, "recipe_count": 4},
            {"cuisine": "thai", "recall": 0.8, "recipe_count": 4},
        ],
        "confusion_pairs": [
            {
                "count": 2,
                "neighbor_similarity": 0.45,
                "predicted_cuisine": "mexican",
                "true_cuisine": "italian",
            }
        ],
        "baseline_naive_bayes": {
            "mean_accuracy": 0.7,
            "mean_macro_f1": 0.65,
            "pooled_oof_log_loss": 0.6,
        },
    }


def make_silver_ingredients_payload() -> dict:
    """Silver-shaped ingredients matching the 6-feature corpus.

    Feature order (sorted): basil(0), dark_soy_sauce(1 -> soy_sauce),
    fish_sauce(2), pasta(3), rice(4), soy_sauce(5).
    """
    entries = [
        ("basil", None, 30, [("fresh basil", 12)]),
        ("dark_soy_sauce", "soy_sauce", 25, [("dark soy", 5)]),
        ("fish_sauce", None, 40, []),
        ("pasta", None, 50, [("penne pasta", 9)]),
        ("rice", None, 60, []),
        ("soy_sauce", None, 70, [("kikkoman soy sauce", 8)]),
    ]
    ingredients = []
    for ingredient_id, parent_id, mention_count, alias_rows in entries:
        evidence = (
            {
                "jsd_bits": 0.49,
                "layer": "statistical_gate",
                "null95_bits": 0.09,
                "variant_count": 25,
            }
            if parent_id is not None
            else None
        )
        ingredients.append(
            {
                "aliases": [
                    {
                        "alias": ingredient_id.replace("_", " "),
                        "rule": None,
                        "source": "canonical_surface_form",
                        "train_frequency": mention_count,
                    }
                ]
                + [
                    {
                        "alias": alias,
                        "rule": None,
                        "source": "mechanical_normalization",
                        "train_frequency": frequency,
                    }
                    for alias, frequency in alias_rows
                ],
                "category": None,
                "id": ingredient_id,
                "name": ingredient_id.replace("_", " "),
                "parent_id": parent_id,
                "preserve_evidence": evidence,
                "train_mention_count": mention_count,
            }
        )
    return {
        "build": dict(MODEL_BUILD_BLOCK),
        "ingredients": ingredients,
        "schema_version": 1,
    }


def make_app_inputs() -> dict:
    """Bundle every synthetic exporter input keyed by name."""
    return {
        "parameters": make_parameters_payload(),
        "calibration": make_calibration_payload(),
        "feature_space": make_feature_space_payload(),
        "evaluation": make_evaluation_payload(),
        "ingredients": make_silver_ingredients_payload(),
        "cuisines": make_cuisines_payload(),
    }
