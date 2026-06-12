"""Explain blend predictions from the model's own arithmetic.

Logistic regression is linear, so an ingredient's contribution to a
cuisine's logit is exactly its feature value times that cuisine's
coefficient — explanations here are the model's computation itself, not a
post-hoc approximation. Pure logic over the rounded parameters payload; no
scikit-learn involved.
"""

from __future__ import annotations

from collections.abc import Sequence

from model_pipeline.calibrate_blend import METRIC_DECIMALS

DEFAULT_EXPLANATION_LIMIT = 5


def compute_ingredient_contributions(
    feature_values: dict[int, float], cuisine_coefficients: Sequence[float]
) -> list[tuple[int, float]]:
    """Rank each present feature's contribution to one cuisine's logit.

    Args:
        feature_values: Feature index -> value for the recipe's active
            features (direct ingredients and parent back-off).
        cuisine_coefficients: The cuisine's coefficient row.

    Returns:
        (feature_index, contribution) pairs sorted by contribution
        descending (feature index breaks ties).
    """
    contributions = [
        (index, value * cuisine_coefficients[index])
        for index, value in feature_values.items()
    ]
    contributions.sort(key=lambda pair: (-pair[1], pair[0]))
    return contributions


def summarize_blend_explanation(
    feature_values: dict[int, float],
    parameters_payload: dict,
    feature_space_payload: dict,
    top_cuisine: str,
    runner_up_cuisine: str,
    limit: int = DEFAULT_EXPLANATION_LIMIT,
) -> dict:
    """Explain why a recipe scored its top cuisine over the runner-up.

    Args:
        feature_values: Feature index -> value for the recipe.
        parameters_payload: Rounded model parameters artifact.
        feature_space_payload: Gold feature space (index -> ingredient id).
        top_cuisine: The blend's leading cuisine.
        runner_up_cuisine: The blend's second cuisine.
        limit: Entries to keep in each list.

    Returns:
        {"top_contributions": [{ingredient_id, contribution}],
         "differentiators": [{ingredient_id, advantage}]} where advantage
        is the contribution gap between top cuisine and runner-up.
    """
    cuisine_position = {
        cuisine: position
        for position, cuisine in enumerate(parameters_payload["cuisines"])
    }
    ingredient_by_index = {
        feature["index"]: feature["ingredient_id"]
        for feature in feature_space_payload["features"]
    }
    top_coefficients = parameters_payload["coefficients"][
        cuisine_position[top_cuisine]
    ]
    runner_up_coefficients = parameters_payload["coefficients"][
        cuisine_position[runner_up_cuisine]
    ]

    top_contributions = [
        {
            "contribution": round(contribution, METRIC_DECIMALS),
            "ingredient_id": ingredient_by_index[index],
        }
        for index, contribution in compute_ingredient_contributions(
            feature_values, top_coefficients
        )[:limit]
    ]

    advantages = [
        (index, value * (top_coefficients[index] - runner_up_coefficients[index]))
        for index, value in feature_values.items()
    ]
    advantages.sort(key=lambda pair: (-pair[1], pair[0]))
    differentiators = [
        {
            "advantage": round(advantage, METRIC_DECIMALS),
            "ingredient_id": ingredient_by_index[index],
        }
        for index, advantage in advantages[:limit]
    ]
    return {
        "differentiators": differentiators,
        "top_contributions": top_contributions,
    }
