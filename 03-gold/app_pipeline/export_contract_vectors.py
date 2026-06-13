"""Build the JS↔Python scoring contract vectors.

Each vector is a fixed recipe scored through the pure scorer against the
trimmed 4-decimal model asset, recording the exact blend, top cuisines,
and explanations the browser must reproduce. Vectors marked example: true
double as the Blend Builder's example chips.
"""

from __future__ import annotations

from model_pipeline.explain_predictions import compute_ingredient_contributions

from app_pipeline.score_blend import (
    build_feature_values,
    compute_logits,
    convert_logits_to_blend,
)

CONTRACT_VECTORS_ASSET_FILENAME = "contract-vectors.json"
BLEND_DECIMALS = 4
EXPLANATION_DECIMALS = 6
EXPLANATION_LIMIT = 5
INTENTIONALLY_UNKNOWN_PREFIX = "not_a_real"

CONTRACT_RECIPES = (
    ("empty recipe falls back to base rates", True, ()),
    ("single strong signal", True, ("tortillas",)),
    ("parent back-off fires", False, ("asian_pear",)),
    ("variant and parent together", False, ("dark_soy_sauce", "soy_sauce")),
    (
        "unknown ids are ignored",
        False,
        ("soy_sauce", "not_a_real_ingredient"),
    ),
    ("duplicate ids are deduped", False, ("garlic", "garlic")),
    (
        "the thai demo",
        True,
        ("fish_sauce", "coconut_milk", "thai_basil", "lime_juice", "rice_noodles"),
    ),
    (
        "twenty-ingredient fusion",
        False,
        (
            "soy_sauce",
            "olive_oil",
            "garlic",
            "onions",
            "tortillas",
            "parmesan_cheese",
            "fish_sauce",
            "garam_masala",
            "buttermilk",
            "ginger",
            "lime_juice",
            "basil",
            "cumin",
            "feta_cheese",
            "coconut_milk",
            "sour_cream",
            "sesame_oil",
            "paprika",
            "cilantro",
            "butter",
        ),
    ),
    ("classic italian", True, ("spaghetti", "olive_oil", "garlic", "basil")),
    (
        "classic indian",
        True,
        ("garam_masala", "ground_turmeric", "ginger", "onions", "cumin"),
    ),
    (
        "classic chinese",
        True,
        ("soy_sauce", "ginger", "sesame_oil", "scallions"),
    ),
    (
        "classic southern us",
        True,
        ("buttermilk", "all_purpose_flour", "butter", "cayenne_pepper"),
    ),
)


def _require_known_ids(
    name: str, ingredient_ids: tuple[str, ...], feature_ids: list[str]
) -> None:
    """Fail fast when a vector references an id the model does not know."""
    known_ids = set(feature_ids)
    for ingredient_id in ingredient_ids:
        if ingredient_id.startswith(INTENTIONALLY_UNKNOWN_PREFIX):
            continue
        if ingredient_id not in known_ids:
            raise ValueError(
                f"contract vector {name!r} references unknown ingredient "
                f"{ingredient_id!r}"
            )


def _rank_cuisines(blend: list[float], cuisine_ids: list[str]) -> list[str]:
    """Order cuisines by descending share, ids breaking ties."""
    return [
        cuisine
        for _, cuisine in sorted(
            zip(blend, cuisine_ids), key=lambda pair: (-pair[0], pair[1])
        )
    ]


def _summarize_explanations(
    feature_values: dict[int, float],
    model_asset: dict,
    top_cuisine: str,
    runner_up_cuisine: str,
) -> tuple[list[dict], list[dict]]:
    """Top contributions toward the leader and differentiators vs runner-up."""
    cuisine_position = {
        cuisine: position
        for position, cuisine in enumerate(model_asset["cuisines"])
    }
    top_coefficients = model_asset["coefficients"][cuisine_position[top_cuisine]]
    runner_up_coefficients = model_asset["coefficients"][
        cuisine_position[runner_up_cuisine]
    ]
    feature_ids = model_asset["feature_ids"]

    top_contributions = [
        {
            "contribution": round(contribution, EXPLANATION_DECIMALS),
            "ingredient_id": feature_ids[index],
        }
        for index, contribution in compute_ingredient_contributions(
            feature_values, top_coefficients
        )[:EXPLANATION_LIMIT]
    ]
    advantages = [
        (
            index,
            value * (top_coefficients[index] - runner_up_coefficients[index]),
        )
        for index, value in feature_values.items()
    ]
    advantages.sort(key=lambda pair: (-pair[1], pair[0]))
    differentiators = [
        {
            "advantage": round(advantage, EXPLANATION_DECIMALS),
            "ingredient_id": feature_ids[index],
        }
        for index, advantage in advantages[:EXPLANATION_LIMIT]
    ]
    return top_contributions, differentiators


def build_contract_vectors_asset(
    model_asset: dict,
    fingerprint: dict,
    recipes: tuple = CONTRACT_RECIPES,
) -> dict:
    """Score every contract recipe and record the expected outputs.

    Args:
        model_asset: The trimmed scoring asset from build_model_asset.
        fingerprint: App build block embedded in the asset.
        recipes: (name, is_example, ingredient_ids) tuples; defaults to
            the pinned production set.

    Returns:
        Asset with one vector per recipe: expected blend at 4 decimals,
        top cuisines, and explanations at 6 decimals.

    Raises:
        ValueError: If a recipe references an unknown ingredient id that
            is not intentionally unknown.
    """
    vectors = []
    for name, is_example, ingredient_ids in recipes:
        _require_known_ids(name, ingredient_ids, model_asset["feature_ids"])
        feature_values = build_feature_values(
            ingredient_ids,
            model_asset["feature_ids"],
            model_asset["parent_indices"],
            model_asset["parent_weight"],
        )
        logits = compute_logits(
            feature_values, model_asset["coefficients"], model_asset["intercepts"]
        )
        blend = convert_logits_to_blend(logits, model_asset["temperature"])
        ranked_cuisines = _rank_cuisines(blend, model_asset["cuisines"])
        top_contributions, differentiators = _summarize_explanations(
            feature_values, model_asset, ranked_cuisines[0], ranked_cuisines[1]
        )
        vectors.append(
            {
                "name": name,
                "example": is_example,
                "ingredient_ids": list(ingredient_ids),
                "expected_blend": [
                    round(value, BLEND_DECIMALS) for value in blend
                ],
                "expected_top_cuisine": ranked_cuisines[0],
                "expected_runner_up": ranked_cuisines[1],
                "expected_top_contributions": top_contributions,
                "expected_differentiators": differentiators,
            }
        )
    return {
        "build": dict(fingerprint),
        "schema_version": 1,
        "vectors": vectors,
    }
