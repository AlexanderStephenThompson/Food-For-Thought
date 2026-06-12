"""Predict a calibrated cuisine blend for raw ingredient strings.

The CLI runs each raw string through the silver tier's resolution chain
(exact alias, cleaning, modifier stripping, brand resolution, token drop),
scores the resolved ingredients with the rounded model parameters, and
prints the calibrated blend plus the model's own per-ingredient
explanation. One command exercises the entire pipeline end to end.

Usage (via the launcher at the tier root):
    .venv/bin/python 03-gold/predict.py \\
        --ingredients "fish sauce, coconut milk, thai basil"
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy

from model_pipeline import locations
from model_pipeline.calibrate_blend import (
    CALIBRATION_FILENAME,
    compute_logits_from_parameters,
    convert_logits_to_blend,
)
from model_pipeline.explain_predictions import summarize_blend_explanation
from model_pipeline.load_gold_datasets import FEATURE_SPACE_FILENAME
from model_pipeline.train_model import PARAMETERS_FILENAME
from silver_pipeline.artifact_io import ARTIFACT_TEXT_ENCODING
from silver_pipeline.resolve_ingredient import (
    IngredientResolver,
    ResolutionResult,
)

DEFAULT_TOP_COUNT = 5
DEFAULT_EXPLANATION_LIMIT = 5
UNRESOLVED_LABEL = "UNRESOLVED"
DIRECT_PRESENCE_VALUE = 1.0
SILVER_VOCABULARY_FILENAME = "ingredients.json"


def split_raw_ingredient_arguments(raw_text: str) -> list[str]:
    """Split the --ingredients argument into trimmed, non-empty strings.

    Args:
        raw_text: Comma-separated raw ingredient list from the CLI.

    Returns:
        Cleaned raw ingredient strings, in input order.
    """
    return [part.strip() for part in raw_text.split(",") if part.strip()]


def build_feature_values(
    resolved_ids: Sequence[str],
    feature_space_payload: dict,
    parent_weight: float,
) -> dict[int, float]:
    """Map resolved ingredient ids to feature values with parent back-off.

    Applies the same max semantics as the training matrices: direct
    presence is 1.0 and always wins; a parent only contributes
    parent_weight when it is not itself present.

    Args:
        resolved_ids: Canonical ingredient ids from the resolver.
        feature_space_payload: Gold feature_space.json content.
        parent_weight: The trained configuration's back-off weight.

    Returns:
        Feature index -> value for every active feature.
    """
    index_by_id = {
        feature["ingredient_id"]: feature["index"]
        for feature in feature_space_payload["features"]
    }
    parent_index_by_index = {
        feature["index"]: feature["parent_index"]
        for feature in feature_space_payload["features"]
    }
    direct_indices = {
        index_by_id[ingredient_id]
        for ingredient_id in resolved_ids
        if ingredient_id in index_by_id
    }
    feature_values = {
        index: DIRECT_PRESENCE_VALUE for index in direct_indices
    }
    if parent_weight <= 0.0:
        return feature_values
    for index in sorted(direct_indices):
        parent_index = parent_index_by_index[index]
        if parent_index is not None and parent_index not in direct_indices:
            feature_values[parent_index] = parent_weight
    return feature_values


def format_resolution_table(
    resolutions: Sequence[tuple[str, ResolutionResult]],
) -> list[str]:
    """Render one line per raw string showing how it resolved.

    Args:
        resolutions: (raw string, ResolutionResult) pairs in input order.

    Returns:
        Display lines; unresolved strings are marked UNRESOLVED.
    """
    lines = []
    for raw_text, result in resolutions:
        resolved_label = (
            result.ingredient_id
            if result.ingredient_id is not None
            else UNRESOLVED_LABEL
        )
        line = f"  {raw_text!r} -> {resolved_label} [{result.method}]"
        if result.dropped_tokens:
            line += f" (dropped: {', '.join(result.dropped_tokens)})"
        lines.append(line)
    return lines


def format_blend_lines(
    blend_by_cuisine: dict[str, float], top_count: int = DEFAULT_TOP_COUNT
) -> list[str]:
    """Render the top blend shares as percentage lines.

    Args:
        blend_by_cuisine: Cuisine -> calibrated probability.
        top_count: Cuisines to show, by descending share.

    Returns:
        Lines like "58.3%  thai" (cuisine name breaks ties).
    """
    ranked = sorted(
        blend_by_cuisine.items(), key=lambda pair: (-pair[1], pair[0])
    )
    return [
        f"{share * 100:.1f}%  {cuisine}" for cuisine, share in ranked[:top_count]
    ]


def _read_json_document(path: Path) -> dict:
    """Parse one artifact file, failing fast with the path in context."""
    try:
        return json.loads(path.read_text(encoding=ARTIFACT_TEXT_ENCODING))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one blend prediction."""
    parser = argparse.ArgumentParser(
        prog=".venv/bin/python 03-gold/predict.py",
        description=(
            "Predict a calibrated cuisine blend, with per-ingredient "
            "explanations, for a comma-separated raw ingredient list."
        ),
    )
    parser.add_argument(
        "--ingredients",
        required=True,
        help='comma-separated raw ingredients, e.g. "fish sauce, coconut milk"',
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_COUNT)
    parser.add_argument(
        "--explanations", type=int, default=DEFAULT_EXPLANATION_LIMIT
    )
    return parser


def main() -> int:
    """Entry point: resolve, score, calibrate, and explain one recipe."""
    arguments = _build_argument_parser().parse_args()
    raw_ingredients = split_raw_ingredient_arguments(arguments.ingredients)
    if not raw_ingredients:
        print("no ingredients given")
        return 1

    parameters = _read_json_document(
        locations.GOLD_MODEL_DIRECTORY / PARAMETERS_FILENAME
    )
    calibration = _read_json_document(
        locations.GOLD_MODEL_DIRECTORY / CALIBRATION_FILENAME
    )
    feature_space = _read_json_document(
        locations.GOLD_DATASETS_DIRECTORY / FEATURE_SPACE_FILENAME
    )
    resolver = IngredientResolver.from_paths(
        locations.SILVER_DATASETS_DIRECTORY / SILVER_VOCABULARY_FILENAME,
        locations.LEXICONS_DIRECTORY,
    )

    resolutions = [(raw, resolver.resolve(raw)) for raw in raw_ingredients]
    print("resolution:")
    for line in format_resolution_table(resolutions):
        print(line)
    resolved_ids = [
        result.ingredient_id
        for _, result in resolutions
        if result.ingredient_id is not None
    ]
    unresolved_count = len(raw_ingredients) - len(resolved_ids)
    if unresolved_count:
        print(f"warning: {unresolved_count} ingredient(s) did not resolve")
    if not resolved_ids:
        print("nothing resolved; cannot predict a blend")
        return 1

    feature_values = build_feature_values(
        resolved_ids,
        feature_space,
        parameters["configuration"]["parent_weight"],
    )
    feature_vector = numpy.zeros((1, feature_space["feature_count"]))
    for index, value in feature_values.items():
        feature_vector[0, index] = value
    logits = compute_logits_from_parameters(
        feature_vector, parameters["coefficients"], parameters["intercepts"]
    )
    blend = convert_logits_to_blend(logits, calibration["temperature"])[0]
    blend_by_cuisine = dict(zip(parameters["cuisines"], blend.tolist()))

    print("\nblend:")
    for line in format_blend_lines(blend_by_cuisine, arguments.top):
        print(f"  {line}")

    ranked_cuisines = sorted(
        blend_by_cuisine, key=lambda cuisine: (-blend_by_cuisine[cuisine], cuisine)
    )
    explanation = summarize_blend_explanation(
        feature_values,
        parameters,
        feature_space,
        top_cuisine=ranked_cuisines[0],
        runner_up_cuisine=ranked_cuisines[1],
        limit=arguments.explanations,
    )
    print(f"\ntop contributions toward {ranked_cuisines[0]}:")
    for entry in explanation["top_contributions"]:
        print(f"  {entry['contribution']:+.3f}  {entry['ingredient_id']}")
    print(
        f"\nwhat separates {ranked_cuisines[0]} from {ranked_cuisines[1]}:"
    )
    for entry in explanation["differentiators"]:
        print(f"  {entry['advantage']:+.3f}  {entry['ingredient_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
