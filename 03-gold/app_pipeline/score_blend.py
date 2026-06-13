"""Pure-Python blend scorer mirroring the browser's JavaScript scorer.

Both sides perform the same IEEE-754 operations in the same order —
ascending-feature-index summation for logits, left-to-right summation for
the softmax normalizer — so contract vectors computed here are reproduced
by the JavaScript exactly, not approximately. The numpy reference in
model_pipeline.calibrate_blend may reorder summation (BLAS); this module
must not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DIRECT_PRESENCE_VALUE = 1.0


def build_feature_values(
    resolved_ids: Sequence[str],
    feature_ids: Sequence[str],
    parent_indices: Sequence[int | None],
    parent_weight: float,
) -> dict[int, float]:
    """Map ingredient ids to feature values with max-semantics back-off.

    Unknown ids are ignored and duplicates collapse; a parent only
    contributes parent_weight when it is not itself present.

    Args:
        resolved_ids: Canonical ingredient ids (may repeat or be unknown).
        feature_ids: The model asset's ingredient id per feature index.
        parent_indices: The model asset's parent index (or None) per index.
        parent_weight: The trained configuration's back-off weight.

    Returns:
        Feature index -> value for every active feature.
    """
    index_by_id = {
        ingredient_id: index for index, ingredient_id in enumerate(feature_ids)
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
        parent_index = parent_indices[index]
        if parent_index is not None and parent_index not in direct_indices:
            feature_values[parent_index] = parent_weight
    return feature_values


def compute_logits(
    feature_values: dict[int, float],
    coefficients: Sequence[Sequence[float]],
    intercepts: Sequence[float],
) -> list[float]:
    """Compute one recipe's per-cuisine logits.

    Summation runs in ascending feature-index order — this ordering is the
    cross-language contract and must match the JavaScript scorer.

    Args:
        feature_values: Feature index -> value for the recipe.
        coefficients: Per-cuisine coefficient rows.
        intercepts: Per-cuisine intercepts.

    Returns:
        One logit per cuisine, in coefficient row order.
    """
    ordered_indices = sorted(feature_values)
    logits = []
    for cuisine_row, intercept in zip(coefficients, intercepts):
        total = intercept
        for index in ordered_indices:
            total += feature_values[index] * cuisine_row[index]
        logits.append(total)
    return logits


def convert_logits_to_blend(
    logits: Sequence[float], temperature: float
) -> list[float]:
    """Convert logits to a calibrated blend via temperature-scaled softmax.

    Args:
        logits: One logit per cuisine.
        temperature: Softmax temperature; 1.0 leaves logits unscaled.

    Returns:
        One probability per cuisine, summing to 1.
    """
    scaled = [logit / temperature for logit in logits]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]
