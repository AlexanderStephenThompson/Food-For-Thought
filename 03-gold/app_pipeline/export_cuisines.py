"""Build the cuisine atlas asset: taxonomy, recall, and the similarity web.

The radial layout is precomputed here — 20 fixed nodes need a stable,
deterministic arrangement, not a force simulation in the browser.
Cuisines are ordered by (family, id) so families cluster as adjacent arcs
on the circle.
"""

from __future__ import annotations

import math

CUISINES_ASSET_FILENAME = "cuisines.json"
POSITION_DECIMALS = 4
FIRST_NODE_ANGLE_RADIANS = -math.pi / 2

DISPLAY_NAME_OVERRIDES = {
    "cajun_creole": "Cajun Creole",
    "southern_us": "Southern US",
}


def format_display_name(cuisine_id: str) -> str:
    """Render a cuisine id as its display name."""
    if cuisine_id in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[cuisine_id]
    return cuisine_id.replace("_", " ").title()


def compute_atlas_positions(ordered_cuisine_ids: list[str]) -> dict[str, dict]:
    """Place cuisines evenly on the unit circle, starting at the top.

    Args:
        ordered_cuisine_ids: Cuisine ids in display order; adjacency on
            the circle follows this order.

    Returns:
        Mapping of cuisine id -> {"x": ..., "y": ...} rounded coordinates.
    """
    step = 2 * math.pi / len(ordered_cuisine_ids)
    positions = {}
    for position_number, cuisine_id in enumerate(ordered_cuisine_ids):
        angle = FIRST_NODE_ANGLE_RADIANS + position_number * step
        positions[cuisine_id] = {
            "x": round(math.cos(angle), POSITION_DECIMALS),
            "y": round(math.sin(angle), POSITION_DECIMALS),
        }
    return positions


def summarize_similarity_edges(cuisines: list[dict]) -> list[dict]:
    """Deduplicate the per-cuisine neighbor lists into undirected edges."""
    similarity_by_pair: dict[tuple[str, str], float] = {}
    for cuisine in cuisines:
        for neighbor in cuisine["neighbors"]:
            pair = tuple(sorted((cuisine["id"], neighbor["id"])))
            existing = similarity_by_pair.get(pair, 0.0)
            similarity_by_pair[pair] = max(existing, neighbor["similarity"])
    edges = [
        {"a": pair[0], "b": pair[1], "similarity": similarity}
        for pair, similarity in similarity_by_pair.items()
    ]
    edges.sort(key=lambda edge: (-edge["similarity"], edge["a"], edge["b"]))
    return edges


def build_cuisines_asset(
    cuisines_payload: dict, evaluation_payload: dict, fingerprint: dict
) -> dict:
    """Merge the silver taxonomy with model recall and the radial layout.

    Args:
        cuisines_payload: Parsed silver cuisines.json document.
        evaluation_payload: Gold reports/evaluation.json content.
        fingerprint: App build block embedded in the asset.

    Returns:
        Asset with cuisines (family-ordered, positioned, recall-annotated)
        and deduplicated similarity edges.
    """
    recall_by_id = {
        entry["cuisine"]: entry["recall"]
        for entry in evaluation_payload["per_cuisine"]
    }
    ordered_cuisines = sorted(
        cuisines_payload["cuisines"],
        key=lambda cuisine: (cuisine["family"], cuisine["id"]),
    )
    positions = compute_atlas_positions(
        [cuisine["id"] for cuisine in ordered_cuisines]
    )
    cuisines = [
        {
            "id": cuisine["id"],
            "name": format_display_name(cuisine["id"]),
            "family": cuisine["family"],
            "recipe_count": cuisine["recipe_count"],
            "recall": recall_by_id[cuisine["id"]],
            "neighbors": list(cuisine["neighbors"]),
            "distinctive": [
                {
                    "id": entry["id"],
                    "name": entry["id"].replace("_", " "),
                    "lift": entry["lift"],
                    "coverage": entry["cuisine_coverage"],
                }
                for entry in cuisine["distinctive_ingredients"]
            ],
            "position": positions[cuisine["id"]],
        }
        for cuisine in ordered_cuisines
    ]
    return {
        "build": dict(fingerprint),
        "schema_version": 1,
        "cuisines": cuisines,
        "edges": summarize_similarity_edges(cuisines_payload["cuisines"]),
    }
