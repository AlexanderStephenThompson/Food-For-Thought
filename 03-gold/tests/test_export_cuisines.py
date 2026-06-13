"""Tests for app_pipeline.export_cuisines."""

import math

from app_pipeline.export_cuisines import build_cuisines_asset
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_evaluation_payload,
)
from tests.model_payload_builders import make_cuisines_payload

POSITION_TOLERANCE = 1e-3


def _build_asset():
    return build_cuisines_asset(
        make_cuisines_payload(), make_evaluation_payload(), APP_BUILD_BLOCK
    )


def test_cuisines_asset_merges_recall_from_evaluation():
    asset = _build_asset()

    recall_by_id = {entry["id"]: entry["recall"] for entry in asset["cuisines"]}
    assert recall_by_id == {"italian": 0.9, "mexican": 0.7, "thai": 0.8}


def test_cuisines_asset_places_positions_on_unit_circle():
    asset = _build_asset()

    for entry in asset["cuisines"]:
        radius = math.hypot(entry["position"]["x"], entry["position"]["y"])
        assert abs(radius - 1.0) < POSITION_TOLERANCE


def test_cuisines_asset_orders_positions_by_family_then_id():
    asset = _build_asset()

    ordered_ids = [entry["id"] for entry in asset["cuisines"]]
    # families: italian=mediterranean, mexican=latin_american, thai=southeast_asian
    assert ordered_ids == ["mexican", "italian", "thai"]


def test_cuisines_asset_dedupes_similarity_edges():
    asset = _build_asset()

    # italian<->mexican appears in both neighbor lists but yields one edge.
    matching_edges = [
        edge
        for edge in asset["edges"]
        if {edge["a"], edge["b"]} == {"italian", "mexican"}
    ]
    assert len(matching_edges) == 1
    assert matching_edges[0]["similarity"] == 0.45


def test_cuisines_asset_builds_display_names():
    asset = _build_asset()

    names_by_id = {entry["id"]: entry["name"] for entry in asset["cuisines"]}
    assert names_by_id["thai"] == "Thai"
