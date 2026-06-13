"""Tests for app_pipeline.export_ingredients."""

from app_pipeline.export_ingredients import build_ingredients_asset
from tests.app_payload_builders import (
    APP_BUILD_BLOCK,
    make_silver_ingredients_payload,
)


def _entries_by_id():
    asset = build_ingredients_asset(
        make_silver_ingredients_payload(), APP_BUILD_BLOCK
    )
    return {entry["id"]: entry for entry in asset["ingredients"]}, asset


def test_ingredients_asset_derives_children_from_parent_links():
    entries, _ = _entries_by_id()

    assert entries["soy_sauce"]["children"] == ["dark_soy_sauce"]
    assert entries["dark_soy_sauce"]["parent_id"] == "soy_sauce"
    assert entries["basil"]["children"] == []


def test_ingredients_asset_slims_aliases_and_drops_canonical_duplicate():
    entries, _ = _entries_by_id()

    pasta_aliases = entries["pasta"]["aliases"]
    assert pasta_aliases == [{"alias": "penne pasta", "train_frequency": 9}]
    assert entries["fish_sauce"]["aliases"] == []


def test_ingredients_asset_keeps_evidence_and_mentions():
    entries, asset = _entries_by_id()

    assert entries["dark_soy_sauce"]["evidence"]["jsd_bits"] == 0.49
    assert entries["basil"]["evidence"] is None
    assert entries["rice"]["mentions"] == 60
    assert [entry["id"] for entry in asset["ingredients"]] == sorted(entries)
