"""Tests for the app exporter's serialization contract (asset_io)."""

import json

from app_pipeline.asset_io import serialize_asset_json


def test_serialize_asset_json_is_compact_and_sorted():
    payload = {"beta": [1, 2], "alpha": {"nested": True}}

    content = serialize_asset_json(payload)

    assert content == '{"alpha":{"nested":true},"beta":[1,2]}\n'


def test_serialize_asset_json_round_trips():
    payload = {"cuisines": ["thai"], "values": [0.1234, None]}

    content = serialize_asset_json(payload)

    assert json.loads(content) == payload
