"""Build the app's ingredient asset: search index and explorer data in one.

One asset serves both jobs — splitting would duplicate 2,813 names and
their aliases across two files for no measurable win. Aliases drop their
provenance fields (the app only matches and ranks) and the canonical
surface form (already present as the name).
"""

from __future__ import annotations

INGREDIENTS_ASSET_FILENAME = "ingredients.json"


def _slim_aliases(entry: dict) -> list[dict]:
    """Keep alias text and frequency; drop the canonical-name duplicate."""
    return [
        {"alias": alias["alias"], "train_frequency": alias["train_frequency"]}
        for alias in entry["aliases"]
        if alias["alias"] != entry["name"]
    ]


def build_ingredients_asset(ingredients_payload: dict, fingerprint: dict) -> dict:
    """Slim the silver vocabulary for the browser.

    Args:
        ingredients_payload: Parsed silver ingredients.json document.
        fingerprint: App build block embedded in the asset.

    Returns:
        Asset with one entry per ingredient: id, name, mentions, slimmed
        aliases, parent_id, derived children, and preserve evidence.
    """
    entries = ingredients_payload["ingredients"]
    children_by_parent: dict[str, list[str]] = {}
    for entry in entries:
        if entry["parent_id"] is not None:
            children_by_parent.setdefault(entry["parent_id"], []).append(
                entry["id"]
            )

    ingredients = [
        {
            "id": entry["id"],
            "name": entry["name"],
            "mentions": entry["train_mention_count"],
            "aliases": _slim_aliases(entry),
            "parent_id": entry["parent_id"],
            "children": sorted(children_by_parent.get(entry["id"], [])),
            "evidence": entry["preserve_evidence"],
        }
        for entry in sorted(entries, key=lambda entry: entry["id"])
    ]
    return {
        "build": dict(fingerprint),
        "schema_version": 1,
        "ingredients": ingredients,
    }
