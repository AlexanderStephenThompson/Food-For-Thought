"""Tests for pipeline.build_cuisines.

All taxonomy tests run on small in-test recipe payloads. Only the production
smoke test reads lexicons/cuisine_families.json, and only ever read-only.
"""

from pathlib import Path

import pytest

from pipeline.build_cuisines import build_cuisines_payload, load_cuisine_families

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FAMILIES_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "build_cuisines" / "cuisine_families.json"
)
PRODUCTION_FAMILIES_PATH = PROJECT_ROOT / "lexicons" / "cuisine_families.json"
EXPECTED_PRODUCTION_CUISINE_COUNT = 20

FINGERPRINT = {
    "train_sha256": "a" * 64,
    "lexicon_fingerprint": "b" * 64,
    "random_seed": 42,
}

THREE_CUISINE_FAMILIES = {
    "alpha": "family_one",
    "beta": "family_one",
    "gamma": "family_two",
}
TWO_CUISINE_FAMILIES = {"alpha": "family_one", "beta": "family_one"}

# Eligibility corpus: one cuisine large enough that a >=10-recipe ingredient
# can still fall below the 2% coverage floor.
BULK_RECIPE_COUNT = 600
LOW_SUPPORT_OCCURRENCES = 9
LOW_COVERAGE_OCCURRENCES = 10
ELIGIBLE_OCCURRENCES = 12

# Lift corpus: two 20-recipe cuisines with engineered ingredient shares.
LIFT_CUISINE_RECIPE_COUNT = 20
ALPHA_SPECIAL_OCCURRENCES = 10
SHARED_ALPHA_OCCURRENCES = 15
SHARED_BETA_OCCURRENCES = 5
BETA_SPECIAL_OCCURRENCES = 10
BETA_RECIPE_ID_OFFSET = 100

DISTINCTIVE_INGREDIENT_CAP = 15
NEIGHBOR_CAP = 4


def _make_recipes_payload(recipe_rows: list[tuple[int, str, list[str]]]) -> dict:
    """Build a minimal staged recipes payload from (id, cuisine, ingredients) rows."""
    recipes = [
        {
            "id": recipe_id,
            "cuisine": cuisine,
            "ingredient_ids": list(ingredient_ids),
            "unresolved_ingredients": [],
            "raw_ingredient_count": len(ingredient_ids),
        }
        for recipe_id, cuisine, ingredient_ids in recipe_rows
    ]
    return {"schema_version": 1, "build": FINGERPRINT, "recipes": recipes}


def _cuisine_by_id(payload: dict, cuisine_id: str) -> dict:
    """Return the cuisine entry with the given id, failing the test if absent."""
    for cuisine in payload["cuisines"]:
        if cuisine["id"] == cuisine_id:
            return cuisine
    raise AssertionError(f"cuisine {cuisine_id!r} not present in payload")


def _neighbor_similarity(payload: dict, cuisine_id: str, neighbor_id: str) -> float:
    """Return the similarity that cuisine_id reports for neighbor_id."""
    for neighbor in _cuisine_by_id(payload, cuisine_id)["neighbors"]:
        if neighbor["id"] == neighbor_id:
            return neighbor["similarity"]
    raise AssertionError(f"{neighbor_id!r} is not a neighbor of {cuisine_id!r}")


def _build_three_cuisine_payload() -> dict:
    """Three small cuisines with partial ingredient overlap, no universal ingredient."""
    rows = [
        (1, "alpha", ["anise", "basil"]),
        (2, "alpha", ["anise"]),
        (3, "beta", ["basil", "cumin"]),
        (4, "beta", ["cumin"]),
        (5, "gamma", ["dill"]),
        (6, "gamma", ["dill", "anise"]),
    ]
    return _make_recipes_payload(rows)


def _build_six_cuisine_payload() -> tuple[dict, dict[str, str]]:
    """Six cuisines: one near-identical pair plus four mutually disjoint ones."""
    rows = [
        (1, "pair_a", ["anise", "basil"]),
        (2, "pair_a", ["anise", "basil"]),
        (3, "pair_b", ["anise", "basil"]),
        (4, "pair_b", ["anise", "cumin"]),
        (5, "far_c", ["dill", "elder"]),
        (6, "far_c", ["dill"]),
        (7, "far_d", ["elder", "fennel"]),
        (8, "far_d", ["fennel"]),
        (9, "far_e", ["ginger", "honey"]),
        (10, "far_e", ["ginger"]),
        (11, "far_f", ["honey", "iris"]),
        (12, "far_f", ["iris"]),
    ]
    cuisine_ids = ["pair_a", "pair_b", "far_c", "far_d", "far_e", "far_f"]
    families = {cuisine_id: "family_any" for cuisine_id in cuisine_ids}
    return _make_recipes_payload(rows), families


def _build_eligibility_payload() -> dict:
    """One 600-recipe cuisine exercising both eligibility thresholds."""
    rows = []
    for recipe_id in range(BULK_RECIPE_COUNT):
        ingredient_ids = ["staple"]
        if recipe_id < LOW_SUPPORT_OCCURRENCES:
            ingredient_ids.append("low_support")
        if recipe_id < LOW_COVERAGE_OCCURRENCES:
            ingredient_ids.append("low_coverage")
        if recipe_id < ELIGIBLE_OCCURRENCES:
            ingredient_ids.append("eligible")
        rows.append((recipe_id, "bulk", ingredient_ids))
    return _make_recipes_payload(rows)


def _build_lift_payload() -> dict:
    """Two 20-recipe cuisines with exact lifts 2.0, 1.5, and 1.0 in alpha."""
    rows = []
    for index in range(LIFT_CUISINE_RECIPE_COUNT):
        ingredient_ids = ["staple"]
        if index < ALPHA_SPECIAL_OCCURRENCES:
            ingredient_ids.append("alpha_special")
        if index < SHARED_ALPHA_OCCURRENCES:
            ingredient_ids.append("shared_more_alpha")
        rows.append((index, "alpha", ingredient_ids))
    for index in range(LIFT_CUISINE_RECIPE_COUNT):
        ingredient_ids = ["staple"]
        if index < BETA_SPECIAL_OCCURRENCES:
            ingredient_ids.append("beta_special")
        if index < SHARED_BETA_OCCURRENCES:
            ingredient_ids.append("shared_more_alpha")
        rows.append((BETA_RECIPE_ID_OFFSET + index, "beta", ingredient_ids))
    return _make_recipes_payload(rows)


def _build_rounding_payload() -> dict:
    """Lift corpus plus one extra alpha recipe so lifts have repeating decimals."""
    payload = _build_lift_payload()
    payload["recipes"].append(
        {
            "id": 200,
            "cuisine": "alpha",
            "ingredient_ids": ["staple"],
            "unresolved_ingredients": [],
            "raw_ingredient_count": 1,
        }
    )
    return payload


def test_load_cuisine_families_ignores_private_keys():
    families = load_cuisine_families(FIXTURE_FAMILIES_PATH)

    assert families == {
        "alpha": "family_one",
        "beta": "family_one",
        "gamma": "family_two",
    }


def test_production_families_cover_twenty_cuisines():
    families = load_cuisine_families(PRODUCTION_FAMILIES_PATH)

    assert len(families) == EXPECTED_PRODUCTION_CUISINE_COUNT
    assert all(isinstance(family, str) and family for family in families.values())
    assert families["thai"] == "southeast_asian"


def test_similarity_symmetric_and_self_excluded():
    payload = build_cuisines_payload(
        _build_three_cuisine_payload(), THREE_CUISINE_FAMILIES, FINGERPRINT
    )

    for cuisine in payload["cuisines"]:
        neighbor_ids = [neighbor["id"] for neighbor in cuisine["neighbors"]]
        assert cuisine["id"] not in neighbor_ids
    for left, right in [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")]:
        assert _neighbor_similarity(payload, left, right) == _neighbor_similarity(
            payload, right, left
        )


def test_neighbors_are_top_four_sorted():
    recipes_payload, families = _build_six_cuisine_payload()

    payload = build_cuisines_payload(recipes_payload, families, FINGERPRINT)

    for cuisine in payload["cuisines"]:
        similarities = [neighbor["similarity"] for neighbor in cuisine["neighbors"]]
        assert len(similarities) == NEIGHBOR_CAP
        assert similarities == sorted(similarities, reverse=True)
    assert _cuisine_by_id(payload, "pair_a")["neighbors"][0]["id"] == "pair_b"
    assert _cuisine_by_id(payload, "pair_b")["neighbors"][0]["id"] == "pair_a"


def test_idf_downweights_universal_ingredients():
    rows = [
        (1, "alpha", ["water", "apple"]),
        (2, "alpha", ["water", "apple"]),
        (3, "beta", ["water", "banana"]),
        (4, "beta", ["water", "banana"]),
    ]

    payload = build_cuisines_payload(
        _make_recipes_payload(rows), TWO_CUISINE_FAMILIES, FINGERPRINT
    )

    # "water" is in every recipe, so idf zeroes it out; nothing else is shared.
    assert _neighbor_similarity(payload, "alpha", "beta") == 0.0


def test_lift_ranked_within_cuisine_only():
    payload = build_cuisines_payload(
        _build_lift_payload(), TWO_CUISINE_FAMILIES, FINGERPRINT
    )

    alpha_entries = _cuisine_by_id(payload, "alpha")["distinctive_ingredients"]
    alpha_ids = [entry["id"] for entry in alpha_entries]
    assert alpha_ids == ["alpha_special", "shared_more_alpha", "staple"]
    assert [entry["lift"] for entry in alpha_entries] == [2.0, 1.5, 1.0]
    assert "beta_special" not in alpha_ids
    beta_ids = [
        entry["id"]
        for entry in _cuisine_by_id(payload, "beta")["distinctive_ingredients"]
    ]
    assert beta_ids[0] == "beta_special"


def test_support_and_coverage_eligibility_enforced():
    payload = build_cuisines_payload(
        _build_eligibility_payload(), {"bulk": "family_bulk"}, FINGERPRINT
    )

    distinctive_ids = [
        entry["id"]
        for entry in _cuisine_by_id(payload, "bulk")["distinctive_ingredients"]
    ]
    assert "eligible" in distinctive_ids  # 12/600 sits exactly on the 2% floor
    assert "staple" in distinctive_ids
    assert "low_support" not in distinctive_ids  # 9 recipes < 10 support floor
    assert "low_coverage" not in distinctive_ids  # 10/600 < 2% coverage floor


def test_distinctive_ingredients_capped_at_fifteen():
    ingredient_ids = [f"item_{index:02d}" for index in range(18)]
    rows = [(recipe_id, "solo", list(ingredient_ids)) for recipe_id in range(20)]

    payload = build_cuisines_payload(
        _make_recipes_payload(rows), {"solo": "family_solo"}, FINGERPRINT
    )

    distinctive = _cuisine_by_id(payload, "solo")["distinctive_ingredients"]
    assert len(distinctive) == DISTINCTIVE_INGREDIENT_CAP
    # All lifts tie at 1.0, so the cap keeps the 15 lowest ids deterministically.
    assert [entry["id"] for entry in distinctive] == ingredient_ids[:15]


def test_missing_family_raises():
    families_without_gamma = {"alpha": "family_one", "beta": "family_one"}

    with pytest.raises(ValueError, match="gamma"):
        build_cuisines_payload(
            _build_three_cuisine_payload(), families_without_gamma, FINGERPRINT
        )


def test_family_without_recipes_raises():
    families_with_ghost = dict(THREE_CUISINE_FAMILIES, ghost="family_ghost")

    with pytest.raises(ValueError, match="ghost"):
        build_cuisines_payload(
            _build_three_cuisine_payload(), families_with_ghost, FINGERPRINT
        )


def test_cuisines_sorted_by_id():
    recipes_payload, families = _build_six_cuisine_payload()

    payload = build_cuisines_payload(recipes_payload, families, FINGERPRINT)

    cuisine_ids = [cuisine["id"] for cuisine in payload["cuisines"]]
    assert cuisine_ids == ["far_c", "far_d", "far_e", "far_f", "pair_a", "pair_b"]


def test_rounding_four_decimals():
    payload = build_cuisines_payload(
        _build_rounding_payload(), TWO_CUISINE_FAMILIES, FINGERPRINT
    )

    alpha_entries = _cuisine_by_id(payload, "alpha")["distinctive_ingredients"]
    alpha_special = next(
        entry for entry in alpha_entries if entry["id"] == "alpha_special"
    )
    # (10/21) / (10/41) = 41/21 = 1.95238...; coverage 10/21 = 0.47619...
    assert alpha_special["lift"] == 1.9524
    assert alpha_special["cuisine_coverage"] == 0.4762
    for cuisine in payload["cuisines"]:
        for neighbor in cuisine["neighbors"]:
            assert neighbor["similarity"] == round(neighbor["similarity"], 4)
        for entry in cuisine["distinctive_ingredients"]:
            assert entry["lift"] == round(entry["lift"], 4)
            assert entry["cuisine_coverage"] == round(entry["cuisine_coverage"], 4)


def test_payload_has_schema_version_build_and_recipe_counts():
    payload = build_cuisines_payload(
        _build_three_cuisine_payload(), THREE_CUISINE_FAMILIES, FINGERPRINT
    )

    assert payload["schema_version"] == 1
    assert payload["build"] == FINGERPRINT
    assert _cuisine_by_id(payload, "alpha")["recipe_count"] == 2
    assert _cuisine_by_id(payload, "alpha")["family"] == "family_one"
