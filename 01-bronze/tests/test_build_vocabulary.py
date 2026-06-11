"""Tests for pipeline.build_vocabulary — the vocabulary build orchestration.

All tests run on synthetic mini-indexes and the small fixture lexicons under
tests/fixtures/build_vocabulary/ so they are independent of production
lexicon churn. One test loads the production lexicon bundle for structural
validity.
"""

from pathlib import Path

import pytest

from pipeline import locations
from pipeline.build_vocabulary import (
    ALIAS_SCOPE_MINIMUM_FREQUENCY,
    apply_pair_outcomes,
    build_review_queue_entries,
    build_vocabulary_from_index,
    count_group_recipes,
    generate_candidate_pairs,
    group_strings_mechanically,
    load_pipeline_lexicons,
    merge_groups_by_strip_and_brand,
    select_representative_cleaned,
)
from tests.recipe_builders import make_index, repeat_recipes

FIXTURE_LEXICONS_DIRECTORY = Path(__file__).parent / "fixtures" / "build_vocabulary"
PRODUCTION_LEXICONS_DIRECTORY = locations.LEXICONS_DIRECTORY


@pytest.fixture(scope="module")
def lexicons():
    return load_pipeline_lexicons(FIXTURE_LEXICONS_DIRECTORY)


class TestMechanicalGrouping:
    def test_plural_and_case_variants_share_a_group(self, lexicons):
        groups = group_strings_mechanically(["Bay Leaves", "bay leaf"], lexicons)

        assert "bay leaf" in groups
        assert set(groups["bay leaf"].members) == {"Bay Leaves", "bay leaf"}

    def test_member_source_is_mechanical_normalization(self, lexicons):
        groups = group_strings_mechanically(["Bay Leaves"], lexicons)

        member = groups["bay leaf"].members["Bay Leaves"]
        assert member.source == "mechanical_normalization"

    def test_manual_alias_joins_target_group(self, lexicons):
        groups = group_strings_mechanically(
            ["dri leav rosemari", "dried rosemary leaves"], lexicons
        )

        assert "dri leav rosemari" not in groups
        group = groups["dried rosemary leaf"]
        assert set(group.members) == {"dri leav rosemari", "dried rosemary leaves"}
        assert group.members["dri leav rosemari"].source == "manual_alias"


class TestStripAndBrandPass:
    def test_strip_merge_into_existing_group(self, lexicons):
        index = make_index(
            repeat_recipes(1, "italian", ["onions"], 5)
            + repeat_recipes(100, "italian", ["chopped onions"], 2)
        )
        groups = group_strings_mechanically(["onions", "chopped onions"], lexicons)

        groups = merge_groups_by_strip_and_brand(groups, lexicons, index)

        assert "chopped onion" not in groups
        group = groups["onion"]
        assert group.members["chopped onions"].source == "modifier_strip"

    def test_brand_string_resolves_into_existing_generic_group(self, lexicons):
        index = make_index(
            repeat_recipes(1, "mexican", ["hot sauce"], 5)
            + repeat_recipes(100, "mexican", ["BrandX Hot Sauce"], 2)
        )
        groups = group_strings_mechanically(["hot sauce", "BrandX Hot Sauce"], lexicons)

        groups = merge_groups_by_strip_and_brand(groups, lexicons, index)

        assert groups["hot sauce"].members["BrandX Hot Sauce"].source == "brand_pattern"

    def test_brand_target_creates_group_when_generic_absent(self, lexicons):
        index = make_index(repeat_recipes(1, "british", ["Zestico Mixer"], 4))
        groups = group_strings_mechanically(["Zestico Mixer"], lexicons)

        groups = merge_groups_by_strip_and_brand(groups, lexicons, index)

        assert "lemon cordial" in groups
        assert groups["lemon cordial"].members["Zestico Mixer"].source == "brand_pattern"

    def test_canonical_name_is_highest_frequency_member(self, lexicons):
        index = make_index(
            repeat_recipes(1, "italian", ["onions"], 5)
            + repeat_recipes(100, "italian", ["chopped onions"], 2)
        )
        groups = group_strings_mechanically(["onions", "chopped onions"], lexicons)
        groups = merge_groups_by_strip_and_brand(groups, lexicons, index)

        assert select_representative_cleaned(groups["onion"], index) == "onions"
        assert count_group_recipes(groups["onion"], index) == 7


class TestCandidatePairGeneration:
    def test_fish_sauce_has_no_parent_in_sauce(self, lexicons):
        index = make_index(
            repeat_recipes(1, "thai", ["fish sauce"], 5)
            + repeat_recipes(100, "italian", ["sauce"], 5)
        )
        groups = group_strings_mechanically(["fish sauce", "sauce"], lexicons)

        pairs = generate_candidate_pairs(groups, lexicons, index)

        assert pairs == []

    def test_whitelisted_modifier_prefix_generates_candidate(self, lexicons):
        index = make_index(
            repeat_recipes(1, "chinese", ["soy sauce"], 5)
            + repeat_recipes(100, "chinese", ["dark soy sauce"], 5)
        )
        groups = group_strings_mechanically(["soy sauce", "dark soy sauce"], lexicons)

        pairs = generate_candidate_pairs(groups, lexicons, index)

        assert [(pair.variant_key, pair.base_key) for pair in pairs] == [
            ("dark soy sauce", "soy sauce")
        ]

    def test_longest_base_wins(self, lexicons):
        index = make_index(
            repeat_recipes(1, "chinese", ["soy sauce"], 5)
            + repeat_recipes(100, "chinese", ["sweet soy sauce"], 5)
            + repeat_recipes(200, "chinese", ["dark sweet soy sauce"], 4)
        )
        groups = group_strings_mechanically(
            ["soy sauce", "sweet soy sauce", "dark sweet soy sauce"], lexicons
        )

        pairs = generate_candidate_pairs(groups, lexicons, index)

        pair_map = {pair.variant_key: pair.base_key for pair in pairs}
        assert pair_map["dark sweet soy sauce"] == "sweet soy sauce"

    def test_single_token_groups_generate_no_pairs(self, lexicons):
        index = make_index(repeat_recipes(1, "indian", ["paprika"], 5))
        groups = group_strings_mechanically(["paprika"], lexicons)

        assert generate_candidate_pairs(groups, lexicons, index) == []

    def test_tail_groups_below_alias_scope_are_excluded(self, lexicons):
        tail_count = ALIAS_SCOPE_MINIMUM_FREQUENCY - 1
        index = make_index(
            repeat_recipes(1, "chinese", ["soy sauce"], 5)
            + repeat_recipes(100, "chinese", ["dark soy sauce"], tail_count)
        )
        groups = group_strings_mechanically(["soy sauce", "dark soy sauce"], lexicons)

        assert generate_candidate_pairs(groups, lexicons, index) == []


class TestEndToEndBuild:
    def build_three_way_index(self):
        """Index with a clear preserve, a clear merge, and a review case.

        Base 'soy sauce' spans chinese/italian evenly. 'dark soy sauce'
        (n=24, all thai — a cuisine absent from the base) must PRESERVE.
        'sweet soy sauce' (n=20, same blend as base) must MERGE at the
        statistical layer. 'dark hot sauce' vs 'hot sauce' (n=10, fully
        concentrated) routes to small-sample REVIEW.
        """
        rows = (
            repeat_recipes(1, "chinese", ["soy sauce"], 25)
            + repeat_recipes(100, "italian", ["soy sauce"], 25)
            + repeat_recipes(200, "thai", ["dark soy sauce"], 24)
            + repeat_recipes(300, "chinese", ["sweet soy sauce"], 12)
            + repeat_recipes(400, "italian", ["sweet soy sauce"], 8)
            + repeat_recipes(500, "mexican", ["hot sauce"], 12)
            + repeat_recipes(600, "thai", ["dark hot sauce"], 10)
        )
        return make_index(rows)

    def test_gate_preserve_sets_parent_key(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        assert "dark soy sauce" in build.groups
        assert build.parent_keys["dark soy sauce"] == "soy sauce"

    def test_gate_merge_absorbs_variant_members(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        assert "sweet soy sauce" not in build.groups
        member = build.groups["soy sauce"].members["sweet soy sauce"]
        assert member.source == "statistical_gate"

    def test_concentrated_small_sample_routes_to_review(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        review_ids = [entry["decision_id"] for entry in build.review_entries]
        assert review_ids == ["dark_hot_sauce__vs__hot_sauce"]
        assert "dark hot sauce" in build.groups

    def test_review_entry_is_self_contained(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        entry = build.review_entries[0]
        required_fields = {
            "decision_id",
            "variant_string",
            "base_string",
            "variant_train_frequency",
            "base_train_frequency",
            "jsd_bits",
            "null95_bits",
            "jsd_to_null_ratio",
            "variant_top_cuisines",
            "base_top_cuisines",
            "example_recipe_ids",
            "gate_route",
            "suggested_decision",
            "suggestion_reason",
        }
        assert required_fields <= set(entry)
        assert entry["variant_string"] == "dark hot sauce"
        assert entry["base_string"] == "hot sauce"
        assert entry["variant_train_frequency"] == 10
        assert entry["suggested_decision"] in {"merge", "preserve"}
        assert entry["variant_top_cuisines"][0]["cuisine"] == "thai"

    def test_review_entry_floats_are_rounded(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        entry = build.review_entries[0]
        for field in ("jsd_bits", "null95_bits", "jsd_to_null_ratio"):
            assert entry[field] == round(entry[field], 4)

    def test_example_recipe_ids_are_sorted_and_capped(self, lexicons):
        build = build_vocabulary_from_index(self.build_three_way_index(), lexicons)

        example_ids = build.review_entries[0]["example_recipe_ids"]
        assert example_ids == sorted(example_ids)
        assert len(example_ids) <= 3

    def test_build_is_deterministic(self, lexicons):
        index = self.build_three_way_index()

        first = build_vocabulary_from_index(index, lexicons)
        second = build_vocabulary_from_index(index, lexicons)

        assert first.review_entries == second.review_entries
        assert sorted(first.groups) == sorted(second.groups)
        assert first.parent_keys == second.parent_keys

    def test_always_merge_lexicon_beats_statistics(self, lexicons):
        rows = (
            repeat_recipes(1, "chinese", ["soy sauce"], 25)
            + repeat_recipes(100, "italian", ["soy sauce"], 25)
            + repeat_recipes(200, "thai", ["low sodium soy sauce"], 24)
        )
        build = build_vocabulary_from_index(make_index(rows), lexicons)

        assert "low sodium soy sauce" not in build.groups
        member = build.groups["soy sauce"].members["low sodium soy sauce"]
        assert member.source == "always_merge_lexicon"

    def test_alias_scope_keys_exclude_tail_groups(self, lexicons):
        rows = repeat_recipes(1, "chinese", ["soy sauce"], 5) + repeat_recipes(
            100, "thai", ["rare thing"], 2
        )
        build = build_vocabulary_from_index(make_index(rows), lexicons)

        assert "soy sauce" in build.alias_scope_keys
        assert "rare thing" not in build.alias_scope_keys
        assert "rare thing" in build.groups


class TestApplyPairOutcomesChains:
    def test_merge_chain_follows_redirects(self, lexicons):
        # Small samples are split across cuisines so they take the
        # small-sample-default MERGE path, not the concentrated REVIEW path.
        index = make_index(
            repeat_recipes(1, "chinese", ["soy sauce"], 6)
            + repeat_recipes(50, "italian", ["soy sauce"], 6)
            + repeat_recipes(100, "chinese", ["sweet soy sauce"], 4)
            + repeat_recipes(150, "italian", ["sweet soy sauce"], 2)
            + repeat_recipes(200, "chinese", ["dark sweet soy sauce"], 3)
            + repeat_recipes(250, "italian", ["dark sweet soy sauce"], 2)
        )
        groups = group_strings_mechanically(
            ["soy sauce", "sweet soy sauce", "dark sweet soy sauce"], lexicons
        )
        pairs = generate_candidate_pairs(groups, lexicons, index)

        groups, preserved, review_candidates, absorbed_into = apply_pair_outcomes(
            groups, pairs, lexicons, index
        )

        assert set(groups) == {"soy sauce"}
        assert "dark sweet soy sauce" in groups["soy sauce"].members
        assert preserved == {}
        assert review_candidates == []
        assert absorbed_into == {
            "dark sweet soy sauce": "sweet soy sauce",
            "sweet soy sauce": "soy sauce",
        }


class TestProductionLexicons:
    def test_production_bundle_loads_and_is_structurally_valid(self):
        lexicons = load_pipeline_lexicons(PRODUCTION_LEXICONS_DIRECTORY)

        assert "dark" not in lexicons.variant_modifier_tokens
        assert "fish" not in lexicons.variant_modifier_tokens
        assert "thai" in lexicons.variant_modifier_tokens
        assert lexicons.manual_aliases["7 Up"] == "lemon lime soda"
        assert len(lexicons.manual_aliases) >= 60

    def test_production_never_strip_supplies_dark_for_candidates(self):
        lexicons = load_pipeline_lexicons(PRODUCTION_LEXICONS_DIRECTORY)

        assert "dark" in lexicons.modifier_lexicon.never_strip_tokens


class TestQueueSerialization:
    def test_entries_sorted_by_decision_id(self, lexicons):
        rows = (
            repeat_recipes(1, "mexican", ["hot sauce"], 12)
            + repeat_recipes(100, "thai", ["dark hot sauce"], 10)
            + repeat_recipes(200, "chinese", ["bean sauce"], 12)
            + repeat_recipes(300, "thai", ["dark bean sauce"], 10)
        )
        build = build_vocabulary_from_index(make_index(rows), lexicons)

        decision_ids = [entry["decision_id"] for entry in build.review_entries]
        assert decision_ids == sorted(decision_ids)

    def test_build_review_queue_entries_round_trips_evidence(self, lexicons):
        rows = (
            repeat_recipes(1, "mexican", ["hot sauce"], 12)
            + repeat_recipes(100, "thai", ["dark hot sauce"], 10)
        )
        index = make_index(rows)
        build = build_vocabulary_from_index(index, lexicons)

        entries = build_review_queue_entries(build.review_candidates, index)

        assert entries == build.review_entries
