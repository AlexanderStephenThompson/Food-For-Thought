"""Tests for pipeline.compile_alias_table — the silver ingredients compiler.

All tests run on synthetic VocabularyBuild structures and mini train
indexes built from handmade Recipe rows, so they are independent of
production data and lexicon churn. Merge-decision loading is exercised
against small JSONL fixtures under tests/fixtures/compile_alias_table/.
"""

import json
from pathlib import Path

import pytest

from pipeline.build_vocabulary import (
    GroupMember,
    PreservedVariant,
    ReviewCandidate,
    VocabularyBuild,
    VocabularyGroup,
)
from pipeline.compile_alias_table import (
    apply_merge_decisions,
    compile_ingredients_payload,
    load_merge_decisions,
    validate_compiled_payload,
)
from pipeline.load_bronze_recipes import Recipe, build_train_index
from pipeline.merge_evidence import CuisineShare, MergeEvidence

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures" / "compile_alias_table"

FAKE_FINGERPRINT = {
    "train_sha256": "0" * 64,
    "lexicon_fingerprint": "f" * 64,
    "random_seed": 42,
}

STANDARD_DECISION_ID = "dark_hot_sauce__vs__hot_sauce"

EVIDENCE_JSD_BITS = 0.51236
EVIDENCE_NULL95_BITS = 0.25004
EVIDENCE_VARIANT_COUNT = 10


def make_index(recipe_rows):
    """Build a TrainIndex from (id, cuisine, ingredients) rows."""
    recipes = [
        Recipe(id=row[0], cuisine=row[1], ingredients=tuple(row[2]))
        for row in recipe_rows
    ]
    return build_train_index(recipes)


def repeat_recipes(start_id, cuisine, ingredients, count):
    """Generate count identical single-ingredient-list recipes."""
    return [(start_id + offset, cuisine, ingredients) for offset in range(count)]


def make_standard_index():
    """Index: hot sauce x12, dark hot sauce x10, bean sauce x8, rare thing x3."""
    rows = (
        repeat_recipes(1, "mexican", ["hot sauce"], 12)
        + repeat_recipes(100, "thai", ["dark hot sauce"], 10)
        + repeat_recipes(200, "chinese", ["bean sauce"], 8)
        + repeat_recipes(300, "thai", ["rare thing"], 3)
    )
    return make_index(rows)


def make_evidence():
    return MergeEvidence(
        variant_count=EVIDENCE_VARIANT_COUNT,
        base_count=12,
        jsd_bits=EVIDENCE_JSD_BITS,
        null95_bits=EVIDENCE_NULL95_BITS,
        jsd_to_null_ratio=2.04931,
        variant_top_cuisines=(CuisineShare(cuisine="thai", share=1.0, lift=2.0),),
        base_top_cuisines=(CuisineShare(cuisine="mexican", share=1.0, lift=1.5),),
    )


def make_group(key, raw_strings=None):
    members = {
        raw: GroupMember(cleaned=key, source="mechanical_normalization")
        for raw in (raw_strings or [key])
    }
    return VocabularyGroup(key=key, members=members)


def make_review_candidate(variant_key="dark hot sauce", base_key="hot sauce"):
    return ReviewCandidate(
        variant_key=variant_key,
        base_key=base_key,
        variant_cleaned=variant_key,
        base_cleaned=base_key,
        variant_raw_strings=(variant_key,),
        evidence=make_evidence(),
        gate_layer="small_sample",
        gate_reason="concentrated small sample",
    )


def make_standard_build(preserved=None, review_candidates=None):
    groups = {
        "hot sauce": make_group("hot sauce"),
        "dark hot sauce": make_group("dark hot sauce"),
        "bean sauce": make_group("bean sauce"),
        "rare thing": make_group("rare thing"),
    }
    if review_candidates is None:
        review_candidates = (make_review_candidate(),)
    return VocabularyBuild(
        groups=groups,
        preserved=dict(preserved or {}),
        alias_scope_keys=frozenset({"hot sauce", "dark hot sauce", "bean sauce"}),
        review_candidates=tuple(review_candidates),
        review_entries=[],
    )


def make_decision(decision_id=STANDARD_DECISION_ID, decision="merge", note="reviewed"):
    return {
        decision_id: {
            "decision_id": decision_id,
            "decision": decision,
            "decided_by": "test_judge",
            "note": note,
        }
    }


def make_alias(alias, frequency, source="mechanical_normalization"):
    return {"alias": alias, "source": source, "rule": None, "train_frequency": frequency}


def make_ingredient(ingredient_id, name, aliases, parent_id=None):
    return {
        "id": ingredient_id,
        "name": name,
        "category": None,
        "parent_id": parent_id,
        "train_mention_count": sum(entry["train_frequency"] for entry in aliases),
        "preserve_evidence": None,
        "aliases": aliases,
    }


def make_payload(ingredients):
    return {
        "schema_version": 1,
        "build": dict(FAKE_FINGERPRINT),
        "ingredients": ingredients,
    }


class TestLoadMergeDecisions:
    def test_valid_file_loads_decisions_by_id(self):
        decisions = load_merge_decisions(FIXTURES_DIRECTORY / "merge_decisions_valid.jsonl")

        assert set(decisions) == {
            "dark_hot_sauce__vs__hot_sauce",
            "dark_bean_sauce__vs__bean_sauce",
            "sweet_hot_sauce__vs__hot_sauce",
        }
        record = decisions["sweet_hot_sauce__vs__hot_sauce"]
        assert record["decision"] == "merge_into:bean sauce"
        assert record["decided_by"] == "human_reviewer"
        assert record["note"] == "mislabeled variant"

    def test_duplicate_decision_id_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            load_merge_decisions(FIXTURES_DIRECTORY / "merge_decisions_duplicate_id.jsonl")

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="decided_by"):
            load_merge_decisions(FIXTURES_DIRECTORY / "merge_decisions_missing_field.jsonl")

    def test_unknown_decision_value_raises(self):
        with pytest.raises(ValueError, match="maybe"):
            load_merge_decisions(FIXTURES_DIRECTORY / "merge_decisions_unknown_decision.jsonl")

    def test_invalid_json_line_raises(self):
        with pytest.raises(ValueError, match="JSON"):
            load_merge_decisions(FIXTURES_DIRECTORY / "merge_decisions_invalid_json.jsonl")


class TestApplyMergeDecisions:
    def test_merge_decision_absorbs_variant_as_manual_review_alias(self):
        build = make_standard_build()
        decisions = make_decision(decision="merge", note="same condiment")

        apply_merge_decisions(build, decisions, make_standard_index())

        assert "dark hot sauce" not in build.groups
        member = build.groups["hot sauce"].members["dark hot sauce"]
        assert member.source == "manual_review"
        assert member.rule == "same condiment"

    def test_merge_into_redirects_to_sibling(self):
        build = make_standard_build()
        decisions = make_decision(decision="merge_into:bean sauce")

        apply_merge_decisions(build, decisions, make_standard_index())

        assert "dark hot sauce" not in build.groups
        assert "dark hot sauce" not in build.groups["hot sauce"].members
        member = build.groups["bean sauce"].members["dark hot sauce"]
        assert member.source == "manual_review"

    def test_merge_into_missing_target_raises(self):
        build = make_standard_build()
        decisions = make_decision(decision="merge_into:no such group")

        with pytest.raises(ValueError, match=STANDARD_DECISION_ID):
            apply_merge_decisions(build, decisions, make_standard_index())

    def test_unresolved_decision_id_raises(self):
        build = make_standard_build()

        with pytest.raises(ValueError, match=STANDARD_DECISION_ID):
            apply_merge_decisions(build, {}, make_standard_index())

    def test_orphan_decision_raises(self):
        build = make_standard_build()
        decisions = make_decision() | make_decision(decision_id="ghost__vs__nothing")

        with pytest.raises(ValueError, match="ghost__vs__nothing"):
            apply_merge_decisions(build, decisions, make_standard_index())


class TestCompileIngredientsPayload:
    def test_preserve_decision_sets_parent_and_evidence(self):
        build = make_standard_build()
        decisions = make_decision(decision="preserve", note="distinct profile")

        payload = compile_ingredients_payload(
            build, decisions, make_standard_index(), FAKE_FINGERPRINT
        )

        by_id = {entry["id"]: entry for entry in payload["ingredients"]}
        variant = by_id["dark_hot_sauce"]
        assert variant["parent_id"] == "hot_sauce"
        assert variant["preserve_evidence"] == {
            "layer": "manual_review",
            "jsd_bits": round(EVIDENCE_JSD_BITS, 4),
            "null95_bits": round(EVIDENCE_NULL95_BITS, 4),
            "variant_count": EVIDENCE_VARIANT_COUNT,
        }

    def test_gate_preserved_variant_links_parent(self):
        preserved = {
            "dark hot sauce": PreservedVariant(
                base_key="hot sauce",
                layer="statistical_gate",
                reason="distinct cuisine profile",
                evidence=make_evidence(),
            )
        }
        build = make_standard_build(preserved=preserved, review_candidates=())

        payload = compile_ingredients_payload(
            build, {}, make_standard_index(), FAKE_FINGERPRINT
        )

        by_id = {entry["id"]: entry for entry in payload["ingredients"]}
        assert by_id["dark_hot_sauce"]["parent_id"] == "hot_sauce"
        assert by_id["dark_hot_sauce"]["preserve_evidence"]["layer"] == "statistical_gate"

    def test_preserve_chain_flattens_to_root_parent(self):
        # Real-data shape: crushed red pepper -> red pepper -> pepper.
        # The schema requires parents to be roots, so both deep variants
        # must link directly to 'pepper'.
        rows = (
            repeat_recipes(1, "italian", ["pepper"], 12)
            + repeat_recipes(100, "mexican", ["red pepper"], 8)
            + repeat_recipes(200, "italian", ["crushed red pepper"], 6)
        )
        index = make_index(rows)
        groups = {
            "pepper": make_group("pepper"),
            "red pepper": make_group("red pepper"),
            "crushed red pepper": make_group("crushed red pepper"),
        }
        preserved = {
            "red pepper": PreservedVariant(
                base_key="pepper",
                layer="statistical_gate",
                reason="distinct profile",
                evidence=make_evidence(),
            ),
            "crushed red pepper": PreservedVariant(
                base_key="red pepper",
                layer="statistical_gate",
                reason="distinct profile",
                evidence=make_evidence(),
            ),
        }
        build = VocabularyBuild(
            groups=groups,
            preserved=preserved,
            alias_scope_keys=frozenset(groups),
            review_candidates=(),
            review_entries=[],
        )

        payload = compile_ingredients_payload(build, {}, index, FAKE_FINGERPRINT)

        by_id = {entry["id"]: entry for entry in payload["ingredients"]}
        assert by_id["red_pepper"]["parent_id"] == "pepper"
        assert by_id["crushed_red_pepper"]["parent_id"] == "pepper"
        validate_compiled_payload(payload, index)

    def test_parent_absorbed_by_decision_raises(self):
        # 'bean sauce' is preserved under 'dark hot sauce', but the review
        # decision merges 'dark hot sauce' away — the parent vanishes.
        preserved = {
            "bean sauce": PreservedVariant(
                base_key="dark hot sauce",
                layer="statistical_gate",
                reason="distinct cuisine profile",
                evidence=make_evidence(),
            )
        }
        build = make_standard_build(preserved=preserved)
        decisions = make_decision(decision="merge")

        with pytest.raises(ValueError, match="bean sauce"):
            compile_ingredients_payload(
                build, decisions, make_standard_index(), FAKE_FINGERPRINT
            )

    def test_tail_groups_stay_out_of_vocabulary(self):
        payload = compile_ingredients_payload(
            make_standard_build(), make_decision(), make_standard_index(), FAKE_FINGERPRINT
        )

        assert "rare_thing" not in {entry["id"] for entry in payload["ingredients"]}

    def test_representative_alias_source_is_canonical_surface_form(self):
        payload = compile_ingredients_payload(
            make_standard_build(), make_decision(), make_standard_index(), FAKE_FINGERPRINT
        )

        by_id = {entry["id"]: entry for entry in payload["ingredients"]}
        aliases = {entry["alias"]: entry for entry in by_id["hot_sauce"]["aliases"]}
        assert aliases["hot sauce"]["source"] == "canonical_surface_form"
        assert aliases["dark hot sauce"]["source"] == "manual_review"

    def test_train_mention_count_counts_distinct_recipes(self):
        payload = compile_ingredients_payload(
            make_standard_build(), make_decision(), make_standard_index(), FAKE_FINGERPRINT
        )

        by_id = {entry["id"]: entry for entry in payload["ingredients"]}
        assert by_id["hot_sauce"]["train_mention_count"] == 22
        aliases = {entry["alias"]: entry for entry in by_id["hot_sauce"]["aliases"]}
        assert aliases["dark hot sauce"]["train_frequency"] == 10

    def test_payload_embeds_schema_version_and_build(self):
        payload = compile_ingredients_payload(
            make_standard_build(), make_decision(), make_standard_index(), FAKE_FINGERPRINT
        )

        assert payload["schema_version"] == 1
        assert payload["build"] == FAKE_FINGERPRINT

    def test_payload_sorted_and_deterministic(self):
        index = make_standard_index()

        first = compile_ingredients_payload(
            make_standard_build(), make_decision(), index, FAKE_FINGERPRINT
        )
        second = compile_ingredients_payload(
            make_standard_build(), make_decision(), index, FAKE_FINGERPRINT
        )

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        ids = [entry["id"] for entry in first["ingredients"]]
        assert ids == sorted(ids)
        for ingredient in first["ingredients"]:
            aliases = [entry["alias"] for entry in ingredient["aliases"]]
            assert aliases == sorted(aliases)


class TestValidateCompiledPayload:
    def coverage_index(self):
        """Index whose strings are fully coverable by the test payloads."""
        rows = (
            repeat_recipes(1, "mexican", ["hot sauce"], 12)
            + repeat_recipes(100, "thai", ["dark hot sauce"], 10)
            + repeat_recipes(200, "chinese", ["bean sauce"], 8)
        )
        return make_index(rows)

    def full_coverage_payload(self):
        return make_payload(
            [
                make_ingredient(
                    "hot_sauce",
                    "hot sauce",
                    [make_alias("hot sauce", 12), make_alias("dark hot sauce", 10)],
                ),
                make_ingredient("bean_sauce", "bean sauce", [make_alias("bean sauce", 8)]),
            ]
        )

    def test_alias_maps_to_exactly_one_id(self):
        payload = make_payload(
            [
                make_ingredient("hot_sauce", "hot sauce", [make_alias("hot sauce", 12)]),
                make_ingredient(
                    "bean_sauce",
                    "bean sauce",
                    [make_alias("bean sauce", 8), make_alias("hot sauce", 12)],
                ),
            ]
        )

        with pytest.raises(ValueError, match="hot sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_canonical_name_never_another_ingredients_alias(self):
        payload = make_payload(
            [
                make_ingredient("hot_sauce", "hot sauce", [make_alias("dark hot sauce", 10)]),
                make_ingredient(
                    "bean_sauce",
                    "bean sauce",
                    [make_alias("bean sauce", 8), make_alias("hot sauce", 12)],
                ),
            ]
        )

        with pytest.raises(ValueError, match="hot sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_slug_collision_raises(self):
        payload = make_payload(
            [
                make_ingredient(
                    "hot_sauce",
                    "hot sauce",
                    [make_alias("hot sauce", 12), make_alias("dark hot sauce", 10)],
                ),
                make_ingredient("hot_sauce", "bean sauce", [make_alias("bean sauce", 8)]),
            ]
        )

        with pytest.raises(ValueError, match="hot_sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_invalid_slug_format_raises(self):
        payload = make_payload(
            [
                make_ingredient(
                    "Hot-Sauce",
                    "hot sauce",
                    [make_alias("hot sauce", 12), make_alias("dark hot sauce", 10)],
                ),
                make_ingredient("bean_sauce", "bean sauce", [make_alias("bean sauce", 8)]),
            ]
        )

        with pytest.raises(ValueError, match="Hot-Sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_parent_missing_raises(self):
        payload = self.full_coverage_payload()
        payload["ingredients"][0]["parent_id"] = "ghost"

        with pytest.raises(ValueError, match="ghost"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_parent_depth_over_two_raises(self):
        payload = make_payload(
            [
                make_ingredient(
                    "hot_sauce",
                    "hot sauce",
                    [make_alias("hot sauce", 12)],
                    parent_id="dark_hot_sauce",
                ),
                make_ingredient(
                    "dark_hot_sauce",
                    "dark hot sauce",
                    [make_alias("dark hot sauce", 10)],
                    parent_id="bean_sauce",
                ),
                make_ingredient("bean_sauce", "bean sauce", [make_alias("bean sauce", 8)]),
            ]
        )

        with pytest.raises(ValueError, match="hot_sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_uncovered_frequent_string_raises(self):
        payload = make_payload(
            [
                make_ingredient(
                    "hot_sauce",
                    "hot sauce",
                    [make_alias("hot sauce", 12), make_alias("dark hot sauce", 10)],
                ),
            ]
        )

        with pytest.raises(ValueError, match="bean sauce"):
            validate_compiled_payload(payload, self.coverage_index())

    def test_low_coverage_ratio_raises(self):
        # 'rare thing' (frequency 3) needs no alias, but its 3 uncovered
        # mentions drag coverage to 30/33 — far below the 98.8% floor.
        with pytest.raises(ValueError, match="coverage"):
            validate_compiled_payload(self.full_coverage_payload(), make_standard_index())

    def test_coverage_statistics_returned(self):
        payload = self.full_coverage_payload()
        payload["ingredients"][0]["parent_id"] = "bean_sauce"

        statistics = validate_compiled_payload(payload, self.coverage_index())

        assert statistics.ingredient_count == 2
        assert statistics.alias_count == 3
        assert statistics.parent_link_count == 1
        assert statistics.covered_mention_count == 30
        assert statistics.total_mention_count == 30
        assert statistics.coverage_ratio == 1.0
