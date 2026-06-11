"""Tests for pipeline.resolve_brands.

Covers lexicon loading (structure validation, regex compilation, frozen
result), brand resolution semantics (keep-list precedence, first-match-wins
ordering, word boundaries, whole-string replacement), input guards, and
structural spot checks of the production lexicons.

Trap strings come from the brands-noise analysis lens over
01-bronze/data/train.json: 'vegeta' must never match inside 'vegetables' and
'nilla' must never match inside 'vanilla'.
"""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from pipeline import locations
from pipeline.resolve_brands import (
    BrandLexicon,
    BrandPattern,
    load_brand_lexicon,
    resolve_brand_to_generic,
)

FIXTURE_LEXICON_PATH = (
    Path(__file__).parent / "fixtures" / "resolve_brands" / "brand_patterns.json"
)
PRODUCTION_LEXICON_PATH = locations.LEXICONS_DIRECTORY / "brand_patterns.json"
PRODUCTION_ALIASES_PATH = locations.LEXICONS_DIRECTORY / "manual_aliases.json"

MOJIBAKE_HELLMANNS = "hellmannâ€™ or best food canola cholesterol free mayonnais"

KNOWN_STEMMED_ARTIFACT_KEYS = (
    "bulk italian sausag",
    "chees fresh mozzarella",
    "chees fresco queso",
    "dri leav rosemari",
    "dri leav thyme",
    "fresh leav spinach",
    "hellmann' or best food real mayonnais",
    "hellmann' or best food light mayonnais",
    "knorr garlic minicub",
    "knorr chipotl minicub",
    "ragu old world style pasta sauc",
    "ragu cheesi classic alfredo sauc",
    "red kidnei beans, rins and drain",
    "sweet italian sausag links, cut into",
    "torn romain lettuc leav",
    MOJIBAKE_HELLMANNS,
)

CLEANED_TARGET_PATTERN = re.compile(r"[a-z0-9% ]+")


@pytest.fixture(scope="module")
def fixture_lexicon() -> BrandLexicon:
    return load_brand_lexicon(FIXTURE_LEXICON_PATH)


@pytest.fixture(scope="module")
def production_lexicon() -> BrandLexicon:
    return load_brand_lexicon(PRODUCTION_LEXICON_PATH)


# --- load_brand_lexicon: structure ---


def test_load_returns_brand_lexicon(fixture_lexicon):
    assert isinstance(fixture_lexicon, BrandLexicon)
    assert isinstance(fixture_lexicon.patterns, tuple)
    assert isinstance(fixture_lexicon.keep_as_ingredient, tuple)


def test_load_result_is_frozen(fixture_lexicon):
    with pytest.raises(dataclasses.FrozenInstanceError):
        fixture_lexicon.patterns = ()


def test_load_compiles_every_pattern(fixture_lexicon):
    every_rule = fixture_lexicon.keep_as_ingredient + fixture_lexicon.patterns

    for rule in every_rule:
        assert isinstance(rule, BrandPattern)
        assert isinstance(rule.compiled_regex, re.Pattern)
        assert rule.compiled_regex.pattern == rule.regex_source
        assert rule.generic_target


def test_load_preserves_pattern_order(fixture_lexicon):
    targets = [rule.generic_target for rule in fixture_lexicon.patterns]

    # 'alfredo sauce' (specific) must stay ahead of 'pasta sauce' (fallback).
    assert targets.index("alfredo sauce") < targets.index("pasta sauce")


def test_load_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_brand_lexicon(tmp_path / "absent.json")


def test_load_rejects_invalid_regex(tmp_path):
    lexicon_path = tmp_path / "bad_regex.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "keep_as_ingredient": [],
                "patterns": [{"generic": "x", "pattern": "(unclosed"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


def test_load_rejects_missing_section(tmp_path):
    lexicon_path = tmp_path / "missing_section.json"
    lexicon_path.write_text(json.dumps({"patterns": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


def test_load_rejects_unknown_section(tmp_path):
    lexicon_path = tmp_path / "unknown_section.json"
    lexicon_path.write_text(
        json.dumps({"keep_as_ingredient": [], "patterns": [], "extra": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


def test_load_rejects_uppercase_in_pattern(tmp_path):
    # Patterns run against cleaned lowercase text; uppercase can never match.
    lexicon_path = tmp_path / "uppercase.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "keep_as_ingredient": [],
                "patterns": [{"generic": "hot sauce", "pattern": "\\bTabasco\\b"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


def test_load_rejects_empty_generic_target(tmp_path):
    lexicon_path = tmp_path / "empty_generic.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "keep_as_ingredient": [],
                "patterns": [{"generic": "", "pattern": "\\bragu\\b"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


def test_load_rejects_entry_with_missing_field(tmp_path):
    lexicon_path = tmp_path / "missing_field.json"
    lexicon_path.write_text(
        json.dumps(
            {"keep_as_ingredient": [], "patterns": [{"pattern": "\\bragu\\b"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_brand_lexicon(lexicon_path)


# --- resolve_brand_to_generic: word boundaries ---


def test_word_boundary_blocks_vegeta_in_vegetables(fixture_lexicon):
    assert resolve_brand_to_generic("mixed vegetables", fixture_lexicon) is None
    assert (
        resolve_brand_to_generic("vegeta seasoning", fixture_lexicon)
        == "vegetable seasoning"
    )


def test_word_boundary_blocks_nilla_in_vanilla(fixture_lexicon):
    assert resolve_brand_to_generic("vanilla extract", fixture_lexicon) is None
    assert resolve_brand_to_generic("nilla wafers", fixture_lexicon) == "vanilla wafers"


def test_word_boundary_blocks_ragu_in_asparagus(fixture_lexicon):
    assert resolve_brand_to_generic("asparagus spears", fixture_lexicon) is None


# --- resolve_brand_to_generic: resolution semantics ---


def test_pattern_resolves_unseen_kraft_string(fixture_lexicon):
    # Made-up SKU never seen in train: the pattern must still generalize.
    resolved = resolve_brand_to_generic("kraft mexican style cheese", fixture_lexicon)

    assert resolved == "shredded mexican cheese blend"


def test_whole_string_resolves_to_mapped_generic(fixture_lexicon):
    # The generic comes from the mapping, never from residue text.
    resolved = resolve_brand_to_generic(
        "kraft grated parmesan cheese", fixture_lexicon
    )

    assert resolved == "grated parmesan cheese"


def test_keep_list_guinness_maps_to_stout(fixture_lexicon):
    assert resolve_brand_to_generic("guinness beer", fixture_lexicon) == "stout"


def test_keep_list_wins_over_brand_patterns(fixture_lexicon):
    # 'kahlua liqueur' also matches the fixture's 'coffee liqueur' pattern;
    # the keep-list entry must take precedence.
    assert resolve_brand_to_generic("kahlua liqueur", fixture_lexicon) == "kahlua"


def test_bare_ragu_maps_to_pasta_sauce(fixture_lexicon):
    assert resolve_brand_to_generic("ragu", fixture_lexicon) == "pasta sauce"


def test_first_matching_pattern_wins(fixture_lexicon):
    # Specific 'ragu ... alfredo' must beat the bare 'ragu' fallback.
    resolved = resolve_brand_to_generic("ragu classic alfredo sauce", fixture_lexicon)

    assert resolved == "alfredo sauce"


def test_no_match_returns_none(fixture_lexicon):
    assert resolve_brand_to_generic("diced tomatoes", fixture_lexicon) is None


# --- resolve_brand_to_generic: input guards ---


def test_resolve_rejects_non_string(fixture_lexicon):
    with pytest.raises(TypeError):
        resolve_brand_to_generic(None, fixture_lexicon)


def test_resolve_rejects_uppercase_text(fixture_lexicon):
    with pytest.raises(ValueError):
        resolve_brand_to_generic("KRAFT Mexican Style Cheese", fixture_lexicon)


def test_resolve_rejects_unstripped_trademark_text(fixture_lexicon):
    with pytest.raises(ValueError):
        resolve_brand_to_generic("old el paso® taco seasoning", fixture_lexicon)


def test_resolve_rejects_apostrophe_text(fixture_lexicon):
    with pytest.raises(ValueError):
        resolve_brand_to_generic("hellmann's mayonnaise", fixture_lexicon)


def test_resolve_rejects_empty_text(fixture_lexicon):
    with pytest.raises(ValueError):
        resolve_brand_to_generic("   ", fixture_lexicon)


# --- production lexicon: structure and spot checks ---


def test_production_lexicon_loads_with_expected_size(production_lexicon):
    assert len(production_lexicon.patterns) >= 150
    assert len(production_lexicon.keep_as_ingredient) >= 12


def test_production_lexicon_resolves_lens_examples(production_lexicon):
    expected_resolutions = {
        "tabasco pepper sauce": "hot sauce",
        "kraft grated parmesan cheese": "grated parmesan cheese",
        "bertolli classico olive oil": "olive oil",
        "soy vay veri veri teriyaki marinade and sauce": "teriyaki sauce",
        "azteca flour tortillas": "flour tortillas",
        "bacardi superior": "white rum",
        "rotel tomatoes": "diced tomatoes with green chiles",
        "velveeta": "processed cheese",
        "philadelphia cream cheese": "cream cheese",
        "ragu": "pasta sauce",
    }

    for cleaned_text, generic in expected_resolutions.items():
        assert (
            resolve_brand_to_generic(cleaned_text, production_lexicon) == generic
        ), cleaned_text


def test_production_keep_list_preserves_brand_ingredients(production_lexicon):
    expected_keeps = {
        "guinness beer": "stout",
        "marmite": "marmite",
        "maggi": "maggi seasoning",
        "old bay seasoning": "old bay seasoning",
        "kahlua": "kahlua",
        "spam": "spam",
        "grand marnier": "grand marnier",
        "baileys irish cream liqueur": "baileys irish cream",
    }

    for cleaned_text, generic in expected_keeps.items():
        assert (
            resolve_brand_to_generic(cleaned_text, production_lexicon) == generic
        ), cleaned_text


def test_production_lexicon_generalizes_to_unseen_test_brands(production_lexicon):
    # Cleaned forms of test.json brand strings that never occur in train.
    unseen_resolutions = {
        "knorr shrimp flavor bouillon cube": "shrimp bouillon",
        "kraft classic ranch dressing": "ranch dressing",
        "thai kitchen red curry paste": "red curry paste",
        "goya corn oil": "corn oil",
    }

    for cleaned_text, generic in unseen_resolutions.items():
        assert (
            resolve_brand_to_generic(cleaned_text, production_lexicon) == generic
        ), cleaned_text


def test_production_lexicon_skips_variety_names(production_lexicon):
    # Capitalized but NOT branded per the lens: varieties stay unresolved.
    for cleaned_text in (
        "san marzano tomatoes",
        "black mission figs",
        "hatch green chiles",
        "vanilla wafers",
    ):
        assert (
            resolve_brand_to_generic(cleaned_text, production_lexicon) is None
        ), cleaned_text


# --- production manual aliases ---


def test_manual_aliases_cover_stemmed_artifacts():
    aliases = json.loads(PRODUCTION_ALIASES_PATH.read_text(encoding="utf-8"))

    for raw_key in KNOWN_STEMMED_ARTIFACT_KEYS:
        assert raw_key in aliases, raw_key


def test_manual_aliases_map_to_expected_targets():
    aliases = json.loads(PRODUCTION_ALIASES_PATH.read_text(encoding="utf-8"))

    assert aliases["bulk italian sausag"] == "bulk italian sausage"
    assert aliases["dri leav rosemari"] == "dried rosemary leaves"
    assert aliases["ragu old world style pasta sauc"] == "pasta sauce"
    assert aliases["hellmann' or best food real mayonnais"] == "mayonnaise"
    assert aliases["knorr garlic minicub"] == "garlic bouillon cube"
    assert aliases["red kidnei beans, rins and drain"] == "red kidney beans"
    assert aliases["7 Up"] == "lemon lime soda"
    assert (
        aliases["2 1/2 to 3 lb. chicken, cut into serving pieces"] == "whole chicken"
    )


def test_manual_alias_targets_are_cleaned_strings():
    aliases = json.loads(PRODUCTION_ALIASES_PATH.read_text(encoding="utf-8"))

    assert aliases
    for raw_key, target in aliases.items():
        assert isinstance(target, str), raw_key
        assert CLEANED_TARGET_PATTERN.fullmatch(target), raw_key
