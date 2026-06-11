"""Tests for silver.pipeline.strip_modifiers.

Covers lexicon loading (frozen structure, fail-fast validation) and the
strip algorithm: protected exact strings, phrase-regex pass for stacked
marketing modifiers, order-independent token pass, conditional guards
(kosher/whole/packed/extra), the never-strip override, and the
empty-result guard.

Behavioral expectations come from the descriptors analysis lens over
bronze/kaggle/train.json (JSD-verified SAFE-STRIP vs KEEP modifiers).
"""

import dataclasses
from pathlib import Path

import pytest

from silver.pipeline import locations
from silver.pipeline.strip_modifiers import (
    ConditionalGuard,
    ModifierLexicon,
    load_modifier_lexicon,
    strip_safe_modifiers,
)

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures" / "strip_modifiers"
SMALL_LEXICON_PATH = FIXTURES_DIRECTORY / "modifier_strip_small.json"
MISSING_KEY_LEXICON_PATH = FIXTURES_DIRECTORY / "modifier_strip_missing_key.json"
BAD_MODE_LEXICON_PATH = FIXTURES_DIRECTORY / "modifier_strip_bad_mode.json"
PRODUCTION_LEXICON_PATH = locations.LEXICONS_DIRECTORY / "modifier_strip.json"

# Verified KEEP tokens the production lexicon must always protect
# (descriptors lens: baby corn JSD=0.795, dry mustard 0.494, ground cumin
# 0.244, roasted red peppers 0.201, plus color/identity words).
PRODUCTION_NEVER_STRIP_MINIMUM = frozenset(
    (
        "ground", "dried", "dry", "green", "white", "dark", "red", "purple",
        "sweet", "sour", "baby", "toasted", "smoked", "roasted", "light",
    )
)
PRODUCTION_STRIP_TOKEN_MINIMUM_COUNT = 50


@pytest.fixture(scope="module")
def small_lexicon() -> ModifierLexicon:
    return load_modifier_lexicon(SMALL_LEXICON_PATH)


@pytest.fixture(scope="module")
def production_lexicon() -> ModifierLexicon:
    return load_modifier_lexicon(PRODUCTION_LEXICON_PATH)


# --- load_modifier_lexicon ---


def test_load_returns_frozen_structured_lexicon(small_lexicon):
    assert isinstance(small_lexicon, ModifierLexicon)
    assert isinstance(small_lexicon.strip_phrases, tuple)
    assert isinstance(small_lexicon.strip_tokens, frozenset)
    assert isinstance(small_lexicon.conditional_tokens, tuple)
    assert all(
        isinstance(guard, ConditionalGuard)
        for guard in small_lexicon.conditional_tokens
    )
    assert isinstance(small_lexicon.never_strip_tokens, frozenset)
    assert isinstance(small_lexicon.protected_strings, frozenset)


def test_load_result_attributes_are_immutable(small_lexicon):
    with pytest.raises(dataclasses.FrozenInstanceError):
        small_lexicon.strip_tokens = frozenset()


def test_load_orders_conditional_guards_by_token(small_lexicon):
    guard_tokens = [guard.token for guard in small_lexicon.conditional_tokens]

    assert guard_tokens == sorted(guard_tokens)


def test_load_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_modifier_lexicon(FIXTURES_DIRECTORY / "does_not_exist.json")


def test_load_missing_required_key_raises_value_error():
    with pytest.raises(ValueError, match="strip_tokens"):
        load_modifier_lexicon(MISSING_KEY_LEXICON_PATH)


def test_load_unknown_guard_mode_raises_value_error():
    with pytest.raises(ValueError, match="guard mode"):
        load_modifier_lexicon(BAD_MODE_LEXICON_PATH)


# --- token pass ---


def test_prep_tokens_stripped_in_any_order(small_lexicon):
    # Word-order variants must converge (descriptors lens warning).
    assert strip_safe_modifiers("chopped cilantro fresh", small_lexicon) == "cilantro"
    assert strip_safe_modifiers("fresh chopped cilantro", small_lexicon) == "cilantro"
    assert strip_safe_modifiers("cilantro fresh chopped", small_lexicon) == "cilantro"


def test_unlisted_tokens_pass_through_unchanged(small_lexicon):
    assert (
        strip_safe_modifiers("boneless chicken thighs", small_lexicon)
        == "boneless chicken thighs"
    )


# --- phrase pass ---


def test_sodium_phrase_spans_stacked_modifiers(small_lexicon):
    stripped = strip_safe_modifiers(
        "fat free less sodium chicken broth", small_lexicon
    )

    assert stripped == "chicken broth"


def test_phrase_removal_leaves_single_spaces(small_lexicon):
    assert strip_safe_modifiers("low sodium soy sauce", small_lexicon) == "soy sauce"


# --- never-strip override ---


def test_ground_is_never_stripped(small_lexicon):
    # Fixture lists "ground" in BOTH strip_tokens and never_strip_tokens;
    # never_strip must win (ground cumin vs cumin seed JSD=0.244).
    assert strip_safe_modifiers("ground cumin", small_lexicon) == "ground cumin"


def test_dried_is_never_stripped(small_lexicon):
    # dried shrimp vs shrimp JSD=0.451 — identity-bearing.
    assert strip_safe_modifiers("dried shrimp", small_lexicon) == "dried shrimp"


# --- conditional guards ---


def test_packed_kept_when_followed_by_in(small_lexicon):
    # "packed in X" is identity-bearing (oil pack vs water pack).
    stripped = strip_safe_modifiers("tuna packed in olive oil", small_lexicon)

    assert stripped == "tuna packed in olive oil"


def test_packed_stripped_when_not_followed_by_in(small_lexicon):
    stripped = strip_safe_modifiers("firmly packed brown sugar", small_lexicon)

    assert stripped == "brown sugar"


def test_extra_stripped_only_in_extra_large(small_lexicon):
    assert strip_safe_modifiers("extra large eggs", small_lexicon) == "eggs"
    # extra-virgin olive oil JSD=0.043 — bare "extra" must survive.
    assert (
        strip_safe_modifiers("extra virgin olive oil", small_lexicon)
        == "extra virgin olive oil"
    )


def test_kosher_stripped_only_for_salt(small_lexicon):
    assert strip_safe_modifiers("kosher salt", small_lexicon) == "salt"
    assert (
        strip_safe_modifiers("kosher dill pickles", small_lexicon)
        == "kosher dill pickles"
    )


def test_whole_stripped_only_for_milk(small_lexicon):
    assert strip_safe_modifiers("whole milk", small_lexicon) == "milk"
    assert (
        strip_safe_modifiers("whole wheat flour", small_lexicon)
        == "whole wheat flour"
    )


# --- protected strings ---


def test_protected_string_frozen_spinach_untouched(small_lexicon):
    # frozen spinach vs fresh spinach JSD=0.119 — the verified exception
    # to globally stripping "frozen".
    assert strip_safe_modifiers("frozen spinach", small_lexicon) == "frozen spinach"


def test_frozen_stripped_outside_protected_string(small_lexicon):
    assert strip_safe_modifiers("frozen peas", small_lexicon) == "peas"


# --- guards on degenerate input ---


def test_empty_result_guard(small_lexicon):
    # Everything strippable -> return the original text, never empty.
    assert strip_safe_modifiers("chopped fresh", small_lexicon) == "chopped fresh"


def test_empty_result_guard_covers_phrase_only_strings(small_lexicon):
    assert strip_safe_modifiers("fat free", small_lexicon) == "fat free"


def test_strip_rejects_non_string_input(small_lexicon):
    with pytest.raises(TypeError):
        strip_safe_modifiers(None, small_lexicon)


# --- production lexicon ---


def test_production_lexicon_is_structurally_valid(production_lexicon):
    guard_tokens = {guard.token for guard in production_lexicon.conditional_tokens}

    assert len(production_lexicon.strip_tokens) >= PRODUCTION_STRIP_TOKEN_MINIMUM_COUNT
    assert PRODUCTION_NEVER_STRIP_MINIMUM <= production_lexicon.never_strip_tokens
    assert {"extra", "kosher", "packed", "whole"} <= guard_tokens
    assert {"frozen spinach", "fresh mozzarella"} <= production_lexicon.protected_strings
    # A token must never be both unconditionally stripped and protected.
    assert not production_lexicon.strip_tokens & production_lexicon.never_strip_tokens
    assert not production_lexicon.strip_tokens & {g.token for g in production_lexicon.conditional_tokens}


def test_production_lexicon_spot_behaviors(production_lexicon):
    assert (
        strip_safe_modifiers("low sodium soy sauce", production_lexicon)
        == "soy sauce"
    )
    assert (
        strip_safe_modifiers("fat free less sodium chicken broth", production_lexicon)
        == "chicken broth"
    )
    assert strip_safe_modifiers("ground beef", production_lexicon) == "ground beef"
    assert (
        strip_safe_modifiers("boneless skinless chicken breasts", production_lexicon)
        == "chicken breasts"
    )
    assert strip_safe_modifiers("2% reduced fat milk", production_lexicon) == "milk"
    assert strip_safe_modifiers("kosher salt", production_lexicon) == "salt"
    assert (
        strip_safe_modifiers("frozen spinach", production_lexicon) == "frozen spinach"
    )
    # Real-data check findings: frozen chopped spinach is italian-skewed
    # (lens: frozen vs fresh spinach JSD=0.119) and grated orange is zest,
    # not fruit (italian/french vs bare orange mexican/spanish).
    assert (
        strip_safe_modifiers("frozen chopped spinach", production_lexicon)
        == "frozen chopped spinach"
    )
    assert (
        strip_safe_modifiers("grated orange", production_lexicon) == "grated orange"
    )
