"""Tests for silver.pipeline.resolve_ingredient — the runtime fallback chain.

Exercises every tier of the IngredientResolver against a handwritten
vocabulary fixture (tests/fixtures/resolve_ingredient/vocabulary.json)
combined with the PRODUCTION lexicons directory, so cleaning, modifier
stripping, brand resolution, and singularization all run through the
real chain.

Fixture design notes:
- 'fish sauce' and a generic 'sauce' ingredient coexist to prove the
  token-drop head rule ('fish sauce' must never collapse to 'sauce').
- scallion (700 mentions) and green_onion (500) share the cleaned form
  'spring onions' to prove count-based collision tie-breaks.
- chickpea and garbanzo_bean both have 250 mentions and share the
  cleaned form 'chickpeas' to prove the smaller-id tie-break.
- dark_soy_sauce carries parent_id soy_sauce: resolution returns the
  child, never the parent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silver.pipeline import locations
from silver.pipeline.build_vocabulary import load_pipeline_lexicons
from silver.pipeline.resolve_ingredient import IngredientResolver, ResolutionResult

VOCABULARY_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "resolve_ingredient" / "vocabulary.json"
)
PRODUCTION_LEXICONS_DIRECTORY = locations.LEXICONS_DIRECTORY


@pytest.fixture(scope="module")
def resolver() -> IngredientResolver:
    return IngredientResolver.from_paths(
        VOCABULARY_FIXTURE_PATH, PRODUCTION_LEXICONS_DIRECTORY
    )


def test_exact_alias_tier_wins(resolver: IngredientResolver) -> None:
    result = resolver.resolve("Thai fish sauce")

    # The raw string would also clean-match, so the method proves tier 1
    # fired before tier 2.
    assert result == ResolutionResult("fish_sauce", "exact_alias", ())


def test_cleaned_match_tier(resolver: IngredientResolver) -> None:
    result = resolver.resolve("Fish Sauce!")

    assert result == ResolutionResult("fish_sauce", "cleaned_match", ())


def test_cleaned_match_falls_back_to_lookup_key(
    resolver: IngredientResolver,
) -> None:
    # 'fish sauces' is no stored cleaned form; only its singularized
    # lookup key 'fish sauce' matches, still under the cleaned_match tier.
    result = resolver.resolve("Fish Sauces")

    assert result == ResolutionResult("fish_sauce", "cleaned_match", ())


def test_modifier_stripped_tier(resolver: IngredientResolver) -> None:
    result = resolver.resolve("chopped fresh cilantro")

    assert result == ResolutionResult("cilantro", "modifier_stripped_match", ())


def test_brand_resolved_tier(resolver: IngredientResolver) -> None:
    # Unseen branded string: never an alias, so it must route through the
    # production brand pattern \bkikkoman\b -> 'soy sauce'.
    result = resolver.resolve("Kikkoman Naturally Brewed Soy Sauce")

    assert result == ResolutionResult("soy_sauce", "brand_resolved_match", ())


def test_token_drop_resolves_cubed_pork_style_string(
    resolver: IngredientResolver,
) -> None:
    # 'speckled' stands in for prep junk like 'cubed' that the modifier
    # lexicon does NOT know, forcing the token-drop tier (literal 'cubed'
    # is a strip token and would resolve one tier earlier).
    result = resolver.resolve("speckled pork shoulder")

    assert result == ResolutionResult(
        "pork_shoulder", "token_drop_match", ("speckled",)
    )


def test_token_drop_never_drops_head_token(
    resolver: IngredientResolver,
) -> None:
    # The only conceivable match ('fish sauce') would require dropping the
    # final head token 'zebra', which is forbidden.
    result = resolver.resolve("fish sauce zebra")

    assert result == ResolutionResult(None, "unresolved", ())


def test_token_drop_never_collapses_to_bare_head(
    resolver: IngredientResolver,
) -> None:
    # Dropping 'mystery' would leave only the bare head 'sauce', which is
    # in the vocabulary — but '<anything> sauce' must never resolve to
    # plain 'sauce'.
    result = resolver.resolve("mystery sauce")

    assert result == ResolutionResult(None, "unresolved", ())


def test_two_token_drop_for_long_strings(resolver: IngredientResolver) -> None:
    # No single drop matches, the key has four tokens, and dropping the
    # two non-final junk tokens reaches 'pork shoulder'.
    result = resolver.resolve("zebra mountain pork shoulder")

    assert result == ResolutionResult(
        "pork_shoulder", "token_drop_match", ("zebra", "mountain")
    )


def test_token_drop_prefers_dropping_modifier_over_noun(
    resolver: IngredientResolver,
) -> None:
    # Dropping the modifier 'red' reaches serrano_pepper (104 mentions);
    # dropping the noun 'serrano' reaches red_pepper (459). Modifier
    # drops outrank mention count, so the specific pepper wins.
    result = resolver.resolve("red serrano peppers")

    assert result == ResolutionResult(
        "serrano_pepper", "token_drop_match", ("red",)
    )


def test_token_drop_prefers_higher_mention_count(
    resolver: IngredientResolver,
) -> None:
    # Dropping 'soy' reaches fish_sauce (600 mentions); dropping 'fish'
    # reaches soy_sauce (4382). Equal-length candidates, neither dropped
    # token is a modifier, so the higher mention count wins.
    result = resolver.resolve("soy fish sauce")

    assert result == ResolutionResult("soy_sauce", "token_drop_match", ("fish",))


def test_tie_breaks_deterministic(resolver: IngredientResolver) -> None:
    # 'Spring Onions' (green_onion, 500) and 'spring onions' (scallion,
    # 700) collide on the cleaned form; the higher mention count wins.
    result = resolver.resolve("SPRING ONIONS")

    assert result == ResolutionResult("scallion", "cleaned_match", ())


def test_tie_break_equal_counts_prefers_smaller_id(
    resolver: IngredientResolver,
) -> None:
    # chickpea and garbanzo_bean both have 250 mentions and collide on
    # the cleaned form 'chickpeas'; the lexicographically smaller id wins.
    result = resolver.resolve("CHICKPEAS")

    assert result == ResolutionResult("chickpea", "cleaned_match", ())


def test_resolves_child_ingredient_not_parent(
    resolver: IngredientResolver,
) -> None:
    result = resolver.resolve("Dark Soy Sauce")

    assert result == ResolutionResult("dark_soy_sauce", "cleaned_match", ())


def test_unresolved_returns_none_method(resolver: IngredientResolver) -> None:
    result = resolver.resolve("galactic moon dust")

    assert result.ingredient_id is None
    assert result.method == "unresolved"
    assert result.dropped_tokens == ()


def test_degenerate_input_unresolved(resolver: IngredientResolver) -> None:
    # Cleaning '14.5' leaves no alphabetic token, which must short-circuit
    # straight to unresolved rather than raise.
    result = resolver.resolve("14.5")

    assert result == ResolutionResult(None, "unresolved", ())


def test_type_and_value_guards(resolver: IngredientResolver) -> None:
    with pytest.raises(TypeError):
        resolver.resolve(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolver.resolve("")
    with pytest.raises(ValueError):
        resolver.resolve("   ")


def test_from_payload_matches_from_paths(
    resolver: IngredientResolver,
) -> None:
    payload = json.loads(VOCABULARY_FIXTURE_PATH.read_text(encoding="utf-8"))
    lexicons = load_pipeline_lexicons(PRODUCTION_LEXICONS_DIRECTORY)
    payload_resolver = IngredientResolver.from_payload(payload, lexicons)

    result = payload_resolver.resolve("Thai fish sauce")

    assert result == resolver.resolve("Thai fish sauce")
    assert result == ResolutionResult("fish_sauce", "exact_alias", ())


def test_production_lexicons_smoke(resolver: IngredientResolver) -> None:
    # Trademark mark stripping, casefolding, and hyphen folding all come
    # from the production cleaning chain and lexicons.
    trademark_result = resolver.resolve("Fish™ Sauce")
    hyphen_result = resolver.resolve("Low-Sodium Soy Sauce")

    assert trademark_result == ResolutionResult("fish_sauce", "cleaned_match", ())
    assert hyphen_result == ResolutionResult("soy_sauce", "cleaned_match", ())
