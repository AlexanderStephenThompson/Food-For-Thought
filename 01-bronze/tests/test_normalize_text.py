"""Tests for silver_pipeline.normalize_text.

Covers the ordered cleaning chain: mojibake repair, trademark stripping,
casefold, quote normalization, quantity stripping, accent folding,
ampersand mapping, apostrophe deletion, punctuation folding, whitespace
collapse, and the degenerate-key guard.

Trap strings come from the brands-noise and morphology analysis lenses
over 01-bronze/data/train.json.
"""

import pytest

from silver_pipeline.normalize_text import (
    DegenerateKeyError,
    clean_ingredient_text,
    collapse_whitespace,
    delete_apostrophes,
    fold_accents,
    map_ampersand_to_and,
    normalize_quotes,
    punctuation_to_space,
    repair_mojibake,
    strip_quantities,
    strip_trademark_marks,
)

MOJIBAKE_HELLMANNS = "hellmannâ€™ or best food canola cholesterol free mayonnais"
REPAIRED_HELLMANNS = "hellmann’ or best food canola cholesterol free mayonnais"

TRAP_STRINGS = (
    MOJIBAKE_HELLMANNS,
    "Old El Paso™ taco seasoning",
    "BACARDI® Superior",
    "Soy Vay® Veri Veri Teriyaki®",
    "I Can't Believe It's Not Butter!® All Purpose Sticks",
    "(14.5 oz.) diced tomatoes",
    "(    oz.) tomato sauce",
    "(   oz.) tomato paste",
    "2 1/2 to 3 lb. chicken, cut into serving pieces",
    "2-lb chicken",
    "pork chops, 1 inch thick",
    "half & half",
    "hellmann's mayonnaise",
    "Piment d'Espelette",
    "tomato purée",
    "crème fraîche",
    "Kahlúa",
    "parmigiano-reggiano cheese",
    "1% low-fat milk",
    "2% reduced-fat milk",
    "Zatarain’s Jambalaya Mix",
    "sheep’s milk cheese",
    "asafetida (powder)",
    "7 Up",
    "v8",
    "licor 43",
)


# --- step 1: repair_mojibake ---


def test_repair_mojibake_restores_hellmanns_string():
    repaired = repair_mojibake(MOJIBAKE_HELLMANNS)

    assert repaired == REPAIRED_HELLMANNS
    assert "’" in repaired


def test_repair_mojibake_leaves_clean_text_untouched():
    assert repair_mojibake("tomato purée") == "tomato purée"


def test_repair_mojibake_falls_back_on_codec_error():
    # Emoji cannot round-trip through cp1252, so repair must fall back.
    unrepairable = "â€ broken \U0001f600"

    assert repair_mojibake(unrepairable) == unrepairable


# --- step 2: strip_trademark_marks ---


def test_trademark_stripped_without_glued_tm():
    cleaned = clean_ingredient_text("Old El Paso™ taco seasoning")

    assert "pasotm" not in cleaned
    assert cleaned == "old el paso taco seasoning"


def test_strip_trademark_marks_removes_all_three_symbols():
    assert strip_trademark_marks("a® b™ c©") == "a b c"


def test_registered_mark_brand_cleans_without_residue():
    assert clean_ingredient_text("BACARDI® Superior") == "bacardi superior"


# --- step 3: casefold ---


def test_clean_casefolds_uppercase_brand():
    cleaned = clean_ingredient_text("KRAFT Mexican Style Cheese")

    assert cleaned == "kraft mexican style cheese"


# --- step 4: normalize_quotes ---


def test_curly_apostrophe_normalized():
    assert normalize_quotes("Zatarain’s") == "Zatarain's"
    assert normalize_quotes("‘quoted’") == "'quoted'"


def test_curly_and_straight_apostrophe_twins_converge():
    cleaned_curly = clean_ingredient_text("sheep’s milk cheese")
    cleaned_straight = clean_ingredient_text("sheep's milk cheese")

    assert cleaned_curly == cleaned_straight == "sheeps milk cheese"


# --- step 5: strip_quantities ---


def test_quantity_parenthetical_removed():
    assert clean_ingredient_text("(14.5 oz.) diced tomatoes") == "diced tomatoes"


def test_blank_ounce_parenthetical_removed():
    assert clean_ingredient_text("(    oz.) tomato sauce") == "tomato sauce"
    assert clean_ingredient_text("(   oz.) tomato paste") == "tomato paste"


def test_leading_pound_range_quantity_removed():
    cleaned = clean_ingredient_text("2 1/2 to 3 lb. chicken, cut into serving pieces")

    assert cleaned == "chicken cut into serving pieces"


def test_interior_inch_quantity_preserved():
    # Only LEADING quantity expressions are stripped.
    cleaned = clean_ingredient_text("pork chops, 1 inch thick")

    assert cleaned == "pork chops 1 inch thick"


def test_percent_token_preserved_in_one_percent_milk():
    cleaned = clean_ingredient_text("1% low-fat milk")

    assert "1%" in cleaned
    assert cleaned == "1% low fat milk"


def test_percent_token_preserved_in_two_percent_milk():
    assert clean_ingredient_text("2% reduced-fat milk") == "2% reduced fat milk"


def test_word_parenthetical_is_not_treated_as_quantity():
    assert strip_quantities("asafetida (powder)") == "asafetida (powder)"
    assert clean_ingredient_text("asafetida (powder)") == "asafetida powder"


# --- step 6: fold_accents ---


def test_accent_fold_tomato_puree():
    assert clean_ingredient_text("tomato purée") == "tomato puree"


def test_accent_fold_creme_fraiche():
    assert clean_ingredient_text("crème fraîche") == "creme fraiche"


def test_accent_fold_kahlua():
    assert clean_ingredient_text("Kahlúa") == "kahlua"


def test_fold_accents_passes_ascii_through():
    assert fold_accents("plain ascii text") == "plain ascii text"


# --- step 7: map_ampersand_to_and ---


def test_ampersand_becomes_and_for_half_and_half():
    assert clean_ingredient_text("half & half") == "half and half"


def test_map_ampersand_never_deletes():
    assert map_ampersand_to_and("a&b") == "a and b"


# --- step 8: delete_apostrophes ---


def test_apostrophe_deleted_from_hellmanns():
    assert delete_apostrophes("hellmann's") == "hellmanns"
    assert clean_ingredient_text("hellmann's mayonnaise") == "hellmanns mayonnaise"


def test_apostrophe_deleted_not_spaced_for_piment_despelette():
    assert clean_ingredient_text("Piment d'Espelette") == "piment despelette"


# --- step 9: punctuation_to_space ---


def test_hyphen_fold_joins_parmigiano_reggiano_pair():
    cleaned_hyphenated = clean_ingredient_text("parmigiano-reggiano cheese")
    cleaned_spaced = clean_ingredient_text("parmigiano reggiano cheese")

    assert cleaned_hyphenated == cleaned_spaced == "parmigiano reggiano cheese"


def test_exclamation_and_punctuation_fold_in_butter_brand():
    cleaned = clean_ingredient_text(
        "I Can't Believe It's Not Butter!® All Purpose Sticks"
    )

    assert cleaned == "i cant believe its not butter all purpose sticks"


def test_slash_becomes_space():
    assert punctuation_to_space("a/b") == "a b"


# --- step 10: collapse_whitespace ---


def test_collapse_whitespace_trims_and_singles_spaces():
    assert collapse_whitespace("  a \t b\n c  ") == "a b c"


# --- step 11: degenerate-key guard ---


def test_degenerate_key_raises():
    with pytest.raises(DegenerateKeyError):
        clean_ingredient_text("14.5")


def test_degenerate_key_error_carries_raw_text():
    with pytest.raises(DegenerateKeyError) as caught:
        clean_ingredient_text("( 12.5 )")

    assert caught.value.raw_text == "( 12.5 )"


def test_degenerate_key_error_is_value_error():
    assert issubclass(DegenerateKeyError, ValueError)


def test_seven_up_survives_degenerate_guard():
    # 'up' contains alphabetic characters, so '7 Up' is not degenerate.
    assert clean_ingredient_text("7 Up") == "7 up"


def test_empty_string_raises_degenerate_key_error():
    with pytest.raises(DegenerateKeyError):
        clean_ingredient_text("   ")


# --- input validation ---


def test_clean_rejects_non_string_input():
    with pytest.raises(TypeError):
        clean_ingredient_text(None)


# --- end-to-end and idempotence ---


def test_full_clean_of_mojibake_hellmanns_string():
    cleaned = clean_ingredient_text(MOJIBAKE_HELLMANNS)

    assert cleaned == "hellmann or best food canola cholesterol free mayonnais"


def test_already_clean_string_is_unchanged():
    assert clean_ingredient_text("diced tomatoes") == "diced tomatoes"


@pytest.mark.parametrize("trap_string", TRAP_STRINGS, ids=repr)
def test_clean_is_idempotent_over_trap_strings(trap_string):
    cleaned_once = clean_ingredient_text(trap_string)

    assert clean_ingredient_text(cleaned_once) == cleaned_once
