"""Tests for pipeline.singularize: token singularization and lookup keys.

The lexicon under test is 01-bronze/lexicons/singularize_exceptions.json, built from
the morphology lens over the 459 s-ending tokens in the train vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import locations
from pipeline.singularize import (
    SingularizeExceptions,
    load_singularize_exceptions,
    make_lookup_key,
    singularize_token,
)

LEXICON_PATH = locations.LEXICONS_DIRECTORY / "singularize_exceptions.json"


@pytest.fixture(scope="module")
def loaded_exceptions() -> SingularizeExceptions:
    return load_singularize_exceptions(LEXICON_PATH)


# --- load_singularize_exceptions ---


def test_load_returns_frozen_structure_with_expected_types(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert isinstance(loaded_exceptions.no_singularize, frozenset)
    assert isinstance(loaded_exceptions.irregular, dict)
    assert "molasses" in loaded_exceptions.no_singularize
    assert loaded_exceptions.irregular["leaves"] == "leaf"


def test_load_rejects_payload_missing_required_keys(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "missing_keys.json"
    lexicon_path.write_text(json.dumps({"irregular": {}}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_singularize_exceptions(lexicon_path)


def test_load_rejects_payload_with_wrong_value_types(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "wrong_types.json"
    payload = {"no_singularize": "oats", "irregular": {"leaves": "leaf"}}
    lexicon_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_singularize_exceptions(lexicon_path)


# --- singularize_token: exception set (rule 1) ---


def test_exception_list_protects_aminos_and_brussels(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("aminos", loaded_exceptions) == "aminos"
    assert singularize_token("brussels", loaded_exceptions) == "brussels"


def test_exception_set_takes_precedence_over_suffix_rules(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    # Without protection: molasses -> molass (sses), krispies -> krispy (ies).
    assert singularize_token("molasses", loaded_exceptions) == "molasses"
    assert singularize_token("krispies", loaded_exceptions) == "krispies"


# --- singularize_token: irregular map (rule 2) ---


def test_irregular_tomatoes(loaded_exceptions: SingularizeExceptions) -> None:
    assert singularize_token("tomatoes", loaded_exceptions) == "tomato"


def test_irregular_potatoes(loaded_exceptions: SingularizeExceptions) -> None:
    assert singularize_token("potatoes", loaded_exceptions) == "potato"


def test_leaves_halves_loaves_take_f_form_via_irregular_map(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("leaves", loaded_exceptions) == "leaf"
    assert singularize_token("halves", loaded_exceptions) == "half"
    assert singularize_token("loaves", loaded_exceptions) == "loaf"


def test_chilies_chillies_chiles_stay_distinct_keys(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    chili = singularize_token("chilies", loaded_exceptions)
    chilli = singularize_token("chillies", loaded_exceptions)
    chile = singularize_token("chiles", loaded_exceptions)

    assert chili == "chili"
    assert chilli == "chilli"
    assert chile == "chile"
    assert len({chili, chilli, chile}) == 3


def test_irregular_map_takes_precedence_over_suffix_rules(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    # Without the map: cookies -> cooky, brownies -> browny (ies rule).
    assert singularize_token("cookies", loaded_exceptions) == "cookie"
    assert singularize_token("brownies", loaded_exceptions) == "brownie"


# --- singularize_token: suffix rules (rules 3-6) ---


def test_ies_becomes_y_for_long_tokens(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("berries", loaded_exceptions) == "berry"
    assert singularize_token("anchovies", loaded_exceptions) == "anchovy"


def test_ves_rule_keeps_chives_olives_cloves(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    # 'ves' must strip only the final 's' -- never become 'f'.
    assert singularize_token("chives", loaded_exceptions) == "chive"
    assert singularize_token("olives", loaded_exceptions) == "olive"
    assert singularize_token("cloves", loaded_exceptions) == "clove"


def test_es_strip_for_ches_shes_xes_endings(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("peaches", loaded_exceptions) == "peach"
    assert singularize_token("radishes", loaded_exceptions) == "radish"
    assert singularize_token("boxes", loaded_exceptions) == "box"


def test_ss_us_is_guard_protects_couscous_hummus_molasses(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("couscous", loaded_exceptions) == "couscous"
    assert singularize_token("hummus", loaded_exceptions) == "hummus"
    assert singularize_token("molasses", loaded_exceptions) == "molasses"


def test_ss_ending_tokens_are_never_stripped(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("swiss", loaded_exceptions) == "swiss"
    assert singularize_token("bass", loaded_exceptions) == "bass"
    assert singularize_token("watercress", loaded_exceptions) == "watercress"


def test_final_s_strip_requires_length_above_three(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert singularize_token("peas", loaded_exceptions) == "pea"
    assert singularize_token("gas", loaded_exceptions) == "gas"


# --- singularize_token: idempotence and input validation ---


def test_singularize_is_idempotent_on_representative_tokens(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    representative_tokens = [
        "leaves", "halves", "tomatoes", "chilies", "chillies", "chiles",
        "chives", "olives", "cloves", "berries", "peaches", "couscous",
        "hummus", "molasses", "aminos", "brussels", "onions", "shallots",
        "boneless", "skinless", "octopuses",
    ]
    for token in representative_tokens:
        once = singularize_token(token, loaded_exceptions)
        twice = singularize_token(once, loaded_exceptions)
        assert twice == once, f"not idempotent: {token} -> {once} -> {twice}"


def test_every_irregular_value_is_a_fixed_point(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    for singular_form in sorted(loaded_exceptions.irregular.values()):
        result = singularize_token(singular_form, loaded_exceptions)
        assert result == singular_form


def test_singularize_token_rejects_empty_and_spaced_tokens(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    with pytest.raises(ValueError):
        singularize_token("", loaded_exceptions)
    with pytest.raises(ValueError):
        singularize_token("bay leaves", loaded_exceptions)


# --- make_lookup_key ---


def test_bay_leaves_becomes_bay_leaf(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert make_lookup_key("bay leaves", loaded_exceptions) == "bay leaf"


def test_make_lookup_key_singularizes_each_token(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    result = make_lookup_key("boneless skinless chicken breasts", loaded_exceptions)
    assert result == "boneless skinless chicken breast"


def test_make_lookup_key_preserves_percent_and_digit_tokens(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    assert make_lookup_key("1% low fat milk", loaded_exceptions) == "1% low fat milk"


def test_make_lookup_key_rejects_uncleaned_text(
    loaded_exceptions: SingularizeExceptions,
) -> None:
    with pytest.raises(ValueError):
        make_lookup_key("Bay Leaves", loaded_exceptions)
    with pytest.raises(ValueError):
        make_lookup_key("bay  leaves", loaded_exceptions)
    with pytest.raises(ValueError):
        make_lookup_key(" bay leaves", loaded_exceptions)
    with pytest.raises(ValueError):
        make_lookup_key("", loaded_exceptions)
