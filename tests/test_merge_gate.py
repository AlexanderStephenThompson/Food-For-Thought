"""Tests for pipeline.merge_gate.

Gate decisions run on synthetic MergeEvidence instances plus a small fixture
lexicon directory (tests/fixtures/merge_gate). One test group loads the four
PRODUCTION lexicons under lexicons/ and replays the validated cases from the
signal-variants analysis (dark soy preserve, lower sodium force-merge,
usukuchi named-variety preserve, green chilies do-not-merge, ...).
"""

import json
from pathlib import Path

import pytest

from pipeline.merge_evidence import CuisineShare, MergeEvidence
from pipeline.merge_gate import (
    BORDERLINE_JSD_BAND,
    BORDERLINE_RATIO_BAND,
    JSD_FLOOR_BITS,
    MIN_SUPPORT,
    NULL_MULTIPLIER,
    SMALL_SAMPLE_REVIEW_MINIMUM,
    SMALL_SAMPLE_REVIEW_SHARE,
    GateAction,
    GateLexicons,
    decide_merge,
    load_gate_lexicons,
)

FIXTURE_LEXICONS_DIRECTORY = Path(__file__).parent / "fixtures" / "merge_gate"
PRODUCTION_LEXICONS_DIRECTORY = Path(__file__).parent.parent / "lexicons"

DEFAULT_BASE_COUNT = 2000
DEFAULT_TOP_SHARE = 0.5


def _make_evidence(
    variant_count: int,
    jsd_bits: float,
    null95_bits: float,
    top_share: float = DEFAULT_TOP_SHARE,
) -> MergeEvidence:
    """Synthetic evidence with a consistent jsd-to-null ratio."""
    ratio = jsd_bits / null95_bits if null95_bits > 0 else 0.0
    return MergeEvidence(
        variant_count=variant_count,
        base_count=DEFAULT_BASE_COUNT,
        jsd_bits=jsd_bits,
        null95_bits=null95_bits,
        jsd_to_null_ratio=ratio,
        variant_top_cuisines=(CuisineShare("chinese", top_share, 2.0),),
        base_top_cuisines=(CuisineShare("chinese", 0.41, 1.5),),
    )


def _write_lexicon_directory(
    directory: Path,
    pattern_source: str = "\\btestmarker\\b",
) -> Path:
    """Write a minimal, structurally valid set of the four lexicon files."""
    files = {
        "always_merge_patterns.json": {
            "patterns": [{"pattern": pattern_source, "reason": "test"}]
        },
        "forced_merge_overrides.json": {
            "overrides": [{"reason": "test", "variant": "forced variant"}]
        },
        "named_varieties.json": {
            "varieties": [{"phrase": "testvariety", "reason": "test"}]
        },
        "do_not_merge.json": {
            "exceptions": [{"reason": "test", "variant": "kept variant"}]
        },
    }
    for file_name, payload in files.items():
        (directory / file_name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return directory


@pytest.fixture(scope="module")
def fixture_lexicons() -> GateLexicons:
    return load_gate_lexicons(FIXTURE_LEXICONS_DIRECTORY)


@pytest.fixture(scope="module")
def production_lexicons() -> GateLexicons:
    return load_gate_lexicons(PRODUCTION_LEXICONS_DIRECTORY)


# ---------------------------------------------------------------------------
# Constants guard the validated thresholds from the divergence analysis.
# ---------------------------------------------------------------------------


def test_gate_constants_match_validated_thresholds():
    assert MIN_SUPPORT == 20
    assert JSD_FLOOR_BITS == 0.07
    assert NULL_MULTIPLIER == 1.5
    assert BORDERLINE_JSD_BAND == (0.06, 0.08)
    assert BORDERLINE_RATIO_BAND == (1.3, 1.7)
    assert SMALL_SAMPLE_REVIEW_SHARE == 0.80
    assert SMALL_SAMPLE_REVIEW_MINIMUM == 5


# ---------------------------------------------------------------------------
# Lexicon loading.
# ---------------------------------------------------------------------------


def test_load_gate_lexicons_returns_frozen_structures(fixture_lexicons):
    assert isinstance(fixture_lexicons.always_merge_patterns, tuple)
    assert all(
        hasattr(pattern, "search")
        for pattern in fixture_lexicons.always_merge_patterns
    )
    assert isinstance(fixture_lexicons.forced_merge_overrides, frozenset)
    assert isinstance(fixture_lexicons.named_varieties, frozenset)
    assert isinstance(fixture_lexicons.do_not_merge, frozenset)
    assert "green chilies" in fixture_lexicons.do_not_merge


def test_load_gate_lexicons_missing_directory_raises(tmp_path):
    missing_directory = tmp_path / "no_such_lexicons"

    with pytest.raises(FileNotFoundError):
        load_gate_lexicons(missing_directory)


def test_load_gate_lexicons_rejects_malformed_entry_list(tmp_path):
    _write_lexicon_directory(tmp_path)
    bad_payload = {"patterns": "not a list"}
    (tmp_path / "always_merge_patterns.json").write_text(
        json.dumps(bad_payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_gate_lexicons(tmp_path)


def test_load_gate_lexicons_rejects_invalid_regex(tmp_path):
    _write_lexicon_directory(tmp_path, pattern_source="(")

    with pytest.raises(ValueError):
        load_gate_lexicons(tmp_path)


# ---------------------------------------------------------------------------
# L0: do-not-merge exceptions.
# ---------------------------------------------------------------------------


def test_do_not_merge_keeps_green_chilies(fixture_lexicons):
    # Merge-looking evidence must not matter: L0 fires before everything.
    evidence = _make_evidence(variant_count=768, jsd_bits=0.01, null95_bits=0.01)

    decision = decide_merge("green chilies", "green chile", evidence, fixture_lexicons)

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "do_not_merge_exception"
    assert decision.evidence is evidence


def test_do_not_merge_beats_always_merge_pattern(fixture_lexicons):
    # Fixture entry "reduced sodium tamari" also matches the sodium pattern;
    # the exception layer must win.
    decision = decide_merge(
        "reduced sodium tamari", "tamari soy sauce", None, fixture_lexicons
    )

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "do_not_merge_exception"


# ---------------------------------------------------------------------------
# L1: always-merge patterns and forced overrides beat all statistics.
# ---------------------------------------------------------------------------


def test_lexicon_layer_beats_statistics_for_lower_sodium(fixture_lexicons):
    # Real case: n=57, JSD=0.1449 (p<0.001) is a recipe-source artifact;
    # the sodium lexicon must force MERGE despite a huge divergence.
    evidence = _make_evidence(variant_count=57, jsd_bits=0.1449, null95_bits=0.020)

    decision = decide_merge(
        "lower sodium soy sauce", "soy sauce", evidence, fixture_lexicons
    )

    assert decision.action is GateAction.MERGE
    assert decision.layer == "always_merge_lexicon"
    assert decision.reason


def test_statistical_merge_low_sodium_despite_significance(fixture_lexicons):
    # Real case: n=425 is significant at p=0.0012 yet JSD=0.0181; the gate
    # must MERGE (here via the sodium pattern, which short-circuits stats).
    evidence = _make_evidence(variant_count=425, jsd_bits=0.0181, null95_bits=0.0136)

    decision = decide_merge(
        "low sodium soy sauce", "soy sauce", evidence, fixture_lexicons
    )

    assert decision.action is GateAction.MERGE


def test_forced_merge_override_matches_exact_string(fixture_lexicons):
    evidence = _make_evidence(variant_count=100, jsd_bits=0.30, null95_bits=0.02)

    decision = decide_merge(
        "mystery sauce blend", "soy sauce", evidence, fixture_lexicons
    )

    assert decision.action is GateAction.MERGE
    assert decision.layer == "forced_merge_override"


# ---------------------------------------------------------------------------
# L2: statistical gate at n >= MIN_SUPPORT.
# ---------------------------------------------------------------------------


def test_statistical_preserve_dark_soy_sauce(fixture_lexicons):
    # Real case: n=312, JSD=0.1105, null95=0.0174 (6.4x) -> clean PRESERVE.
    evidence = _make_evidence(variant_count=312, jsd_bits=0.1105, null95_bits=0.0174)

    decision = decide_merge("dark soy sauce", "soy sauce", evidence, fixture_lexicons)

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "statistical_gate"


def test_pure_statistical_merge_below_jsd_floor(fixture_lexicons):
    # Real case shape: canola oil vs vegetable oil, JSD=0.0156 (0.87x null).
    evidence = _make_evidence(variant_count=300, jsd_bits=0.0156, null95_bits=0.018)

    decision = decide_merge("canola oil", "vegetable oil", evidence, fixture_lexicons)

    assert decision.action is GateAction.MERGE
    assert decision.layer == "statistical_gate"


def test_borderline_jsd_band_routes_to_review(fixture_lexicons):
    # JSD inside (0.06, 0.08) is flip-sensitive: REVIEW even though the
    # clean verdict (below floor with a huge ratio) would otherwise apply.
    evidence = _make_evidence(variant_count=100, jsd_bits=0.065, null95_bits=0.010)

    decision = decide_merge("white vinegar", "vinegar", evidence, fixture_lexicons)

    assert decision.action is GateAction.REVIEW
    assert decision.layer == "statistical_gate"


def test_borderline_ratio_band_routes_to_review(fixture_lexicons):
    # Ratio 1.5 sits inside (1.3, 1.7) although JSD=0.15 clears the floor.
    evidence = _make_evidence(variant_count=150, jsd_bits=0.15, null95_bits=0.10)

    decision = decide_merge("sherry wine vinegar", "sherry vinegar", evidence, fixture_lexicons)

    assert decision.action is GateAction.REVIEW
    assert decision.layer == "statistical_gate"


def test_borderline_review_wins_over_clean_preserve(fixture_lexicons):
    # Tamari numbers: JSD=0.0771 at 1.6x null pass the preserve test but sit
    # inside BOTH borderline bands -> REVIEW must win.
    evidence = _make_evidence(variant_count=90, jsd_bits=0.0771, null95_bits=0.0482)

    decision = decide_merge("tamari sauce", "soy sauce", evidence, fixture_lexicons)

    assert decision.action is GateAction.REVIEW
    assert decision.layer == "statistical_gate"


def test_light_is_not_always_merged(production_lexicons):
    # Real case: light soy sauce (n=347, JSD=0.0992, 6.2x null) is a Chinese
    # variety. Production patterns must not contain a bare "light" rule, so
    # the case reaches L2 and PRESERVEs.
    evidence = _make_evidence(variant_count=347, jsd_bits=0.0992, null95_bits=0.016)

    decision = decide_merge("light soy sauce", "soy sauce", evidence, production_lexicons)

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "statistical_gate"


# ---------------------------------------------------------------------------
# L3: small-sample layer (n < MIN_SUPPORT or no evidence).
# ---------------------------------------------------------------------------


def test_named_variety_preserves_usukuchi_at_n_one(fixture_lexicons):
    evidence = _make_evidence(
        variant_count=1, jsd_bits=0.650, null95_bits=0.0, top_share=1.0
    )

    decision = decide_merge(
        "usukuchi soy sauce", "soy sauce", evidence, fixture_lexicons
    )

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "named_variety_lexicon"


def test_named_variety_preserved_without_evidence(fixture_lexicons):
    decision = decide_merge("kecap manis", "soy sauce", None, fixture_lexicons)

    assert decision.action is GateAction.PRESERVE
    assert decision.layer == "named_variety_lexicon"


def test_named_variety_requires_word_boundary(fixture_lexicons):
    # Fixture lexicon lists "tamari" as a variety; "tamarind paste" must NOT
    # match it (naive substring matching corrupts the tamarind family).
    decision = decide_merge("tamarind paste", "tamarind", None, fixture_lexicons)

    assert decision.action is GateAction.MERGE
    assert decision.layer == "small_sample_default"


def test_small_n_high_concentration_routes_to_review(fixture_lexicons):
    evidence = _make_evidence(
        variant_count=10, jsd_bits=0.40, null95_bits=0.28, top_share=0.9
    )

    decision = decide_merge("obscure chile blend", "chili powder", evidence, fixture_lexicons)

    assert decision.action is GateAction.REVIEW
    assert decision.layer == "small_sample_review"


def test_small_n_default_merges(fixture_lexicons):
    evidence = _make_evidence(
        variant_count=14, jsd_bits=0.05, null95_bits=0.22, top_share=0.4
    )

    decision = decide_merge("mild curry powder", "curry powder", evidence, fixture_lexicons)

    assert decision.action is GateAction.MERGE
    assert decision.layer == "small_sample_default"


def test_small_n_below_review_minimum_merges_despite_concentration(fixture_lexicons):
    evidence = _make_evidence(
        variant_count=3, jsd_bits=0.50, null95_bits=0.40, top_share=0.95
    )

    decision = decide_merge("rare spice mix", "spice mix", evidence, fixture_lexicons)

    assert decision.action is GateAction.MERGE
    assert decision.layer == "small_sample_default"


def test_no_evidence_defaults_to_small_sample_merge(fixture_lexicons):
    decision = decide_merge("unseen test string", "base string", None, fixture_lexicons)

    assert decision.action is GateAction.MERGE
    assert decision.layer == "small_sample_default"
    assert decision.evidence is None


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_decide_merge_rejects_empty_variant(fixture_lexicons):
    with pytest.raises(ValueError):
        decide_merge("", "soy sauce", None, fixture_lexicons)


def test_decide_merge_rejects_blank_base(fixture_lexicons):
    with pytest.raises(ValueError):
        decide_merge("dark soy sauce", "   ", None, fixture_lexicons)


def test_decide_merge_rejects_wrong_evidence_type(fixture_lexicons):
    with pytest.raises(TypeError):
        decide_merge("dark soy sauce", "soy sauce", {"jsd_bits": 0.1}, fixture_lexicons)


# ---------------------------------------------------------------------------
# Production lexicons: structural validity and validated spot cases.
# ---------------------------------------------------------------------------


def test_production_lexicons_cover_required_entries(production_lexicons):
    assert "lower sodium soy sauce" in production_lexicons.forced_merge_overrides
    assert "light brown sugar" in production_lexicons.forced_merge_overrides
    for variety in ("usukuchi", "kecap manis", "dende"):
        assert variety in production_lexicons.named_varieties
    for exception in (
        "green chilies",
        "frozen spinach",
        "fresh mozzarella",
        "sweet soy sauce",
    ):
        assert exception in production_lexicons.do_not_merge


def test_production_patterns_match_marketing_not_identity(production_lexicons):
    def has_pattern_match(text: str) -> bool:
        return any(
            pattern.search(text)
            for pattern in production_lexicons.always_merge_patterns
        )

    for marketing_string in (
        "low sodium soy sauce",
        "fat free less sodium chicken broth",
        "lite coconut milk",
        "no salt added diced tomatoes",
        "gluten free soy sauce",
        "33% less sodium ham",
        "regular soy sauce",
        "store bought low sodium chicken stock",
        "homemade chicken stock",
    ):
        assert has_pattern_match(marketing_string), marketing_string
    for identity_string in (
        "light soy sauce",
        "light brown sugar",
        "light rum",
        "dark soy sauce",
        "sweet soy sauce",
    ):
        assert not has_pattern_match(identity_string), identity_string


def test_production_gate_replays_validated_cases(production_lexicons):
    lower_sodium = decide_merge(
        "lower sodium soy sauce",
        "soy sauce",
        _make_evidence(variant_count=57, jsd_bits=0.1449, null95_bits=0.020),
        production_lexicons,
    )
    sweet_soy = decide_merge(
        "sweet soy sauce",
        "soy sauce",
        _make_evidence(variant_count=29, jsd_bits=0.1144, null95_bits=0.124),
        production_lexicons,
    )
    dende_oil = decide_merge(
        "dende oil",
        "vegetable oil",
        _make_evidence(variant_count=7, jsd_bits=0.9, null95_bits=0.0, top_share=1.0),
        production_lexicons,
    )
    hot_smoked_paprika = decide_merge(
        "hot smoked paprika",
        "smoked paprika",
        _make_evidence(variant_count=13, jsd_bits=0.2, null95_bits=0.0, top_share=0.54),
        production_lexicons,
    )
    smoked_paprika = decide_merge(
        "smoked paprika",
        "paprika",
        _make_evidence(variant_count=345, jsd_bits=0.0777, null95_bits=0.011),
        production_lexicons,
    )

    assert lower_sodium.action is GateAction.MERGE
    assert sweet_soy.action is GateAction.PRESERVE
    assert sweet_soy.layer == "do_not_merge_exception"
    assert dende_oil.action is GateAction.PRESERVE
    assert dende_oil.layer == "named_variety_lexicon"
    assert hot_smoked_paprika.action is GateAction.PRESERVE
    assert hot_smoked_paprika.layer == "named_variety_lexicon"
    assert smoked_paprika.action is GateAction.PRESERVE
    assert smoked_paprika.layer == "do_not_merge_exception"
