"""Per-token singularization for stable ingredient lookup keys.

Converts already-cleaned ingredient text (lowercase, punctuation-free output
of normalize_text) into canonical singular-form keys for alias-table joins.
Suffix rules are guarded by a data-derived exception lexicon
(01-bronze/lexicons/singularize_exceptions.json) built from the 459 s-ending tokens in
the train vocabulary, so mass nouns ("molasses"), brands ("doritos"), and
foreign plurals ("herbes") survive unchanged while irregular plurals
("leaves" -> "leaf") map explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

NO_SINGULARIZE_KEY = "no_singularize"
IRREGULAR_KEY = "irregular"

IES_SUFFIX = "ies"
IES_REPLACEMENT = "y"
# 'ies' -> 'y' and 'ves' -> 've' only fire above this length ("pies" stays).
LONG_SUFFIX_MINIMUM_LENGTH = 5
VES_SUFFIX = "ves"
# 'ves' strips only the final 's' (chives -> chive); leaf/half/loaf live in
# the irregular map because a global 'ves' -> 'f' rule corrupts olives/cloves.
ES_STRIP_SUFFIXES = ("ches", "shes", "sses", "xes", "zes")
ES_STRIP_LENGTH = 2
FINAL_S_MINIMUM_LENGTH = 4
# Guard suffixes auto-protect couscous, hummus, asparagus, swiss, boneless.
FINAL_S_GUARD_SUFFIXES = ("ss", "us", "is")
TOKEN_SEPARATOR = " "


@dataclass(frozen=True)
class SingularizeExceptions:
    """Frozen exception lexicon guarding the suffix-based singularizer.

    Attributes:
        no_singularize: Tokens returned unchanged (mass nouns, brands,
            foreign plurals, upstream stem-junk like 'mayonnais').
        irregular: Explicit plural-to-singular overrides applied before any
            suffix rule (e.g. 'leaves' -> 'leaf', 'tomatoes' -> 'tomato').
    """

    no_singularize: frozenset[str]
    irregular: dict[str, str]


def load_singularize_exceptions(path: Path) -> SingularizeExceptions:
    """Load and validate the singularization exception lexicon.

    Args:
        path: Path to a UTF-8 JSON file with keys 'no_singularize'
            (list of tokens) and 'irregular' (plural-to-singular mapping).

    Returns:
        A frozen SingularizeExceptions ready to pass to singularize_token.

    Raises:
        FileNotFoundError: If the lexicon file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required keys are missing or values have wrong types.

    Examples:
        >>> exceptions = load_singularize_exceptions(
        ...     Path("01-bronze/lexicons/singularize_exceptions.json"))
        >>> "molasses" in exceptions.no_singularize
        True
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_lexicon_payload(payload, path)
    return SingularizeExceptions(
        no_singularize=frozenset(payload[NO_SINGULARIZE_KEY]),
        irregular=dict(payload[IRREGULAR_KEY]),
    )


def singularize_token(token: str, exceptions: SingularizeExceptions) -> str:
    """Singularize one cleaned token using guarded deterministic rules.

    Rule order: (1) no_singularize set, (2) irregular map, (3) 'ies' -> 'y',
    (4) 'ves' -> 've' (strip 's' only), (5) 'ches'/'shes'/'sses'/'xes'/'zes'
    -> strip 'es', (6) strip final 's' unless ending 'ss'/'us'/'is'.

    Args:
        token: A single lowercase token with no internal whitespace.
        exceptions: Lexicon from load_singularize_exceptions.

    Returns:
        The singular key form of the token; unchanged when no rule applies.

    Raises:
        ValueError: If the token is empty or contains whitespace.

    Examples:
        >>> singularize_token("leaves", exceptions)
        'leaf'
        >>> singularize_token("couscous", exceptions)
        'couscous'
    """
    if not token or any(character.isspace() for character in token):
        raise ValueError(
            f"singularize_token expects one non-empty token, got {token!r}"
        )
    if token in exceptions.no_singularize:
        return token
    if token in exceptions.irregular:
        return exceptions.irregular[token]
    return _apply_suffix_rules(token)


def make_lookup_key(
    cleaned_text: str, exceptions: SingularizeExceptions
) -> str:
    """Build a canonical lookup key by singularizing every token.

    The input must already be cleaned by normalize_text (lowercase, single
    spaces, no leading/trailing whitespace); this function never re-cleans
    and rejects text that violates the contract.

    Args:
        cleaned_text: Cleaned ingredient text, e.g. 'bay leaves'.
        exceptions: Lexicon from load_singularize_exceptions.

    Returns:
        Space-joined singularized tokens, e.g. 'bay leaf'.

    Raises:
        ValueError: If cleaned_text is empty, contains uppercase characters,
            or has leading/trailing/doubled spaces.

    Examples:
        >>> make_lookup_key("bay leaves", exceptions)
        'bay leaf'
    """
    _validate_cleaned_text(cleaned_text)
    tokens = cleaned_text.split(TOKEN_SEPARATOR)
    singular_tokens = [
        singularize_token(token, exceptions) for token in tokens
    ]
    return TOKEN_SEPARATOR.join(singular_tokens)


def _apply_suffix_rules(token: str) -> str:
    """Apply ordered plural-suffix rules 3-6 to one unexcepted token."""
    is_long_enough = len(token) >= LONG_SUFFIX_MINIMUM_LENGTH
    if is_long_enough and token.endswith(IES_SUFFIX):
        return token[: -len(IES_SUFFIX)] + IES_REPLACEMENT
    if is_long_enough and token.endswith(VES_SUFFIX):
        return token[:-1]
    if is_long_enough and token.endswith(ES_STRIP_SUFFIXES):
        return token[:-ES_STRIP_LENGTH]
    has_strippable_s = (
        len(token) >= FINAL_S_MINIMUM_LENGTH
        and token.endswith("s")
        and not token.endswith(FINAL_S_GUARD_SUFFIXES)
    )
    if has_strippable_s:
        return token[:-1]
    return token


def _validate_lexicon_payload(payload: object, path: Path) -> None:
    """Raise ValueError unless the payload matches the lexicon schema."""
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: lexicon root must be a JSON object")
    for required_key in (NO_SINGULARIZE_KEY, IRREGULAR_KEY):
        if required_key not in payload:
            raise ValueError(f"{path}: missing required key {required_key!r}")
    no_singularize = payload[NO_SINGULARIZE_KEY]
    if not isinstance(no_singularize, list) or not all(
        isinstance(token, str) for token in no_singularize
    ):
        raise ValueError(
            f"{path}: {NO_SINGULARIZE_KEY!r} must be a list of strings"
        )
    irregular = payload[IRREGULAR_KEY]
    if not isinstance(irregular, dict) or not all(
        isinstance(plural, str) and isinstance(singular, str)
        for plural, singular in irregular.items()
    ):
        raise ValueError(
            f"{path}: {IRREGULAR_KEY!r} must map strings to strings"
        )


def _validate_cleaned_text(cleaned_text: str) -> None:
    """Raise ValueError if text violates the cleaned-input contract."""
    if not cleaned_text:
        raise ValueError("make_lookup_key requires non-empty cleaned text")
    if cleaned_text != cleaned_text.lower():
        raise ValueError(
            f"make_lookup_key requires lowercase input, got {cleaned_text!r}"
        )
    has_malformed_spacing = any(
        token == "" for token in cleaned_text.split(TOKEN_SEPARATOR)
    )
    if has_malformed_spacing:
        raise ValueError(
            "make_lookup_key requires single-spaced text without leading or "
            f"trailing spaces, got {cleaned_text!r}"
        )
