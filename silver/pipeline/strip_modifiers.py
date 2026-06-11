"""Strip cuisine-neutral modifiers from cleaned ingredient text.

Implements the SAFE-STRIP side of the descriptors analysis lens over
bronze/kaggle/train.json: marketing phrases (low sodium, fat free), prep
tokens (chopped, minced), size/grade words, and part words are removed,
while identity-bearing modifiers (ground, dried, baby, roasted, ...)
are protected via a never-strip list, per-token conditional guards
(kosher/whole/packed/extra), and a protected exact-string list.

Lexicon data lives in silver/lexicons/modifier_strip.json and is loaded
explicitly through load_modifier_lexicon — no import-time file I/O.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Guard modes for conditional tokens (lexicon "conditional_tokens" entries).
GUARD_MODE_STRIP_WHEN_HEAD_IN = "strip_when_head_in"
GUARD_MODE_STRIP_WHEN_NEXT_IN = "strip_when_next_in"
GUARD_MODE_KEEP_WHEN_NEXT_IN = "keep_when_next_in"
KNOWN_GUARD_MODES = frozenset(
    (
        GUARD_MODE_STRIP_WHEN_HEAD_IN,
        GUARD_MODE_STRIP_WHEN_NEXT_IN,
        GUARD_MODE_KEEP_WHEN_NEXT_IN,
    )
)

REQUIRED_LEXICON_KEYS = (
    "conditional_tokens",
    "never_strip_tokens",
    "protected_strings",
    "strip_phrases",
    "strip_tokens",
)
REQUIRED_GUARD_KEYS = ("mode", "values")

PHRASE_REPLACEMENT = " "
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class ConditionalGuard:
    """Strip condition for a single modifier token.

    Attributes:
        token: The modifier token this guard applies to, e.g. "kosher".
        mode: One of KNOWN_GUARD_MODES controlling when the token is
            stripped (by head noun, by next token, or kept by next token).
        values: Comparison tokens for the mode, e.g. {"salt"} for kosher.
    """

    token: str
    mode: str
    values: frozenset[str]


@dataclass(frozen=True)
class ModifierLexicon:
    """Frozen lexicon driving strip_safe_modifiers.

    Attributes:
        strip_phrases: Regex source strings removed before tokenization,
            in application order (they span stacked marketing phrases).
        strip_tokens: Tokens dropped unconditionally during the token pass.
        conditional_tokens: Guards for tokens that are only sometimes safe
            to strip, sorted by token for determinism.
        never_strip_tokens: Tokens that must never be dropped; overrides
            strip_tokens and conditional_tokens.
        protected_strings: Exact cleaned strings returned unchanged.
    """

    strip_phrases: tuple[str, ...]
    strip_tokens: frozenset[str]
    conditional_tokens: tuple[ConditionalGuard, ...]
    never_strip_tokens: frozenset[str]
    protected_strings: frozenset[str]


def load_modifier_lexicon(path: str | Path) -> ModifierLexicon:
    """Load and validate a modifier-strip lexicon from a JSON file.

    Args:
        path: Path to a UTF-8 JSON file with keys strip_phrases (list of
            regex sources), strip_tokens, conditional_tokens (mapping of
            token -> {mode, values}), never_strip_tokens, protected_strings.

    Returns:
        A frozen ModifierLexicon with guards sorted by token.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required keys are missing, a phrase regex does not
            compile, or a guard uses an unknown mode.

    Examples:
        >>> lexicon = load_modifier_lexicon("silver/lexicons/modifier_strip.json")
        >>> strip_safe_modifiers("minced garlic", lexicon)
        'garlic'
    """
    with open(path, encoding="utf-8") as lexicon_file:
        raw_lexicon = json.load(lexicon_file)
    _validate_required_keys(raw_lexicon, path)
    _validate_phrases_compile(raw_lexicon["strip_phrases"], path)
    guards = tuple(
        _to_conditional_guard(token, guard_spec, path)
        for token, guard_spec in sorted(raw_lexicon["conditional_tokens"].items())
    )
    return ModifierLexicon(
        strip_phrases=tuple(raw_lexicon["strip_phrases"]),
        strip_tokens=frozenset(raw_lexicon["strip_tokens"]),
        conditional_tokens=guards,
        never_strip_tokens=frozenset(raw_lexicon["never_strip_tokens"]),
        protected_strings=frozenset(raw_lexicon["protected_strings"]),
    )


def strip_safe_modifiers(cleaned_text: str, lexicon: ModifierLexicon) -> str:
    """Remove cuisine-neutral modifier phrases and tokens from cleaned text.

    Args:
        cleaned_text: Already-cleaned ingredient text (lowercase, hyphens
            and other punctuation folded to spaces upstream).
        lexicon: Lexicon returned by load_modifier_lexicon.

    Returns:
        Text with safe modifiers removed; the input unchanged when it is a
        protected string or when stripping would leave nothing.

    Raises:
        TypeError: If cleaned_text is not a string.

    Examples:
        >>> strip_safe_modifiers("chopped cilantro fresh", lexicon)
        'cilantro'
        >>> strip_safe_modifiers("fat free less sodium chicken broth", lexicon)
        'chicken broth'
    """
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"cleaned_text must be str, got {type(cleaned_text).__name__}"
        )
    if cleaned_text in lexicon.protected_strings:
        return cleaned_text
    without_phrases = _remove_strip_phrases(cleaned_text, lexicon.strip_phrases)
    kept_tokens = _remove_strip_tokens(without_phrases.split(), lexicon)
    if not kept_tokens:
        return cleaned_text
    return " ".join(kept_tokens)


def _validate_required_keys(raw_lexicon: dict, path: str | Path) -> None:
    """Raise ValueError naming any missing top-level lexicon keys."""
    missing_keys = [key for key in REQUIRED_LEXICON_KEYS if key not in raw_lexicon]
    if missing_keys:
        raise ValueError(
            f"modifier lexicon {path} is missing keys: {', '.join(missing_keys)}"
        )


def _validate_phrases_compile(phrases: list[str], path: str | Path) -> None:
    """Raise ValueError if any strip phrase is not a valid regex."""
    for phrase in phrases:
        try:
            re.compile(phrase)
        except re.error as compile_error:
            raise ValueError(
                f"modifier lexicon {path} has invalid strip phrase "
                f"{phrase!r}: {compile_error}"
            ) from compile_error


def _to_conditional_guard(
    token: str, guard_spec: dict, path: str | Path
) -> ConditionalGuard:
    """Build one validated ConditionalGuard from its JSON specification."""
    missing_keys = [key for key in REQUIRED_GUARD_KEYS if key not in guard_spec]
    if missing_keys:
        raise ValueError(
            f"modifier lexicon {path} guard {token!r} is missing keys: "
            f"{', '.join(missing_keys)}"
        )
    if guard_spec["mode"] not in KNOWN_GUARD_MODES:
        raise ValueError(
            f"modifier lexicon {path} guard {token!r} has unknown guard mode "
            f"{guard_spec['mode']!r}; expected one of {sorted(KNOWN_GUARD_MODES)}"
        )
    return ConditionalGuard(
        token=token,
        mode=guard_spec["mode"],
        values=frozenset(guard_spec["values"]),
    )


def _remove_strip_phrases(text: str, phrases: tuple[str, ...]) -> str:
    """Delete each strip phrase regex in order, then normalize whitespace."""
    for phrase in phrases:
        text = re.sub(phrase, PHRASE_REPLACEMENT, text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _remove_strip_tokens(tokens: list[str], lexicon: ModifierLexicon) -> list[str]:
    """Drop strip-listed tokens, honoring never-strip and conditional guards."""
    guards_by_token = {guard.token: guard for guard in lexicon.conditional_tokens}
    is_dropped = [
        token in lexicon.strip_tokens
        and token not in lexicon.never_strip_tokens
        and token not in guards_by_token
        for token in tokens
    ]
    for index, token in enumerate(tokens):
        if token in guards_by_token and token not in lexicon.never_strip_tokens:
            is_dropped[index] = _is_guarded_strip(
                guards_by_token[token], tokens, index, is_dropped
            )
    return [token for token, dropped in zip(tokens, is_dropped) if not dropped]


def _is_guarded_strip(
    guard: ConditionalGuard, tokens: list[str], index: int, is_dropped: list[bool]
) -> bool:
    """Decide whether the conditional token at index should be dropped."""
    has_next_token = index + 1 < len(tokens)
    next_token = tokens[index + 1] if has_next_token else ""
    if guard.mode == GUARD_MODE_STRIP_WHEN_NEXT_IN:
        return next_token in guard.values
    if guard.mode == GUARD_MODE_KEEP_WHEN_NEXT_IN:
        return next_token not in guard.values
    return _head_token(tokens, index, is_dropped) in guard.values


def _head_token(tokens: list[str], excluded_index: int, is_dropped: list[bool]) -> str:
    """Return the last surviving token other than the one under evaluation.

    The head noun is what remains after unconditional strips, e.g. "salt"
    in "kosher salt" — used by strip_when_head_in guards.
    """
    for index in range(len(tokens) - 1, -1, -1):
        if index == excluded_index or is_dropped[index]:
            continue
        return tokens[index]
    return ""
