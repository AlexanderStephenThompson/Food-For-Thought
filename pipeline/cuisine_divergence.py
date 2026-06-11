"""Jensen-Shannon divergence harness for (variant, base) ingredient merges.

Measures how differently two groups of raw ingredient strings distribute over
the 20 train cuisines, and calibrates that divergence against a Monte Carlo
multinomial null (what JSD a random sample of the variant's size drawn from
the base distribution would show). Produces MergeEvidence consumed by the
merge gate; the informational CLI verdict here mirrors the gate thresholds
but the gate module owns the actual decision.

Method and thresholds follow the signal-variants analysis: PRESERVE requires
support >= 20, JSD >= 0.07 bits, and JSD >= 1.5x the n-matched null95.

CLI usage:
    python -m pipeline.cuisine_divergence --variant "dark soy sauce" \\
        --base "soy sauce"
"""

import argparse
import math
import random
from collections import Counter
from collections.abc import Collection
from itertools import accumulate

from pipeline.load_bronze_recipes import TrainIndex, build_train_index, load_train_recipes
from pipeline.merge_evidence import CuisineShare, MergeEvidence

# Merge-gate thresholds validated in the signal-variants analysis
# (dark soy 0.1105 bits / 6.4x null -> PRESERVE; low sodium 0.0181 -> MERGE).
MIN_SUPPORT = 20
JSD_FLOOR_BITS = 0.07
NULL_MULTIPLIER = 1.5

VERDICT_PRESERVE = "PRESERVE"
VERDICT_MERGE = "MERGE"
VERDICT_REVIEW = "REVIEW"

DEFAULT_TRIALS = 1_000
DEFAULT_SEED = 42
NULL95_PERCENTILE = 0.95
TOP_CUISINE_COUNT = 5
# Memo-key rounding: distributions equal to 6 decimals share a null95.
MEMO_KEY_DIGITS = 6

_NULL95_CACHE: dict[tuple[tuple[float, ...], int, int, int], float] = {}


def compute_cuisine_distribution(
    raw_strings: Collection[str], index: TrainIndex
) -> tuple[float, ...]:
    """Compute the cuisine distribution of recipes containing any given string.

    The strings are treated as ONE group: the distribution is taken over the
    union of recipe ids matching any string, so a recipe containing several of
    the strings is counted once (recipe-level, not mention-level).

    Args:
        raw_strings: Raw ingredient strings (whole-string matches only).
        index: Train index providing string -> recipe-id lookups.

    Returns:
        Shares over index.cuisine_names (same order), summing to 1.0; all
        zeros when no recipe contains any of the strings.

    Raises:
        TypeError: If raw_strings is a bare string instead of a collection.
    """
    recipe_ids = _recipe_ids_for_strings(raw_strings, index)
    return _distribution_from_recipe_ids(recipe_ids, index)


def jensen_shannon_divergence_bits(
    p: tuple[float, ...], q: tuple[float, ...]
) -> float:
    """Compute the Jensen-Shannon divergence between two distributions in bits.

    Symmetric, bounded in [0, 1] with log base 2; zero-probability entries
    contribute nothing (0 * log 0 == 0).

    Args:
        p: First distribution.
        q: Second distribution, same length as p.

    Returns:
        Divergence in bits: 0.0 for identical inputs, 1.0 for disjoint support.

    Raises:
        ValueError: If lengths differ or any entry is negative.
    """
    if len(p) != len(q):
        raise ValueError(f"distribution lengths differ: {len(p)} vs {len(q)}")
    if any(value < 0.0 for value in p) or any(value < 0.0 for value in q):
        raise ValueError("distributions must not contain negative probabilities")
    midpoint = tuple((p_value + q_value) / 2.0 for p_value, q_value in zip(p, q))
    divergence = (
        _kullback_leibler_bits(p, midpoint) + _kullback_leibler_bits(q, midpoint)
    ) / 2.0
    # Floating-point noise can dip a hair below zero for identical inputs.
    return max(0.0, divergence)


def monte_carlo_null95(
    base_distribution: tuple[float, ...],
    sample_size: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Estimate the 95th-percentile JSD expected under the null hypothesis.

    Each trial draws a multinomial sample of sample_size from
    base_distribution (cumulative-weight inversion) and measures the JSD
    between the sample's empirical distribution and the base. Results are
    memoized by (rounded base distribution, sample_size, trials, seed) since
    the vocabulary build re-tests many variants against the same base.

    Args:
        base_distribution: Null distribution to sample from; positive mass.
        sample_size: Recipes in the variant group (multinomial draw size).
        trials: Monte Carlo repetitions; 1,000 is stable across seeds.
        seed: Seed for random.Random, making results reproducible.

    Returns:
        The 95th-percentile JSD (bits) across trials.

    Raises:
        ValueError: If sample_size or trials is below 1, or the base
            distribution has no positive mass.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if sum(base_distribution) <= 0.0:
        raise ValueError("base distribution must have positive total mass")
    memo_key = (
        tuple(round(value, MEMO_KEY_DIGITS) for value in base_distribution),
        sample_size,
        trials,
        seed,
    )
    if memo_key in _NULL95_CACHE:
        return _NULL95_CACHE[memo_key]
    generator = random.Random(seed)
    cumulative_weights = list(accumulate(base_distribution))
    divergences = sorted(
        jensen_shannon_divergence_bits(
            base_distribution,
            _sample_distribution(generator, cumulative_weights, sample_size),
        )
        for _ in range(trials)
    )
    percentile_position = min(trials - 1, math.ceil(NULL95_PERCENTILE * trials) - 1)
    _NULL95_CACHE[memo_key] = divergences[percentile_position]
    return _NULL95_CACHE[memo_key]


def evaluate_merge_candidate(
    variant_strings: Collection[str],
    base_strings: Collection[str],
    index: TrainIndex,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> MergeEvidence:
    """Assemble divergence evidence for one (variant, base) merge candidate.

    Args:
        variant_strings: Raw strings forming the variant group.
        base_strings: Raw strings forming the base group.
        index: Train index built from the labeled recipes.
        trials: Monte Carlo trials for the null calibration.
        seed: Seed for the Monte Carlo generator.

    Returns:
        MergeEvidence with recipe counts, JSD in bits, the n-matched null95,
        their ratio (0.0 when null95 is 0), and the top cuisines by share
        (with lift against each cuisine's overall train share) for both sides.

    Raises:
        TypeError: If either string group is a bare string.
    """
    variant_recipe_ids = _recipe_ids_for_strings(variant_strings, index)
    base_recipe_ids = _recipe_ids_for_strings(base_strings, index)
    variant_distribution = _distribution_from_recipe_ids(variant_recipe_ids, index)
    base_distribution = _distribution_from_recipe_ids(base_recipe_ids, index)
    jsd_bits = jensen_shannon_divergence_bits(variant_distribution, base_distribution)
    null95_bits = 0.0
    if variant_recipe_ids and base_recipe_ids:
        null95_bits = monte_carlo_null95(
            base_distribution, len(variant_recipe_ids), trials, seed
        )
    ratio = jsd_bits / null95_bits if null95_bits > 0.0 else 0.0
    return MergeEvidence(
        variant_count=len(variant_recipe_ids),
        base_count=len(base_recipe_ids),
        jsd_bits=jsd_bits,
        null95_bits=null95_bits,
        jsd_to_null_ratio=ratio,
        variant_top_cuisines=_top_cuisine_shares(variant_distribution, index),
        base_top_cuisines=_top_cuisine_shares(base_distribution, index),
    )


def classify_merge_verdict(evidence: MergeEvidence) -> str:
    """Classify evidence with the informational PRESERVE/MERGE/REVIEW rule.

    Mirrors layer 2 of the three-layer gate: PRESERVE iff support >= 20 AND
    JSD >= 0.07 bits AND JSD >= 1.5x null95; below-support candidates go to
    REVIEW (lexicon layers own them). The merge-gate module owns the real
    decision; this verdict only annotates CLI output.

    Args:
        evidence: Divergence evidence for one (variant, base) candidate.

    Returns:
        One of "PRESERVE", "MERGE", or "REVIEW".
    """
    if evidence.variant_count < MIN_SUPPORT:
        return VERDICT_REVIEW
    is_above_floor = evidence.jsd_bits >= JSD_FLOOR_BITS
    is_above_null = evidence.jsd_bits >= NULL_MULTIPLIER * evidence.null95_bits
    if is_above_floor and is_above_null:
        return VERDICT_PRESERVE
    return VERDICT_MERGE


def _recipe_ids_for_strings(
    raw_strings: Collection[str], index: TrainIndex
) -> frozenset[int]:
    """Union the recipe ids containing any of the raw strings (whole-string)."""
    if isinstance(raw_strings, str):
        raise TypeError(
            "raw_strings must be a collection of strings, not a bare string; "
            f"got {raw_strings!r}"
        )
    recipe_ids: set[int] = set()
    for raw_string in raw_strings:
        recipe_ids |= index.string_to_recipe_ids.get(raw_string, frozenset())
    return frozenset(recipe_ids)


def _distribution_from_recipe_ids(
    recipe_ids: frozenset[int], index: TrainIndex
) -> tuple[float, ...]:
    """Share of each cuisine (in index.cuisine_names order) among the recipes."""
    if not recipe_ids:
        return tuple(0.0 for _ in index.cuisine_names)
    cuisine_counts = Counter(
        index.recipe_id_to_cuisine[recipe_id] for recipe_id in recipe_ids
    )
    total = len(recipe_ids)
    return tuple(cuisine_counts.get(name, 0) / total for name in index.cuisine_names)


def _kullback_leibler_bits(
    p: tuple[float, ...], reference: tuple[float, ...]
) -> float:
    """KL(p || reference) in bits, with 0 * log 0 treated as 0."""
    return sum(
        p_value * math.log2(p_value / reference_value)
        for p_value, reference_value in zip(p, reference)
        if p_value > 0.0
    )


def _sample_distribution(
    generator: random.Random,
    cumulative_weights: list[float],
    sample_size: int,
) -> tuple[float, ...]:
    """Draw one multinomial sample via cumulative weights; return its shares."""
    dimension = len(cumulative_weights)
    draws = generator.choices(
        range(dimension), cum_weights=cumulative_weights, k=sample_size
    )
    draw_counts = Counter(draws)
    return tuple(draw_counts.get(position, 0) / sample_size for position in range(dimension))


def _format_top_cuisines(shares: tuple[CuisineShare, ...]) -> str:
    """Render top cuisines as 'name share% (liftx)' joined by commas."""
    if not shares:
        return "(none)"
    return ", ".join(
        f"{entry.cuisine} {entry.share:.1%} ({entry.lift:.1f}x)" for entry in shares
    )


def _top_cuisine_shares(
    distribution: tuple[float, ...], index: TrainIndex
) -> tuple[CuisineShare, ...]:
    """Top cuisines by share (descending, names break ties), with lift."""
    entries = []
    for position, name in enumerate(index.cuisine_names):
        share = distribution[position]
        if share <= 0.0:
            continue
        prior = index.cuisine_recipe_counts[name] / index.recipe_count
        entries.append(CuisineShare(cuisine=name, share=share, lift=share / prior))
    entries.sort(key=lambda entry: (-entry.share, entry.cuisine))
    return tuple(entries[:TOP_CUISINE_COUNT])


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one (variant, base) divergence check."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.cuisine_divergence",
        description=(
            "Measure cuisine-distribution divergence between a variant and a "
            "base ingredient string on the real train data."
        ),
    )
    parser.add_argument("--variant", required=True, help="variant raw string")
    parser.add_argument("--base", required=True, help="base raw string")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    """Run the divergence check on the real train data and print a verdict."""
    arguments = _build_argument_parser().parse_args()
    index = build_train_index(load_train_recipes())
    evidence = evaluate_merge_candidate(
        {arguments.variant},
        {arguments.base},
        index,
        trials=arguments.trials,
        seed=arguments.seed,
    )
    print(f"variant '{arguments.variant}': n={evidence.variant_count}")
    print(f"base    '{arguments.base}': n={evidence.base_count}")
    print(f"jsd_bits={evidence.jsd_bits:.4f}")
    print(f"null95_bits={evidence.null95_bits:.4f} (n={evidence.variant_count})")
    print(f"jsd_to_null_ratio={evidence.jsd_to_null_ratio:.2f}")
    print(f"variant top cuisines: {_format_top_cuisines(evidence.variant_top_cuisines)}")
    print(f"base    top cuisines: {_format_top_cuisines(evidence.base_top_cuisines)}")
    verdict = classify_merge_verdict(evidence)
    print(
        f"verdict: {verdict} (informational; gate: n>={MIN_SUPPORT}, "
        f"jsd>={JSD_FLOOR_BITS}, jsd>={NULL_MULTIPLIER}x null95)"
    )


if __name__ == "__main__":
    main()
