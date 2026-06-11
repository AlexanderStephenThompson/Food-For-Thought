"""Shared evidence types exchanged between the divergence harness and the merge gate.

pipeline.cuisine_divergence produces MergeEvidence for each (variant, base)
merge candidate; pipeline.merge_gate consumes it to decide merge or preserve,
and the review queue serializes it for human/LLM judgment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuisineShare:
    """Share of one cuisine within an ingredient's recipe distribution.

    Attributes:
        cuisine: Cuisine identifier, e.g. "thai".
        share: Fraction of the ingredient's recipes labeled with this cuisine (0-1).
        lift: share divided by the cuisine's overall share of all train recipes.
    """

    cuisine: str
    share: float
    lift: float


@dataclass(frozen=True)
class MergeEvidence:
    """Statistical evidence for one (variant, base) merge candidate.

    Attributes:
        variant_count: Number of train recipes containing the variant.
        base_count: Number of train recipes containing the base.
        jsd_bits: Jensen-Shannon divergence (log base 2) between the variant's
            and the base's cuisine distributions.
        null95_bits: 95th percentile JSD expected under the null hypothesis
            that the variant is a random sample of the base's recipes.
        jsd_to_null_ratio: jsd_bits / null95_bits, or 0.0 when null95_bits is 0.
        variant_top_cuisines: Highest-share cuisines for the variant, descending.
        base_top_cuisines: Highest-share cuisines for the base, descending.
    """

    variant_count: int
    base_count: int
    jsd_bits: float
    null95_bits: float
    jsd_to_null_ratio: float
    variant_top_cuisines: tuple[CuisineShare, ...]
    base_top_cuisines: tuple[CuisineShare, ...]
