// Pure blend scorer — the browser half of the JS-Python scoring contract.
// Mirrors app_pipeline/score_blend.py operation for operation: ascending
// feature-index summation for logits, left-to-right for the softmax
// normalizer, so the shipped contract vectors reproduce exactly.

const DIRECT_PRESENCE_VALUE = 1.0;

// Map ingredient ids to feature values with max-semantics parent back-off.
// Unknown ids are ignored, duplicates collapse, and a parent only
// contributes parentWeight when it is not itself present.
export function buildFeatureValues(resolvedIds, featureIds, parentIndices, parentWeight) {
  const indexById = new Map(featureIds.map((id, index) => [id, index]));
  const directIndices = new Set();
  for (const ingredientId of resolvedIds) {
    if (indexById.has(ingredientId)) {
      directIndices.add(indexById.get(ingredientId));
    }
  }
  const featureValues = new Map();
  for (const index of directIndices) {
    featureValues.set(index, DIRECT_PRESENCE_VALUE);
  }
  if (parentWeight <= 0.0) {
    return featureValues;
  }
  for (const index of [...directIndices].sort((left, right) => left - right)) {
    const parentIndex = parentIndices[index];
    if (parentIndex !== null && !directIndices.has(parentIndex)) {
      featureValues.set(parentIndex, parentWeight);
    }
  }
  return featureValues;
}

// Compute one recipe's per-cuisine logits. Summation runs in ascending
// feature-index order — the cross-language contract.
export function computeLogits(featureValues, coefficients, intercepts) {
  const orderedIndices = [...featureValues.keys()].sort((left, right) => left - right);
  return coefficients.map((cuisineRow, cuisinePosition) => {
    let total = intercepts[cuisinePosition];
    for (const index of orderedIndices) {
      total += featureValues.get(index) * cuisineRow[index];
    }
    return total;
  });
}

// Convert logits to a calibrated blend via temperature-scaled softmax.
export function convertLogitsToBlend(logits, temperature) {
  const scaled = logits.map((logit) => logit / temperature);
  const peak = Math.max(...scaled);
  const exponentials = scaled.map((value) => Math.exp(value - peak));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

// Order cuisines by descending share, cuisine id breaking ties.
export function rankBlendEntries(blend, cuisineIds) {
  return blend
    .map((share, position) => ({ cuisine: cuisineIds[position], share }))
    .sort((left, right) =>
      right.share - left.share || (left.cuisine < right.cuisine ? -1 : 1),
    );
}
