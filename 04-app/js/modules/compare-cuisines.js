// Compare two cuisines by their coefficient differences. A positive
// delta means the ingredient pushes harder toward cuisine A; negative,
// toward B. Returns the strongest pull in each direction.

// Split the top coefficient gaps into toward-A and toward-B lists.
export function computeCoefficientDeltas(modelAsset, cuisineA, cuisineB, limit) {
  const cuisinePosition = new Map(
    modelAsset.cuisines.map((cuisine, position) => [cuisine, position]),
  );
  if (cuisineA === cuisineB) {
    return { towardA: [], towardB: [] };
  }
  const coefficientsA = modelAsset.coefficients[cuisinePosition.get(cuisineA)];
  const coefficientsB = modelAsset.coefficients[cuisinePosition.get(cuisineB)];

  const deltas = modelAsset.feature_ids.map((ingredientId, index) => ({
    ingredientId,
    delta: coefficientsA[index] - coefficientsB[index],
  }));
  const byId = (left, right) => (left.ingredientId < right.ingredientId ? -1 : 1);

  const towardA = [...deltas]
    .filter((entry) => entry.delta > 0)
    .sort((left, right) => right.delta - left.delta || byId(left, right))
    .slice(0, limit);
  const towardB = [...deltas]
    .filter((entry) => entry.delta < 0)
    .sort((left, right) => left.delta - right.delta || byId(left, right))
    .slice(0, limit)
    .map((entry) => ({ ingredientId: entry.ingredientId, delta: -entry.delta }));
  return { towardA, towardB };
}
