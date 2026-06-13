// Explain a blend from the model's own arithmetic. A linear model's
// contribution is exactly feature value times coefficient, so these are
// the model's computation, not an approximation. Mirrors
// model_pipeline/explain_predictions.py.

// Rank each present feature's contribution to one cuisine's logit.
// Returns [featureIndex, contribution] pairs, contribution descending
// with feature index breaking ties.
export function computeIngredientContributions(featureValues, cuisineCoefficients) {
  return [...featureValues.entries()]
    .map(([index, value]) => [index, value * cuisineCoefficients[index]])
    .sort((left, right) => right[1] - left[1] || left[0] - right[0]);
}

// Explain why a recipe scored its top cuisine over the runner-up:
// the top contributions toward the leader, and the ingredients whose
// coefficient gap most favors the leader over the runner-up.
export function summarizeBlendExplanation(
  featureValues, modelAsset, topCuisine, runnerUpCuisine, limit,
) {
  const cuisinePosition = new Map(
    modelAsset.cuisines.map((cuisine, position) => [cuisine, position]),
  );
  const topCoefficients = modelAsset.coefficients[cuisinePosition.get(topCuisine)];
  const runnerUpCoefficients =
    modelAsset.coefficients[cuisinePosition.get(runnerUpCuisine)];
  const featureIds = modelAsset.feature_ids;

  const topContributions = computeIngredientContributions(featureValues, topCoefficients)
    .slice(0, limit)
    .map(([index, contribution]) => ({
      ingredientId: featureIds[index],
      contribution,
    }));

  const differentiators = [...featureValues.entries()]
    .map(([index, value]) => [
      index,
      value * (topCoefficients[index] - runnerUpCoefficients[index]),
    ])
    .sort((left, right) => right[1] - left[1] || left[0] - right[0])
    .slice(0, limit)
    .map(([index, advantage]) => ({ ingredientId: featureIds[index], advantage }));

  return { topContributions, differentiators };
}
