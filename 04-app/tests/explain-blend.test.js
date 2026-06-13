import assert from "node:assert/strict";
import { test } from "node:test";

import {
  computeIngredientContributions,
  summarizeBlendExplanation,
} from "../js/modules/explain-blend.js";

const MODEL_ASSET = {
  cuisines: ["italian", "mexican", "thai"],
  feature_ids: ["basil", "dark_soy_sauce", "fish_sauce", "pasta", "rice", "soy_sauce"],
  coefficients: [
    [2.0, -1.0, -1.0, 2.0, -0.5, -0.5],
    [-1.0, -1.0, -1.0, -1.0, 3.0, -0.5],
    [-2.0, 3.0, 3.0, -2.0, -0.25, 1.0],
  ],
};

test("contribution equals value times coefficient", () => {
  const contributions = computeIngredientContributions(
    new Map([[1, 1.0], [5, 0.3]]),
    MODEL_ASSET.coefficients[2],
  );

  const byIndex = new Map(contributions);
  assert.equal(byIndex.get(1), 3.0);
  assert.equal(byIndex.get(5), 0.3 * 1.0);
});

test("contributions sort descending with index tiebreak", () => {
  const contributions = computeIngredientContributions(
    new Map([[1, 1.0], [2, 1.0], [0, 1.0]]),
    MODEL_ASSET.coefficients[2],
  );

  assert.deepEqual(
    contributions.map(([index]) => index),
    [1, 2, 0],
  );
});

test("summarizeBlendExplanation ranks differentiators between top two", () => {
  const explanation = summarizeBlendExplanation(
    new Map([[1, 1.0], [2, 1.0], [5, 0.3]]),
    MODEL_ASSET,
    "thai",
    "mexican",
    2,
  );

  assert.equal(explanation.topContributions.length, 2);
  assert.ok(
    ["dark_soy_sauce", "fish_sauce"].includes(
      explanation.differentiators[0].ingredientId,
    ),
  );
  // thai coefficient 3.0 minus mexican -1.0 on a 1.0-valued feature.
  assert.equal(explanation.differentiators[0].advantage, 4.0);
});
