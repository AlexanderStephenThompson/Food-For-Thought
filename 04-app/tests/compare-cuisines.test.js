import assert from "node:assert/strict";
import { test } from "node:test";

import { computeCoefficientDeltas } from "../js/modules/compare-cuisines.js";

const MODEL_ASSET = {
  cuisines: ["italian", "mexican", "thai"],
  feature_ids: ["basil", "dark_soy_sauce", "fish_sauce", "pasta"],
  coefficients: [
    [2.0, -1.0, -1.0, 2.0],
    [-1.0, -1.0, -1.0, -1.0],
    [-2.0, 3.0, 3.0, -2.0],
  ],
};

test("deltas split into toward-a and toward-b, largest first", () => {
  const comparison = computeCoefficientDeltas(MODEL_ASSET, "thai", "italian", 2);

  assert.deepEqual(
    comparison.towardA.map((entry) => entry.ingredientId),
    ["dark_soy_sauce", "fish_sauce"],
  );
  assert.equal(comparison.towardA[0].delta, 4.0);
  assert.deepEqual(
    comparison.towardB.map((entry) => entry.ingredientId),
    ["basil", "pasta"],
  );
  assert.equal(comparison.towardB[0].delta, 4.0);
});

test("comparing a cuisine with itself yields empty lists", () => {
  const comparison = computeCoefficientDeltas(MODEL_ASSET, "thai", "thai", 3);

  assert.deepEqual(comparison.towardA, []);
  assert.deepEqual(comparison.towardB, []);
});
