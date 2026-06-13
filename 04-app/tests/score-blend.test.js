import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildFeatureValues,
  computeLogits,
  convertLogitsToBlend,
  rankBlendEntries,
} from "../js/modules/score-blend.js";

const FEATURE_IDS = ["basil", "dark_soy_sauce", "fish_sauce", "pasta", "rice", "soy_sauce"];
const PARENT_INDICES = [null, 5, null, null, null, null];
const COEFFICIENTS = [
  [2.1235, -1.0, -1.0, 2.0, -0.5, -0.6543],
  [-1.0, -1.0, -1.0, -1.0, 3.0, -0.5],
  [-2.0, 3.1111, 3.0, -2.0, -0.25, 1.0],
];
const INTERCEPTS = [0.25, -0.1, -0.15];

test("buildFeatureValues applies parent back-off to absent parents", () => {
  const featureValues = buildFeatureValues(
    ["dark_soy_sauce", "fish_sauce"], FEATURE_IDS, PARENT_INDICES, 1.0,
  );

  assert.deepEqual(
    [...featureValues.entries()].sort((left, right) => left[0] - right[0]),
    [[1, 1.0], [2, 1.0], [5, 1.0]],
  );
});

test("buildFeatureValues lets direct presence win over parent weight", () => {
  const featureValues = buildFeatureValues(
    ["dark_soy_sauce", "soy_sauce"], FEATURE_IDS, PARENT_INDICES, 0.5,
  );

  assert.equal(featureValues.get(5), 1.0);
});

test("buildFeatureValues ignores unknown and duplicate ids", () => {
  const featureValues = buildFeatureValues(
    ["pasta", "pasta", "not_a_real_ingredient"], FEATURE_IDS, PARENT_INDICES, 1.0,
  );

  assert.deepEqual([...featureValues.entries()], [[3, 1.0]]);
});

test("computeLogits matches manual arithmetic", () => {
  const featureValues = new Map([[0, 1.0], [3, 1.0]]);

  const logits = computeLogits(featureValues, COEFFICIENTS, INTERCEPTS);

  assert.equal(logits[0], 0.25 + 2.1235 + 2.0);
  assert.equal(logits[2], -0.15 - 2.0 - 2.0);
});

test("convertLogitsToBlend sums to one and keeps order", () => {
  const blend = convertLogitsToBlend([3.0, 1.0, -2.0], 1.5);

  const total = blend.reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(total - 1.0) < 1e-12);
  assert.equal(Math.max(...blend), blend[0]);
});

test("rankBlendEntries sorts by share then cuisine id", () => {
  const ranked = rankBlendEntries([0.2, 0.6, 0.2], ["italian", "mexican", "thai"]);

  assert.deepEqual(
    ranked.map((entry) => entry.cuisine),
    ["mexican", "italian", "thai"],
  );
  assert.equal(ranked[0].share, 0.6);
});
