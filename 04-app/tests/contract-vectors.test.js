// The JS-Python scoring contract: replay every vector the exporter
// computed through the pure Python scorer and assert the browser's
// scorer reproduces it — blends within half a 4th-decimal ulp,
// explanations within half a 6th-decimal ulp, rankings exactly.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  summarizeBlendExplanation,
} from "../js/modules/explain-blend.js";
import {
  buildFeatureValues,
  computeLogits,
  convertLogitsToBlend,
  rankBlendEntries,
} from "../js/modules/score-blend.js";

const BLEND_TOLERANCE = 5e-5 + 1e-12;
const EXPLANATION_TOLERANCE = 5e-7 + 1e-12;

const modelAsset = JSON.parse(
  readFileSync(new URL("../data/model.json", import.meta.url), "utf-8"),
);
const vectorsAsset = JSON.parse(
  readFileSync(new URL("../data/contract-vectors.json", import.meta.url), "utf-8"),
);

for (const vector of vectorsAsset.vectors) {
  test(`contract: ${vector.name}`, () => {
    const featureValues = buildFeatureValues(
      vector.ingredient_ids,
      modelAsset.feature_ids,
      modelAsset.parent_indices,
      modelAsset.parent_weight,
    );
    const logits = computeLogits(
      featureValues, modelAsset.coefficients, modelAsset.intercepts,
    );
    const blend = convertLogitsToBlend(logits, modelAsset.temperature);

    blend.forEach((share, position) => {
      assert.ok(
        Math.abs(share - vector.expected_blend[position]) < BLEND_TOLERANCE,
        `share for ${modelAsset.cuisines[position]}: ${share} vs ${vector.expected_blend[position]}`,
      );
    });

    const ranked = rankBlendEntries(blend, modelAsset.cuisines);
    assert.equal(ranked[0].cuisine, vector.expected_top_cuisine);
    assert.equal(ranked[1].cuisine, vector.expected_runner_up);

    const explanation = summarizeBlendExplanation(
      featureValues,
      modelAsset,
      vector.expected_top_cuisine,
      vector.expected_runner_up,
      vector.expected_top_contributions.length,
    );
    vector.expected_top_contributions.forEach((expected, position) => {
      const actual = explanation.topContributions[position];
      assert.equal(actual.ingredientId, expected.ingredient_id);
      assert.ok(
        Math.abs(actual.contribution - expected.contribution)
          < EXPLANATION_TOLERANCE,
      );
    });
    vector.expected_differentiators.forEach((expected, position) => {
      const actual = explanation.differentiators[position];
      assert.equal(actual.ingredientId, expected.ingredient_id);
      assert.ok(
        Math.abs(actual.advantage - expected.advantage) < EXPLANATION_TOLERANCE,
      );
    });
  });
}
