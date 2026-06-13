import assert from "node:assert/strict";
import { test } from "node:test";

import {
  matchIngredients,
  normalizeQuery,
} from "../js/modules/search-ingredients.js";

const INGREDIENTS = [
  { id: "soy_sauce", name: "soy sauce", mentions: 70, aliases: [{ alias: "kikkoman soy sauce", train_frequency: 8 }] },
  { id: "dark_soy_sauce", name: "dark soy sauce", mentions: 25, aliases: [] },
  { id: "soybean_paste", name: "soybean paste", mentions: 12, aliases: [] },
  { id: "rice", name: "rice", mentions: 60, aliases: [] },
];

test("normalizeQuery lowercases and collapses whitespace", () => {
  assert.equal(normalizeQuery("  Dark   SOY "), "dark soy");
});

test("name prefix outranks substring matches", () => {
  const matches = matchIngredients("soy", INGREDIENTS, 4);

  assert.deepEqual(
    matches.map((entry) => entry.id),
    ["soy_sauce", "soybean_paste", "dark_soy_sauce"],
  );
});

test("alias text is searchable", () => {
  const matches = matchIngredients("kikkoman", INGREDIENTS, 4);

  assert.deepEqual(matches.map((entry) => entry.id), ["soy_sauce"]);
});

test("empty query returns no matches", () => {
  assert.deepEqual(matchIngredients("   ", INGREDIENTS, 4), []);
});

test("limit caps the result count", () => {
  const matches = matchIngredients("s", INGREDIENTS, 2);

  assert.equal(matches.length, 2);
});
