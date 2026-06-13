import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatCount,
  formatPercentage,
  formatSignedNumber,
} from "../js/modules/format-display.js";

test("formatPercentage renders one decimal", () => {
  assert.equal(formatPercentage(0.583), "58.3%");
  assert.equal(formatPercentage(0.0), "0.0%");
});

test("formatSignedNumber always carries a sign", () => {
  assert.equal(formatSignedNumber(0.1234, 3), "+0.123");
  assert.equal(formatSignedNumber(-0.5, 3), "-0.500");
});

test("formatCount adds thousands separators", () => {
  assert.equal(formatCount(39774), "39,774");
});
