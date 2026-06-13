// Display formatters shared across pages. Pure string output.

const PERCENTAGE_DECIMALS = 1;
const PERCENT_PER_UNIT = 100;

// Render a 0..1 share as a one-decimal percentage, e.g. "58.3%".
export function formatPercentage(share) {
  return `${(share * PERCENT_PER_UNIT).toFixed(PERCENTAGE_DECIMALS)}%`;
}

// Render a number with an explicit sign, e.g. "+0.123".
export function formatSignedNumber(value, decimals) {
  const fixed = value.toFixed(decimals);
  return value >= 0 ? `+${fixed}` : fixed;
}

// Render an integer with thousands separators, e.g. "39,774".
export function formatCount(value) {
  return value.toLocaleString("en-US");
}
