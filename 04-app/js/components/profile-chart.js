// Render diverging bars from a centre axis. Each entry pulls right when
// positive, left when negative, scaled against the largest magnitude.

import { formatSignedNumber } from "../modules/format-display.js";

const VALUE_DECIMALS = 2;

// entries: [{ label, value }]. Returns nothing; fills the container.
export function renderProfileChart(container, entries) {
  const peak = Math.max(...entries.map((entry) => Math.abs(entry.value)), 1e-9);
  const rows = entries.map((entry) => {
    const row = document.createElement("div");
    row.className = "profile-chart__row";

    const label = document.createElement("span");
    label.className = "profile-chart__name";
    label.textContent = entry.label;

    const negativeSide = document.createElement("span");
    negativeSide.className = "profile-chart__side profile-chart__side--negative";
    const positiveSide = document.createElement("span");
    positiveSide.className = "profile-chart__side profile-chart__side--positive";

    const bar = document.createElement("span");
    bar.style.setProperty("--profile-magnitude", Math.abs(entry.value) / peak);
    if (entry.value >= 0) {
      bar.className = "profile-chart__bar profile-chart__bar--positive";
      positiveSide.append(bar);
    } else {
      bar.className = "profile-chart__bar profile-chart__bar--negative";
      negativeSide.append(bar);
    }

    const value = document.createElement("span");
    value.className = "profile-chart__value";
    value.textContent = formatSignedNumber(entry.value, VALUE_DECIMALS);

    row.append(label, negativeSide, positiveSide, value);
    return row;
  });
  container.replaceChildren(...rows);
}
