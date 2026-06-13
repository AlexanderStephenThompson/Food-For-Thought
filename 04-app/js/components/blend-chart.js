// Render the 20-cuisine blend as ranked horizontal bars. The top three
// ranks carry the data-viz colors; the rest stay neutral, keeping the
// palette at three colors no matter which cuisines lead.

import { formatPercentage } from "../modules/format-display.js";

const RANK_CLASS = [
  "blend-chart__row--rank-first",
  "blend-chart__row--rank-second",
  "blend-chart__row--rank-third",
];

// rankedEntries: [{ cuisine, name, share }] already sorted descending.
export function renderBlendChart(container, rankedEntries) {
  const rows = rankedEntries.map((entry, position) => {
    const row = document.createElement("div");
    row.className = "blend-chart__row";
    if (position < RANK_CLASS.length) {
      row.classList.add(RANK_CLASS[position]);
    }

    const name = document.createElement("span");
    name.className = "blend-chart__name";
    name.textContent = entry.name;

    const track = document.createElement("span");
    track.className = "blend-chart__track";
    const bar = document.createElement("span");
    bar.className = "blend-chart__bar";
    bar.style.setProperty("--blend-share", entry.share);
    track.append(bar);

    const value = document.createElement("span");
    value.className = "blend-chart__value";
    value.textContent = formatPercentage(entry.share);

    row.append(name, track, value);
    return row;
  });
  container.replaceChildren(...rows);
}
