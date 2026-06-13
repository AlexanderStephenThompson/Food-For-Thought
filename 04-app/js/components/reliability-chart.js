// Render the calibration reliability diagram as SVG: one bar per
// confidence bin showing measured accuracy, against the perfect-
// calibration diagonal. A visually-hidden table carries the same numbers
// for assistive tech.

import { formatPercentage } from "../modules/format-display.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const VIEW_WIDTH = 400;
const VIEW_HEIGHT = 300;
const PLOT_INSET = 8;
const PLOT_SIZE = VIEW_WIDTH - PLOT_INSET * 2;

function createSvgElement(name, attributes) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function plotX(fraction) {
  return PLOT_INSET + fraction * PLOT_SIZE;
}

function plotY(fraction) {
  return VIEW_HEIGHT - PLOT_INSET - fraction * PLOT_SIZE;
}

// bins: [{ bin, count, mean_confidence, accuracy }]. label/figure caption
// describe the insight in the caller's own words.
export function renderReliabilityChart(container, bins, { label, caption }) {
  const figure = createSvgElement("svg", {
    class: "reliability-chart__figure",
    viewBox: `0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`,
    role: "img",
    "aria-label": label,
  });

  figure.append(
    createSvgElement("line", {
      class: "reliability-chart__axis",
      x1: plotX(0), y1: plotY(0), x2: plotX(1), y2: plotY(0),
    }),
    createSvgElement("line", {
      class: "reliability-chart__diagonal",
      x1: plotX(0), y1: plotY(0), x2: plotX(1), y2: plotY(1),
    }),
  );

  const binWidth = PLOT_SIZE / bins.length;
  bins.forEach((entry, position) => {
    if (entry.count === 0) {
      return;
    }
    const barHeight = entry.accuracy * PLOT_SIZE;
    figure.append(
      createSvgElement("rect", {
        class: "reliability-chart__bar",
        x: PLOT_INSET + position * binWidth + binWidth * 0.15,
        y: plotY(entry.accuracy),
        width: binWidth * 0.7,
        height: barHeight,
      }),
    );
  });

  const figureWrapper = document.createElement("figure");
  figureWrapper.className = "reliability-chart";
  figureWrapper.append(figure);

  const figcaption = document.createElement("figcaption");
  figcaption.className = "reliability-chart__caption";
  figcaption.textContent = caption;
  figureWrapper.append(figcaption);

  figureWrapper.append(renderHiddenTable(bins));
  container.replaceChildren(figureWrapper);
}

function appendCell(row, text, tagName = "td") {
  const cell = document.createElement(tagName);
  cell.textContent = text;
  row.append(cell);
}

function renderHiddenTable(bins) {
  const table = document.createElement("table");
  table.className = "u-visually-hidden";

  const caption = document.createElement("caption");
  caption.textContent = "Reliability bins: confidence versus accuracy";
  table.append(caption);

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Confidence bin", "Recipes", "Mean confidence", "Accuracy"].forEach((label) =>
    appendCell(headRow, label, "th"),
  );
  head.append(headRow);
  table.append(head);

  const body = document.createElement("tbody");
  bins.forEach((entry, position) => {
    const lower = formatPercentage(position / bins.length);
    const upper = formatPercentage((position + 1) / bins.length);
    const row = document.createElement("tr");
    appendCell(row, `${lower}–${upper}`);
    appendCell(row, String(entry.count));
    appendCell(row, formatPercentage(entry.mean_confidence));
    appendCell(row, formatPercentage(entry.accuracy));
    body.append(row);
  });
  table.append(body);
  return table;
}
