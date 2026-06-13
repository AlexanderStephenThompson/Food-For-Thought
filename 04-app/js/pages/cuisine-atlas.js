// Cuisine Atlas page: the similarity web, a cuisine grid, and a detail
// view with distinctive ingredients, neighbors, and honest recall.

import { renderSimilarityWeb } from "../components/similarity-web.js";
import { fetchAsset } from "../data-loader.js";
import { formatPercentage } from "../modules/format-display.js";

const LIFT_DECIMALS = 1;
const HONEST_RECALL_CEILING = 0.5;

function selectElement(role) {
  return document.querySelector(`[data-role="${role}"]`);
}

async function start() {
  const atlas = await fetchAsset("cuisines.json");
  const cuisineById = new Map(atlas.cuisines.map((cuisine) => [cuisine.id, cuisine]));

  renderWeb(atlas);
  renderGrid(atlas.cuisines);

  const requestedId = new URLSearchParams(window.location.search).get("id");
  if (requestedId && cuisineById.has(requestedId)) {
    renderDetail(cuisineById.get(requestedId), cuisineById);
  }
}

function renderWeb(atlas) {
  const closest = atlas.edges[0];
  const closestName = (id) => atlas.cuisines.find((c) => c.id === id).name;
  selectElement("web-heading").textContent =
    `${closestName(closest.a)} and ${closestName(closest.b)} sit closest`;
  renderSimilarityWeb(
    selectElement("similarity-web"),
    atlas.cuisines,
    atlas.edges,
    {
      label:
        "A circle of 20 cuisines. Lines connect each cuisine to its nearest "
        + "neighbors; thicker lines mean more shared ingredients. "
        + `${closestName(closest.a)} and ${closestName(closest.b)} are the `
        + "closest pair.",
      caption: "Each cuisine links to its four nearest neighbors. Select one to open it.",
    },
  );
}

function renderGrid(cuisines) {
  const cards = [...cuisines]
    .sort((left, right) => right.recipe_count - left.recipe_count)
    .map((cuisine) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.className = "card card--interactive";
      link.href = `./cuisines.html?id=${cuisine.id}`;
      link.append(
        buildStat(cuisine.name, `${cuisine.recipe_count.toLocaleString("en-US")} recipes`),
        buildStat(formatPercentage(cuisine.recall), "recall", true),
      );
      item.append(link);
      return item;
    });
  selectElement("cuisine-grid").replaceChildren(...cards);
}

function buildStat(value, label, isAccent) {
  const stat = document.createElement("div");
  stat.className = isAccent ? "stat stat--accent" : "stat";
  const valueElement = document.createElement("span");
  valueElement.className = "stat__value";
  valueElement.textContent = value;
  const labelElement = document.createElement("span");
  labelElement.className = "stat__label";
  labelElement.textContent = label;
  stat.append(valueElement, labelElement);
  return stat;
}

function renderDetail(cuisine, cuisineById) {
  document.title = `Food for Thought — ${cuisine.name}`;
  selectElement("detail-section").hidden = false;
  selectElement("detail-heading").textContent = `What makes ${cuisine.name}, ${cuisine.name}`;
  selectElement("recall-note").textContent = recallSentence(cuisine, cuisineById);
  selectElement("distinctive-heading").textContent =
    `Most distinctive ingredients`;

  const distinctiveRows = cuisine.distinctive.map((entry) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const link = document.createElement("a");
    link.href = `./ingredients.html?id=${entry.id}`;
    link.textContent = entry.name;
    name.append(link);
    const lift = document.createElement("td");
    lift.className = "data-table__number";
    lift.textContent = `${entry.lift.toFixed(LIFT_DECIMALS)}×`;
    const coverage = document.createElement("td");
    coverage.className = "data-table__number";
    coverage.textContent = formatPercentage(entry.coverage);
    row.append(name, lift, coverage);
    return row;
  });
  selectElement("distinctive-body").replaceChildren(...distinctiveRows);

  const neighborRows = cuisine.neighbors.map((neighbor) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const link = document.createElement("a");
    link.href = `./cuisines.html?id=${neighbor.id}`;
    link.textContent = (cuisineById.get(neighbor.id) ?? { name: neighbor.id }).name;
    name.append(link);
    const similarity = document.createElement("td");
    similarity.className = "data-table__number";
    similarity.textContent = neighbor.similarity.toFixed(2);
    row.append(name, similarity);
    return row;
  });
  selectElement("neighbors-body").replaceChildren(...neighborRows);
}

function recallSentence(cuisine, cuisineById) {
  const recall = formatPercentage(cuisine.recall);
  if (cuisine.recall < HONEST_RECALL_CEILING) {
    const neighborName = (cuisineById.get(cuisine.neighbors[0].id) ?? {}).name
      ?? cuisine.neighbors[0].id;
    return `The model catches fewer than half of ${cuisine.name} recipes (${recall}). `
      + `${cuisine.name} shares its pantry with cuisines like ${neighborName}, `
      + `and the model says so by confusing them.`;
  }
  return `The model recognizes ${recall} of held-out ${cuisine.name} recipes.`;
}

start();
