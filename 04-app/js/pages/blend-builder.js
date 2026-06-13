// Blend Builder page: pick ingredients, score a live calibrated blend,
// and explain it from the model's own coefficients.

import { attachCombobox } from "../components/combobox.js";
import { renderBlendChart } from "../components/blend-chart.js";
import { renderProfileChart } from "../components/profile-chart.js";
import { fetchAsset } from "../data-loader.js";
import { summarizeBlendExplanation } from "../modules/explain-blend.js";
import { formatPercentage } from "../modules/format-display.js";
import { matchIngredients } from "../modules/search-ingredients.js";
import {
  buildFeatureValues,
  computeLogits,
  convertLogitsToBlend,
  rankBlendEntries,
} from "../modules/score-blend.js";

const MAX_MATCHES = 8;
const TOP_BLEND_DISPLAY = 8;
const SUMMARY_TOP_COUNT = 3;
const EXPLANATION_LIMIT = 5;

function selectElement(role) {
  return document.querySelector(`[data-role="${role}"]`);
}

async function start() {
  const [model, ingredientsAsset, cuisinesAsset, contractVectors] = await Promise.all([
    fetchAsset("model.json"),
    fetchAsset("ingredients.json"),
    fetchAsset("cuisines.json"),
    fetchAsset("contract-vectors.json"),
  ]);

  const nameByCuisine = new Map(
    cuisinesAsset.cuisines.map((cuisine) => [cuisine.id, cuisine.name]),
  );
  const nameByIngredient = new Map(
    ingredientsAsset.ingredients.map((entry) => [entry.id, entry.name]),
  );
  const selectedIds = [];

  const input = document.getElementById("ingredient-search");
  const listbox = document.getElementById("ingredient-results");
  const chipList = selectElement("selected-chips");
  const blendHeading = selectElement("blend-heading");
  const blendSummary = selectElement("blend-summary");
  const blendNote = selectElement("blend-note");
  const blendChart = selectElement("blend-chart");
  const whyNote = selectElement("why-note");
  const contributionsCard = selectElement("contributions-card");
  const contributionsHeading = selectElement("contributions-heading");
  const contributionsChart = selectElement("contributions-chart");
  const differentiatorsCard = selectElement("differentiators-card");
  const differentiatorsHeading = selectElement("differentiators-heading");
  const differentiatorsChart = selectElement("differentiators-chart");

  function ingredientName(ingredientId) {
    return nameByIngredient.get(ingredientId) ?? ingredientId.replace(/_/g, " ");
  }

  function renderChips() {
    const chips = selectedIds.map((ingredientId) => {
      const item = document.createElement("li");
      item.className = "chip";
      const label = document.createElement("span");
      label.textContent = ingredientName(ingredientId);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "chip__remove";
      remove.setAttribute("aria-label", `Remove ${ingredientName(ingredientId)}`);
      remove.textContent = "×";
      remove.addEventListener("click", () => {
        selectedIds.splice(selectedIds.indexOf(ingredientId), 1);
        renderChips();
        update();
        input.focus();
      });
      item.append(label, remove);
      return item;
    });
    chipList.replaceChildren(...chips);
  }

  function renderExplanations(rankedEntries, featureValues) {
    if (selectedIds.length === 0) {
      contributionsCard.hidden = true;
      differentiatorsCard.hidden = true;
      whyNote.hidden = false;
      return;
    }
    whyNote.hidden = true;
    const leader = rankedEntries[0];
    const runnerUp = rankedEntries[1];
    const explanation = summarizeBlendExplanation(
      featureValues, model, leader.cuisine, runnerUp.cuisine, EXPLANATION_LIMIT,
    );

    contributionsHeading.textContent = `Votes toward ${leader.name}`;
    renderProfileChart(
      contributionsChart,
      explanation.topContributions.map((entry) => ({
        label: ingredientName(entry.ingredientId),
        value: entry.contribution,
      })),
    );
    contributionsCard.hidden = false;

    differentiatorsHeading.textContent =
      `What separates ${leader.name} from ${runnerUp.name}`;
    renderProfileChart(
      differentiatorsChart,
      explanation.differentiators.map((entry) => ({
        label: ingredientName(entry.ingredientId),
        value: entry.advantage,
      })),
    );
    differentiatorsCard.hidden = false;
  }

  function update() {
    const featureValues = buildFeatureValues(
      selectedIds, model.feature_ids, model.parent_indices, model.parent_weight,
    );
    const logits = computeLogits(featureValues, model.coefficients, model.intercepts);
    const blend = convertLogitsToBlend(logits, model.temperature);
    const ranked = rankBlendEntries(blend, model.cuisines).map((entry) => ({
      ...entry,
      name: nameByCuisine.get(entry.cuisine) ?? entry.cuisine,
    }));

    renderBlendChart(blendChart, ranked.slice(0, TOP_BLEND_DISPLAY));

    const leader = ranked[0];
    if (selectedIds.length === 0) {
      blendHeading.textContent = "With no ingredients, the model guesses base rates";
      blendNote.textContent =
        `Italian leads at ${formatPercentage(leader.share)} only because it `
        + "dominates the training data. Add an ingredient and the evidence takes over.";
    } else {
      blendHeading.textContent = `Reads as ${formatPercentage(leader.share)} ${leader.name}`;
      blendNote.textContent = "";
    }
    blendSummary.textContent = ranked
      .slice(0, SUMMARY_TOP_COUNT)
      .map((entry) => `${entry.name} ${formatPercentage(entry.share)}`)
      .join(", ");

    renderExplanations(ranked, featureValues);
  }

  function addIngredient(ingredientId) {
    if (!selectedIds.includes(ingredientId)) {
      selectedIds.push(ingredientId);
      renderChips();
      update();
    }
  }

  attachCombobox({
    input,
    listbox,
    getMatches(query) {
      return matchIngredients(query, ingredientsAsset.ingredients, MAX_MATCHES)
        .filter((entry) => !selectedIds.includes(entry.id))
        .map((entry) => ({
          id: entry.id,
          name: entry.name,
          meta: `${entry.mentions.toLocaleString("en-US")} recipes`,
        }));
    },
    onSelect(match) {
      addIngredient(match.id);
    },
  });

  renderExampleChips(contractVectors, addIngredient);
  update();
}

function renderExampleChips(contractVectors, addIngredient) {
  const exampleList = selectElement("example-chips");
  const examples = contractVectors.vectors.filter(
    (vector) => vector.example && vector.ingredient_ids.length > 0,
  );
  const chips = examples.map((vector) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip chip--example";
    button.textContent = exampleLabel(vector.name);
    button.addEventListener("click", () => {
      vector.ingredient_ids.forEach(addIngredient);
    });
    item.append(button);
    return item;
  });
  exampleList.replaceChildren(...chips);
}

function exampleLabel(vectorName) {
  return vectorName.replace(/^the /, "").replace(/^classic /, "");
}

start();
