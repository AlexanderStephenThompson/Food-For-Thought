// Versus page: pick two cuisines, see the ingredients whose coefficients
// most separate them, in each direction.

import { fetchAsset } from "../data-loader.js";
import { computeCoefficientDeltas } from "../modules/compare-cuisines.js";
import { renderProfileChart } from "../components/profile-chart.js";

const DIFFERENTIATOR_LIMIT = 10;
const DEFAULT_A = "thai";
const DEFAULT_B = "vietnamese";

function selectElement(role) {
  return document.querySelector(`[data-role="${role}"]`);
}

async function start() {
  const [model, atlas] = await Promise.all([
    fetchAsset("model.json"),
    fetchAsset("cuisines.json"),
  ]);

  const nameByCuisine = new Map(atlas.cuisines.map((cuisine) => [cuisine.id, cuisine.name]));
  const ingredientNames = await loadIngredientNames();
  const orderedCuisines = [...atlas.cuisines].sort((left, right) =>
    left.name < right.name ? -1 : 1,
  );

  const selectA = selectElement("select-a");
  const selectB = selectElement("select-b");
  fillSelect(selectA, orderedCuisines);
  fillSelect(selectB, orderedCuisines);

  const params = new URLSearchParams(window.location.search);
  selectA.value = pickInitial(params.get("a"), DEFAULT_A, nameByCuisine);
  selectB.value = pickInitial(params.get("b"), DEFAULT_B, nameByCuisine);

  function update() {
    const cuisineA = selectA.value;
    const cuisineB = selectB.value;
    const nameA = nameByCuisine.get(cuisineA);
    const nameB = nameByCuisine.get(cuisineB);

    selectElement("result-heading").textContent =
      cuisineA === cuisineB
        ? `Pick two different cuisines`
        : `${nameA} versus ${nameB}`;
    selectElement("result-summary").textContent = `Comparing ${nameA} and ${nameB}`;

    const comparison = computeCoefficientDeltas(
      model, cuisineA, cuisineB, DIFFERENTIATOR_LIMIT,
    );
    selectElement("toward-a-heading").textContent = `Says ${nameA}`;
    selectElement("toward-b-heading").textContent = `Says ${nameB}`;
    renderProfileChart(
      selectElement("toward-a-chart"),
      comparison.towardA.map((entry) => ({
        label: ingredientNames.get(entry.ingredientId) ?? entry.ingredientId,
        value: entry.delta,
      })),
    );
    renderProfileChart(
      selectElement("toward-b-chart"),
      comparison.towardB.map((entry) => ({
        label: ingredientNames.get(entry.ingredientId) ?? entry.ingredientId,
        value: entry.delta,
      })),
    );
  }

  selectA.addEventListener("change", update);
  selectB.addEventListener("change", update);
  update();
}

async function loadIngredientNames() {
  const ingredientsAsset = await fetchAsset("ingredients.json");
  return new Map(ingredientsAsset.ingredients.map((entry) => [entry.id, entry.name]));
}

function fillSelect(select, cuisines) {
  select.replaceChildren(
    ...cuisines.map((cuisine) => {
      const option = document.createElement("option");
      option.value = cuisine.id;
      option.textContent = cuisine.name;
      return option;
    }),
  );
}

function pickInitial(requested, fallback, nameByCuisine) {
  return requested && nameByCuisine.has(requested) ? requested : fallback;
}

start();
