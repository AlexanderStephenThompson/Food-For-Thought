// Ingredient Explorer page: search the vocabulary, and for one ingredient
// show its per-cuisine coefficient profile, aliases, and family evidence.

import { attachCombobox } from "../components/combobox.js";
import { renderProfileChart } from "../components/profile-chart.js";
import { fetchAsset } from "../data-loader.js";
import { matchIngredients } from "../modules/search-ingredients.js";

const MAX_MATCHES = 8;
const COMMON_COUNT = 20;
const PROFILE_TOP_COUNT = 8;

function selectElement(role) {
  return document.querySelector(`[data-role="${role}"]`);
}

function ingredientLink(ingredientId, label) {
  const item = document.createElement("li");
  const link = document.createElement("a");
  link.className = "chip chip--example";
  link.href = `./ingredients.html?id=${ingredientId}`;
  link.textContent = label;
  item.append(link);
  return item;
}

async function start() {
  const [model, ingredientsAsset, cuisinesAsset] = await Promise.all([
    fetchAsset("model.json"),
    fetchAsset("ingredients.json"),
    fetchAsset("cuisines.json"),
  ]);

  const nameByCuisine = new Map(
    cuisinesAsset.cuisines.map((cuisine) => [cuisine.id, cuisine.name]),
  );
  const ingredientById = new Map(
    ingredientsAsset.ingredients.map((entry) => [entry.id, entry]),
  );
  const featureIndexById = new Map(
    model.feature_ids.map((id, index) => [id, index]),
  );

  const input = document.getElementById("ingredient-search");
  const listbox = document.getElementById("ingredient-results");
  attachCombobox({
    input,
    listbox,
    getMatches(query) {
      return matchIngredients(query, ingredientsAsset.ingredients, MAX_MATCHES).map(
        (entry) => ({
          id: entry.id,
          name: entry.name,
          meta: `${entry.mentions.toLocaleString("en-US")} recipes`,
        }),
      );
    },
    onSelect(match) {
      window.location.search = `?id=${match.id}`;
    },
  });

  renderCommonList(ingredientsAsset.ingredients);

  const requestedId = new URLSearchParams(window.location.search).get("id");
  if (requestedId && ingredientById.has(requestedId)) {
    renderDetail(
      ingredientById.get(requestedId),
      model,
      featureIndexById,
      nameByCuisine,
      ingredientById,
    );
  }
}

function renderCommonList(ingredients) {
  const common = [...ingredients]
    .sort((left, right) => right.mentions - left.mentions)
    .slice(0, COMMON_COUNT);
  selectElement("common-list").replaceChildren(
    ...common.map((entry) => ingredientLink(entry.id, entry.name)),
  );
}

function renderDetail(ingredient, model, featureIndexById, nameByCuisine, ingredientById) {
  document.title = `Food for Thought — ${ingredient.name}`;
  selectElement("default-section").hidden = true;
  const detailSection = selectElement("detail-section");
  detailSection.hidden = false;

  const featureIndex = featureIndexById.get(ingredient.id);
  const coefficients = model.cuisines.map((cuisine, position) => ({
    cuisine,
    name: nameByCuisine.get(cuisine) ?? cuisine,
    value: model.coefficients[position][featureIndex],
  }));
  const ranked = [...coefficients].sort((left, right) => right.value - left.value);
  const leader = ranked[0];

  selectElement("detail-heading").textContent =
    `${ingredient.name} is ${leader.name} evidence`;

  const shown = [...ranked.slice(0, PROFILE_TOP_COUNT / 2),
    ...ranked.slice(-PROFILE_TOP_COUNT / 2)];
  renderProfileChart(
    selectElement("profile-chart"),
    shown.map((entry) => ({ label: entry.name, value: entry.value })),
  );
  selectElement("profile-caption").textContent =
    `Strongest pull toward ${leader.name}; strongest push away from `
    + `${ranked[ranked.length - 1].name}.`;

  renderAliases(ingredient);
  renderFamily(ingredient, ingredientById);
}

function renderAliases(ingredient) {
  const list = selectElement("aliases-list");
  const empty = selectElement("aliases-empty");
  if (ingredient.aliases.length === 0) {
    list.replaceChildren();
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  const chips = ingredient.aliases
    .slice()
    .sort((left, right) => right.train_frequency - left.train_frequency)
    .map((alias) => {
      const item = document.createElement("li");
      item.className = "chip";
      const label = document.createElement("span");
      label.textContent = alias.alias;
      const meta = document.createElement("span");
      meta.className = "combobox__option-meta";
      meta.textContent = `${alias.train_frequency}×`;
      item.append(label, meta);
      return item;
    });
  list.replaceChildren(...chips);
}

function renderFamily(ingredient, ingredientById) {
  const card = selectElement("family-card");
  const note = selectElement("family-note");
  const parent = ingredient.parent_id ? ingredientById.get(ingredient.parent_id) : null;

  if (!parent && ingredient.children.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  note.replaceChildren();

  if (parent && ingredient.evidence) {
    const ratio = ingredient.evidence.jsd_bits / ingredient.evidence.null95_bits;
    appendSentence(
      note,
      `Kept separate from `,
      linkTo(parent.id, parent.name),
      `: its cuisine distribution diverges ${ingredient.evidence.jsd_bits} bits, `
      + `${ratio.toFixed(1)}× the ${ingredient.evidence.null95_bits}-bit noise `
      + `threshold across ${ingredient.evidence.variant_count} recipes.`,
    );
  } else if (parent) {
    appendSentence(note, `A variant of `, linkTo(parent.id, parent.name), `.`);
  }
  if (ingredient.children.length > 0) {
    note.append(
      document.createElement("br"),
      buildChildrenSentence(ingredient.children, ingredientById),
    );
  }
}

function buildChildrenSentence(childIds, ingredientById) {
  const wrapper = document.createElement("span");
  wrapper.append("Variants kept distinct: ");
  childIds.forEach((childId, position) => {
    if (position > 0) {
      wrapper.append(", ");
    }
    wrapper.append(linkTo(childId, (ingredientById.get(childId) ?? { name: childId }).name));
  });
  wrapper.append(".");
  return wrapper;
}

function linkTo(ingredientId, label) {
  const link = document.createElement("a");
  link.href = `./ingredients.html?id=${ingredientId}`;
  link.textContent = label;
  return link;
}

function appendSentence(container, ...parts) {
  parts.forEach((part) => container.append(part));
}

start();
