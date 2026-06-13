// Model Card page: how the model works, how calibrated it is, and where
// it fails — read straight from the evaluation and calibration artifacts.

import { renderReliabilityChart } from "../components/reliability-chart.js";
import { fetchAsset } from "../data-loader.js";
import { formatCount, formatPercentage } from "../modules/format-display.js";

const HONEST_RECALL_CEILING = 0.5;
const DOMINANT_BIN_INDEX = 7;

function selectElement(role) {
  return document.querySelector(`[data-role="${role}"]`);
}

function buildStat(value, label) {
  const item = document.createElement("li");
  const stat = document.createElement("div");
  stat.className = "stat";
  const valueElement = document.createElement("span");
  valueElement.className = "stat__value";
  valueElement.textContent = value;
  const labelElement = document.createElement("span");
  labelElement.className = "stat__label";
  labelElement.textContent = label;
  stat.append(valueElement, labelElement);
  item.append(stat);
  return item;
}

async function start() {
  const card = await fetchAsset("model-card.json");
  renderCalibration(card.calibration);
  renderRecall(card.per_cuisine);
  renderConfusion(card.confusion_pairs);
  renderRecipe(card);
}

function renderCalibration(calibration) {
  const dominant = calibration.reliability.after[DOMINANT_BIN_INDEX];
  selectElement("calibration-heading").textContent = dominant.count > 0
    ? `When it says ${formatPercentage(dominant.mean_confidence)}, it is right `
      + `${formatPercentage(dominant.accuracy)} of the time`
    : "How honest the probabilities are";

  renderReliabilityChart(
    selectElement("reliability-chart"),
    calibration.reliability.after,
    {
      label:
        "Reliability after calibration: for each confidence band, the bar "
        + "height is the share the model actually got right. Bars tracking the "
        + "diagonal mean the stated confidence matches reality.",
      caption:
        "Each bar is a confidence band; its height is measured accuracy. The "
        + "dashed line is perfect calibration.",
    },
  );

  selectElement("calibration-stats").replaceChildren(
    buildStat(calibration.temperature.toFixed(3), "temperature"),
    buildStat(
      `${formatPercentage(calibration.ece_before)} → ${formatPercentage(calibration.ece_after)}`,
      "calibration error",
    ),
    buildStat(
      `${calibration.log_loss_before.toFixed(3)} → ${calibration.log_loss_after.toFixed(3)}`,
      "log loss",
    ),
  );
}

function renderRecall(perCuisine) {
  const weakest = [...perCuisine]
    .filter((entry) => entry.recall < HONEST_RECALL_CEILING)
    .sort((left, right) => left.recall - right.recall)
    .map((entry) => `${entry.name} (${formatPercentage(entry.recall)})`);
  selectElement("accuracy-note").textContent = weakest.length > 0
    ? `The model is strong on the big, distinctive cuisines and weak on the `
      + `ones that share a pantry. It misses more than half of: ${weakest.join(", ")}.`
    : "Recall on held-out recipes the model never trained on.";

  const rows = [...perCuisine]
    .sort((left, right) => right.recall - left.recall)
    .map((entry) => {
      const row = document.createElement("tr");
      if (entry.recall < HONEST_RECALL_CEILING) {
        row.className = "data-table__row--muted";
      }
      const name = document.createElement("td");
      name.textContent = entry.name;
      const recipes = document.createElement("td");
      recipes.className = "data-table__number";
      recipes.textContent = formatCount(entry.recipe_count);
      const recall = document.createElement("td");
      recall.className = "data-table__number";
      recall.textContent = formatPercentage(entry.recall);
      row.append(name, recipes, recall);
      return row;
    });
  selectElement("recall-body").replaceChildren(...rows);
}

function renderConfusion(pairs) {
  const leader = pairs[0];
  selectElement("confusion-note").textContent =
    `${leader.count} ${leader.true_name} recipes read as ${leader.predicted_name}. `
    + `The taxonomy already rated them close`
    + (leader.neighbor_similarity !== null
      ? ` (${leader.neighbor_similarity} similarity).`
      : `.`);

  const rows = pairs.map((pair) => {
    const row = document.createElement("tr");
    const predicted = document.createElement("td");
    predicted.textContent = pair.predicted_name;
    const actual = document.createElement("td");
    actual.textContent = pair.true_name;
    const count = document.createElement("td");
    count.className = "data-table__number";
    count.textContent = formatCount(pair.count);
    const similarity = document.createElement("td");
    similarity.className = "data-table__number";
    similarity.textContent =
      pair.neighbor_similarity !== null ? pair.neighbor_similarity.toFixed(2) : "—";
    row.append(predicted, actual, count, similarity);
    return row;
  });
  selectElement("confusion-body").replaceChildren(...rows);
}

function renderRecipe(card) {
  selectElement("recipe-stats").replaceChildren(
    buildStat(formatPercentage(card.mean.accuracy), "mean accuracy"),
    buildStat(
      formatPercentage(card.baseline_naive_bayes.mean_accuracy),
      "naive bayes baseline",
    ),
    buildStat(formatCount(card.training.recipe_count), "training recipes"),
    buildStat(formatCount(card.training.feature_count), "ingredients"),
  );
}

start();
