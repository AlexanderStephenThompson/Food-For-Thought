// Rank ingredients against a search query. Name prefix beats alias
// prefix beats substring; ties break toward higher training frequency,
// then id. A flat scan over ~7,100 strings is well under a millisecond.

const RANK_NAME_PREFIX = 0;
const RANK_ALIAS_PREFIX = 1;
const RANK_SUBSTRING = 2;
const RANK_NONE = 3;

// Lowercase and collapse internal whitespace.
export function normalizeQuery(text) {
  return text.toLowerCase().trim().replace(/\s+/g, " ");
}

function scoreIngredient(ingredient, query) {
  if (ingredient.name.startsWith(query)) {
    return RANK_NAME_PREFIX;
  }
  if (ingredient.aliases.some((alias) => alias.alias.startsWith(query))) {
    return RANK_ALIAS_PREFIX;
  }
  if (
    ingredient.name.includes(query)
    || ingredient.aliases.some((alias) => alias.alias.includes(query))
  ) {
    return RANK_SUBSTRING;
  }
  return RANK_NONE;
}

// Return up to `limit` ingredients matching the query, best rank first.
export function matchIngredients(query, ingredients, limit) {
  const normalized = normalizeQuery(query);
  if (normalized === "") {
    return [];
  }
  return ingredients
    .map((ingredient) => ({ ingredient, rank: scoreIngredient(ingredient, normalized) }))
    .filter((scored) => scored.rank !== RANK_NONE)
    .sort((left, right) =>
      left.rank - right.rank
      || right.ingredient.mentions - left.ingredient.mentions
      || (left.ingredient.id < right.ingredient.id ? -1 : 1),
    )
    .slice(0, limit)
    .map((scored) => scored.ingredient);
}
