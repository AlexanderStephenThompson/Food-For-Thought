// Fetch and cache the generated data assets. Each asset is loaded once
// per page and shared across the modules that need it.

const ASSET_DIRECTORY = "./data/";
const assetCache = new Map();

// Fetch one data asset by file name, caching the parsed result.
export async function fetchAsset(fileName) {
  if (assetCache.has(fileName)) {
    return assetCache.get(fileName);
  }
  const response = await fetch(`${ASSET_DIRECTORY}${fileName}`);
  if (!response.ok) {
    throw new Error(`failed to load ${fileName}: ${response.status}`);
  }
  const payload = await response.json();
  assetCache.set(fileName, payload);
  return payload;
}
