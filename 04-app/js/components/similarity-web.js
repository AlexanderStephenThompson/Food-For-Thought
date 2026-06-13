// Render the cuisine similarity web as SVG. Nodes sit at precomputed
// circular positions; edges are stroked by similarity. Each node is a
// link into the atlas detail. The figure's aria-label carries the
// meaning; the decorative edges are hidden from assistive tech.

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const VIEW_SIZE = 400;
const CENTRE = VIEW_SIZE / 2;
const NODE_RADIUS_RATIO = 0.86;
const DOT_RADIUS = 5;
const MIN_EDGE_WIDTH = 0.4;
const MAX_EDGE_WIDTH = 3.2;

function createSvgElement(name, attributes) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function projectPosition(position) {
  return {
    x: CENTRE + position.x * CENTRE * NODE_RADIUS_RATIO,
    y: CENTRE + position.y * CENTRE * NODE_RADIUS_RATIO,
  };
}

// cuisines: [{ id, name, position }]. edges: [{ a, b, similarity }].
export function renderSimilarityWeb(container, cuisines, edges, { label, caption }) {
  const positionById = new Map(
    cuisines.map((cuisine) => [cuisine.id, projectPosition(cuisine.position)]),
  );
  const peakSimilarity = Math.max(...edges.map((edge) => edge.similarity), 1e-9);

  const figure = createSvgElement("svg", {
    class: "similarity-web__figure",
    viewBox: `0 0 ${VIEW_SIZE} ${VIEW_SIZE}`,
    role: "img",
    "aria-label": label,
  });

  for (const edge of edges) {
    const start = positionById.get(edge.a);
    const end = positionById.get(edge.b);
    figure.append(
      createSvgElement("line", {
        class: "similarity-web__edge",
        x1: start.x, y1: start.y, x2: end.x, y2: end.y,
        "stroke-width":
          MIN_EDGE_WIDTH
          + (edge.similarity / peakSimilarity) * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH),
        "aria-hidden": "true",
      }),
    );
  }

  for (const cuisine of cuisines) {
    const point = positionById.get(cuisine.id);
    const link = createSvgElement("a", {
      class: "similarity-web__node",
      href: `./cuisines.html?id=${cuisine.id}`,
    });
    link.append(
      createSvgElement("circle", {
        class: "similarity-web__node-dot",
        cx: point.x, cy: point.y, r: DOT_RADIUS,
      }),
    );
    const label = createSvgElement("text", {
      class: "similarity-web__node-label",
      x: point.x,
      y: point.y - DOT_RADIUS * 2,
    });
    label.textContent = cuisine.name;
    link.append(label);
    figure.append(link);
  }

  const figureWrapper = document.createElement("figure");
  figureWrapper.className = "similarity-web";
  figureWrapper.append(figure);
  const figcaption = document.createElement("figcaption");
  figcaption.className = "similarity-web__caption";
  figcaption.textContent = caption;
  figureWrapper.append(figcaption);
  container.replaceChildren(figureWrapper);
}
