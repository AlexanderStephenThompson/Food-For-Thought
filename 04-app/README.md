# App tier — the client-side blend explorer

A static, dependency-free web app that runs the cuisine model entirely in
the browser. The model is a 20×2,813 coefficient matrix and a softmax, so
scoring a dish is a dot product — no server, no API, nothing sent anywhere.

## Pages

| Page | What it does |
|------|--------------|
| `index.html` | Blend Builder — pick ingredients, watch the calibrated blend update live, see which ingredients vote for the leading cuisine |
| `ingredients.html` | Ingredient Explorer — search the 2,813-ingredient vocabulary; per-ingredient cuisine pull, aliases, and family evidence |
| `cuisines.html` | Cuisine Atlas — the similarity web, distinctive ingredients per cuisine, and honest per-cuisine recall |
| `versus.html` | Versus — the ingredients whose coefficients most separate any two cuisines |
| `model.html` | Model Card — how it works, the calibration reliability diagram, and where it fails |

## The data is generated, not authored

Everything in `data/` is build output of `03-gold/app_pipeline/`, derived
from the gold model artifacts and the silver taxonomy. Never hand-edit it —
rebuild it:

```bash
.venv/bin/python 03-gold/build_app.py                    # rebuild data/
.venv/bin/python 03-gold/build_app.py --check-idempotent # prove it matches disk
```

The exporter trims coefficients to four decimals (the blend displays at
four, contributions at three) and ships a set of contract vectors — recipes
scored through the Python model whose exact outputs the browser's
JavaScript scorer must reproduce. That contract is enforced by
`tests/contract-vectors.test.js`, so the in-browser math provably equals the
training-time math.

## Run it locally

```bash
python3 -m http.server -d 04-app        # then open http://localhost:8000
```

It must be served over HTTP, not opened as a `file://` URL — ES modules and
`fetch` need an origin. Every path in the app is relative, so it works from
any base path, which is what makes it GitHub Pages-ready. Publishing is a
manual step (enable Pages on the repo, point it at this directory); nothing
deploys automatically.

## Tests

```bash
node --test "04-app/tests/*.test.js"     # needs Node >= 22.7
```

The pure modules under `js/modules/` are unit-tested; the DOM components and
page glue are verified by serving the app. The contract-vector test is the
load-bearing one — it reads the real shipped `data/` assets.

## Conventions worth knowing

- **No build step and no `package.json`.** Node runs the `.js` test files
  directly via ESM auto-detection.
- **The nav is duplicated in all five pages.** Editing it means touching
  five files — the price of zero build tooling and a nav that works without
  JavaScript.
- **CSS follows the 4-file + `components/` architecture.** Every value is a
  token in `styles/global.css`; component files own one block each.
