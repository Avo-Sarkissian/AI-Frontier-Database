# AI Frontier — Static GitHub Pages Migration (Pyodide) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a static, always-on, free copy of the AI Frontier dashboard at `https://avo-sarkissian.github.io/AI-Frontier-Database/` that reuses the existing Python chart code unchanged, by running it in the browser via Pyodide.

**Architecture:** A `/docs` static site served by GitHub Pages from `main:/docs`. A build script (`build_static.py`) pre-renders every chart's default state to figure JSON and bundles the Python chart/data modules + CSVs into `docs/`. The page paints the pre-rendered figures instantly with Plotly.js, then boots Pyodide in the background; once ready, a Python bridge module (`static_api.py`) recomputes figures on every filter change, exactly as the Dash callbacks do today.

**Tech Stack:** Python 3.12 (Pyodide 0.27.x), pandas, numpy, plotly (figure construction), Plotly.js (CDN, rendering), vanilla JS, GitHub Pages. pytest for the Python unit tests.

## Global Constraints

- Reuse `assets/style.css` verbatim — no visual redesign.
- Do not change `app.py`'s runtime behavior; the Dash app must still run with `python app.py` after this work. (Task 0 refactors `app.py` to delegate 6 pure helpers to `static_helpers` — behavior must stay byte-identical; all call sites unchanged.)
- Pure helpers live ONCE in `static_helpers.py` (imports only `re` + `pandas`) and are imported by `app.py`, `static_api.py`, and `build_static.py`. Do not duplicate them.
- Existing source modules that may change: `app.py` (Task 0, delegation only) and `components/stack_recommender.py` (Task 3, additive — new dash-free path; existing dash-component path preserved).
- `static_api.py` and `static_helpers.py` must not import `dash`, `requests`, `flask`, or `gunicorn` (Pyodide has none of them).
- Bundle preserves package layout so `Path(__file__).parent / "raw" / ...` resolves: `data/ingest.py` → `data/raw/aa_models.csv`.
- All `static_api` functions return **JSON strings** (via `fig.to_json()` / `json.dumps(...)`) — never live objects — so JS parses with `JSON.parse`.
- Pin Pyodide to a single concrete version across `app.js` (use `v0.27.7`); pin Plotly.js to a version compatible with the installed plotly.py major (plotly.py 6.x → Plotly.js 3.x; use `plotly-3.0.1.min.js`).
- Per CLAUDE.md: commit after every task and push to `origin main`.
- Tabs to support (10): `overview`, `recommend`, `landscape`, `rankings`, `compare`, `budget`, `table`, `local`, `image`, `video`.

---

## File Structure

**New files**
- `static_helpers.py` (root; bundled into the zip) — pure dash-free helpers shared by `app.py`, `static_api.py`, `build_static.py`.
- `build_static.py` (root) — pre-renders default figures, copies CSS, builds `docs/pybundle.zip`, writes `docs/figures/manifest.json`.
- `static_api.py` (root; bundled into the zip) — the bridge: ports the 19 callbacks to plain JSON-returning functions.
- `docs/index.html` — UI shell (header, stat bar, global filters, tab nav, empty chart containers, detail panel).
- `docs/app.js` — render default figures; boot Pyodide; mount bundle; wire controls → `static_api` → `Plotly.react`; pure-JS extras (CSV export, share-URL, detail panel, tab switching, presets, search debounce).
- `docs/assets/style.css` — copied from `assets/style.css` by the build.
- `docs/figures/*.json` — generated default figures.
- `docs/figures/manifest.json` — generated metadata (model count, provider count, floor price, peak quality, default radar/diverse5, cache timestamp).
- `docs/pybundle.zip` — generated; contains `components/`, `data/` (loaders + `raw/*.csv`), `static_api.py`.
- `docs/.nojekyll` — empty file so GitHub Pages serves `_`-prefixed and all static paths verbatim.
- `tests/test_stack_recommender_html.py`, `tests/test_static_api.py`, `tests/test_build_static.py`.

**Modified files**
- `app.py` — Task 0: delegate 6 pure helpers to `static_helpers` (behavior unchanged).
- `components/stack_recommender.py` — split selection from rendering; add HTML-string renderer.
- `.claude/skills/refresh-models/SKILL.md` (or the skill's script) — change the deploy step.
- `README.md` — document the static site + build command.
- `.gitignore` — ensure `docs/pybundle.zip` and generated figures are NOT ignored (they must ship); but ignore `docs/figures/*.png` if any.

---

## Phase 0 — Shared dash-free helpers

Goal: one source of truth for the pure helper functions, imported by `app.py`, `static_api.py`, and `build_static.py` — no duplication.

### Task 0: Extract `static_helpers.py` and delegate from `app.py`

**Files:**
- Create: `static_helpers.py`
- Modify: `app.py` (replace 6 helper defs with imports/shims — call sites unchanged)
- Create: `tests/test_static_helpers.py`

**Interfaces:**
- Produces (all pure, dash-free):
  - `apply_filters(df, providers, min_quality, search="") -> DataFrame`
  - `compute_diverse5(df) -> list[str]`
  - `ctx_to_k(c) -> float | None`
  - `quality_label(pct: float) -> str`
  - `provider_options(df) -> list[dict]`
  - `model_options(df) -> list[dict]`
- Consumed by: `app.py` (Task 0), `build_static.py` (Task 1), `static_api.py` (Tasks 4–5).

**Constraint:** `static_helpers.py` imports only `re` and `pandas` — never `dash`. `app.py` runtime behavior must be byte-identical after refactor.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_static_helpers.py
from data.ingest import get_models
import static_helpers as h

def test_apply_filters_provider_and_quality():
    df = get_models()
    out = h.apply_filters(df, ["Anthropic"], 40, "")
    assert len(out) > 0 and (out["provider"] == "Anthropic").all() and (out["quality"] >= 40).all()

def test_compute_diverse5_returns_up_to_five():
    df = get_models()
    picks = h.compute_diverse5(df)
    assert 0 < len(picks) <= 5 and len(set(picks)) == len(picks)

def test_ctx_to_k_and_quality_label():
    assert h.ctx_to_k("1m") == 1000 and h.ctx_to_k("128k") == 128
    assert h.quality_label(95) == "Exceptional" and h.quality_label(10) == "Limited"

def test_options_shape():
    df = get_models()
    assert all(set(o) == {"label", "value"} for o in h.provider_options(df))
    assert all(set(o) == {"label", "value"} for o in h.model_options(df))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_static_helpers.py -v`
Expected: FAIL (`static_helpers` does not exist).

- [ ] **Step 3: Create `static_helpers.py`** — move the bodies of `_apply_filters` (parameterized on `df`), `_compute_diverse5`, `_ctx_to_k`, `_quality_label`, `_provider_options`, `_model_options` from `app.py` (lines 86–194) **verbatim** (renamed without the leading underscore; `apply_filters` takes `df` as its first parameter).

```python
"""Pure, dash-free helpers shared by app.py, static_api.py, build_static.py.
Imports only stdlib + pandas so this module loads under Pyodide."""
import re
import pandas as pd

def apply_filters(df, providers, min_quality, search=""):
    filtered = df.copy()
    if providers:
        filtered = filtered[filtered["provider"].isin(providers)]
    if min_quality and float(min_quality) > 0:
        filtered = filtered[filtered["quality"] >= float(min_quality)]
    if search and str(search).strip():
        pat = re.escape(str(search).strip())
        mask = (filtered["model"].str.contains(pat, case=False, na=False) |
                filtered["provider"].str.contains(pat, case=False, na=False))
        filtered = filtered[mask]
    return filtered

# compute_diverse5, ctx_to_k, quality_label, provider_options, model_options:
# move verbatim from app.py (86-194); compute_diverse5/ctx_to_k/quality_label
# already take their argument, so they port unchanged (just drop the underscore).
```

- [ ] **Step 4: Modify `app.py` to delegate** — replace the 6 helper definitions with imports plus a single shim for `_apply_filters` (which closes over the module global `df`):

```python
from static_helpers import (
    apply_filters,
    compute_diverse5 as _compute_diverse5,
    ctx_to_k as _ctx_to_k,
    quality_label as _quality_label,
    provider_options as _provider_options,
    model_options as _model_options,
)

def _apply_filters(providers, min_quality, search=""):
    return apply_filters(df, providers, min_quality, search)
```

Delete the original `def _apply_filters / _compute_diverse5 / _ctx_to_k / _quality_label / _provider_options / _model_options` bodies. Leave all call sites untouched.

- [ ] **Step 5: Run tests + verify the Dash app imports**

Run: `python -m pytest tests/test_static_helpers.py -v` → PASS.
Run: `python -c "import app; print('app imports OK')"` → prints OK (no error). Then `python app.py` briefly and confirm Overview/Compare/Table render unchanged (Playwright screenshot at `http://localhost:8050`).

- [ ] **Step 6: Commit**

```bash
git add static_helpers.py app.py tests/test_static_helpers.py
git commit -m "refactor: extract pure dash-free helpers into static_helpers"
git push origin main
```

---

## Phase 1 — Static-first dashboard (no Pyodide yet)

Goal: a visibly correct, fully-styled dashboard at `docs/index.html` showing every chart's default state, served as pure static files. This de-risks layout/CSS before any Pyodide work.

### Task 1: Build script scaffold + default figure export

**Files:**
- Create: `build_static.py`
- Create: `tests/test_build_static.py`

**Interfaces:**
- Produces: `build_static.py` exposes `main()` and helper `export_default_figures(out_dir: Path) -> list[str]` returning the list of figure JSON filenames written. Writes `docs/figures/<id>.json` for each chart id below and `docs/figures/manifest.json`.

Default figure ids and their source calls (mirror `app.py` initial `figure=` values):

| id | source call |
|----|-------------|
| `pareto` | `build_pareto_scatter(df)` |
| `quadrant` | `build_quadrant(df)` (overview speed mode) |
| `treemap` | `build_treemap(df)` |
| `provider_leaderboard` | `build_provider_leaderboard(df)` |
| `rankings` | `build_rankings(df, top_n=25, metric="intelligence")` |
| `value_leaders` | `build_value_leaders(df)` |
| `radar` | `build_radar(df, DIVERSE5)` |
| `cost_calc` | `build_cost_calc(df, monthly_tokens_m=1.0)` |
| `local_scatter` | `build_local_scatter(get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792, hw_type="nvidia"), vram_gb=32, quant="Q4")` |
| `local_compat` | `build_local_compat(get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792, hw_type="nvidia"), quant="Q4")` |
| `image_faceted` | `build_image_faceted(get_image_df())` |
| `video_rankings` | `build_video_rankings(get_video_df())` |
| `video_scatter` | `build_video_scatter(get_video_df()[get_video_df()["price_per_sec"] > 0])` |

`manifest.json` fields: `{"model_count": int, "provider_count": int, "floor_price": "$0.xxx", "peak_quality": "xx.x", "diverse5": [..], "model_options": [{"label","value"}...], "provider_options": [...], "generated": "Mon DD HH:MM"}` — computed exactly as the stat bar / `_provider_options` / `_model_options` / `_compute_diverse5` do in `app.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_static.py
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def test_build_produces_figures_and_manifest(tmp_path):
    # Build writes into docs/figures by default; assert key artifacts exist & parse.
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    figdir = ROOT / "docs" / "figures"
    for fid in ["pareto", "treemap", "rankings", "radar", "cost_calc",
                "local_scatter", "image_faceted", "video_rankings"]:
        p = figdir / f"{fid}.json"
        assert p.exists(), f"missing {p}"
        fig = json.loads(p.read_text())
        assert "data" in fig and "layout" in fig
    manifest = json.loads((figdir / "manifest.json").read_text())
    assert int(manifest["model_count"]) > 0
    assert manifest["provider_options"] and manifest["diverse5"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_static.py -v`
Expected: FAIL (`build_static.py` does not exist / no figures written).

- [ ] **Step 3: Write `build_static.py`**

```python
"""Pre-render default figures + bundle Python for the static Pyodide site."""
import json, shutil, zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.ingest import get_models
from data.local_models import get_local_df
from data.image_models import get_image_df
from data.video_models import get_video_df
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap import build_treemap
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders
from components.charts.radar import build_radar
from components.charts.cost_calc import build_cost_calc
from components.charts.local_scatter import build_local_scatter
from components.charts.local_compat import build_local_compat
from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings, build_video_scatter

from static_helpers import compute_diverse5, provider_options, model_options

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
FIG  = DOCS / "figures"

# Do NOT import app.py — it starts background scrapers at import time. Shared
# pure logic lives in static_helpers (imported above).


def export_default_figures(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_models()
    diverse5 = compute_diverse5(df)
    local_df = get_local_df(quant="Q4", vram_gb=32, bandwidth_gbps=1792, hw_type="nvidia")
    img_df = get_image_df()
    vdf = get_video_df()
    vpaid = vdf[vdf["price_per_sec"] > 0] if not vdf.empty else vdf
    figures = {
        "pareto":               build_pareto_scatter(df),
        "quadrant":             build_quadrant(df),
        "treemap":              build_treemap(df),
        "provider_leaderboard": build_provider_leaderboard(df),
        "rankings":             build_rankings(df, top_n=25, metric="intelligence"),
        "value_leaders":        build_value_leaders(df),
        "radar":                build_radar(df, diverse5),
        "cost_calc":            build_cost_calc(df, monthly_tokens_m=1.0),
        "local_scatter":        build_local_scatter(local_df, vram_gb=32, quant="Q4"),
        "local_compat":         build_local_compat(local_df, quant="Q4"),
        "image_faceted":        build_image_faceted(img_df),
        "video_rankings":       build_video_rankings(vdf),
        "video_scatter":        build_video_scatter(vpaid if not vpaid.empty else vdf),
    }
    written = []
    for fid, fig in figures.items():
        (out_dir / f"{fid}.json").write_text(fig.to_json())
        written.append(f"{fid}.json")

    manifest = {
        "model_count":      int(len(df)),
        "provider_count":   int(df["provider"].nunique()),
        "floor_price":      f"${df['price'].min():.3f}",
        "peak_quality":     f"{df['quality'].max():.1f}",
        "diverse5":         diverse5,
        "provider_options": provider_options(df),
        "model_options":    model_options(df),
        "p75":              round(float(df["quality"].quantile(0.75)), 1),
        "p90":              round(float(df["quality"].quantile(0.90)), 1),
        "generated":        datetime.now().strftime("%b %d  %H:%M"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return written


def copy_css():
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "assets" / "style.css", DOCS / "assets" / "style.css")


def main():
    export_default_figures(FIG)
    copy_css()
    (DOCS / ".nojekyll").write_text("")
    # build_pybundle() added in Task 9.
    print("Static build complete →", DOCS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_static.py -v`
Expected: PASS. Also manually: `python build_static.py` prints "Static build complete".

- [ ] **Step 5: Commit**

```bash
git add build_static.py tests/test_build_static.py docs/figures docs/assets/style.css docs/.nojekyll
git commit -m "feat(static): build script pre-renders default figures + manifest"
git push origin main
```

### Task 2: `index.html` shell + static figure rendering

**Files:**
- Create: `docs/index.html`
- Create: `docs/app.js`

**Interfaces:**
- Consumes: `docs/figures/*.json`, `docs/figures/manifest.json` from Task 1.
- Produces: a static page rendering all default figures; global object `window.AF` with `{ pyReady: false }` and a `renderFigure(divId, figId)` helper used in later phases.

The HTML must reproduce the Dash layout structure (classes from `style.css`): `.header`, `.stat-bar` (4 `.stat`), `.filters` (provider `<select multiple>`, min-score `<select>`, search `<input>`, CSV button, preset buttons), `.tabs` (10 buttons), and one panel `<div>` per tab containing the chart container `<div>`s plus that tab's local controls. Include the detail panel markup with ids `detail-panel`, `detail-panel-body`, `detail-close`, `detail-add-compare`.

- [ ] **Step 1: Write `docs/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Frontier</title>
  <link rel="stylesheet" href="assets/style.css" />
  <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
</head>
<body>
  <div class="header">
    <div class="header-left"><h1>AI FRONTIER</h1>
      <span class="subtitle">LLM comparison dashboard</span></div>
    <div style="display:flex;align-items:center;gap:8px">
      <button id="btn-refresh" class="header-icon-btn" title="Reload">⟳</button>
      <button id="btn-share" class="header-icon-btn" title="Copy URL">↗</button>
    </div>
  </div>

  <div class="stat-bar">
    <div class="stat"><div id="stat-model-count" class="stat-value">—</div><div class="stat-label">Models tracked</div></div>
    <div class="stat"><div id="stat-provider-count" class="stat-value">—</div><div class="stat-label">Providers</div></div>
    <div class="stat"><div id="stat-floor-price" class="stat-value" style="color:#00d4ff">—</div><div class="stat-label">Floor price / 1M</div></div>
    <div class="stat"><div id="stat-peak-quality" class="stat-value">—</div><div class="stat-label">Peak intelligence</div></div>
  </div>

  <div class="filters">
    <span class="filter-label">PROVIDER</span>
    <select id="filter-provider" multiple style="min-width:220px"></select>
    <div class="filter-sep"></div>
    <span class="filter-label">MIN SCORE</span>
    <select id="filter-quality">
      <option value="0">≥ 0</option><option value="10">≥ 10</option>
      <option value="15">≥ 15</option><option value="20">≥ 20</option>
      <option value="25">≥ 25</option><option value="30">≥ 30</option>
      <option value="35">≥ 35</option><option value="40">≥ 40</option>
      <option value="45">≥ 45</option><option value="50">≥ 50</option>
    </select>
    <div class="filter-sep"></div>
    <span class="filter-label">SEARCH</span>
    <input id="model-search" class="search-input" type="text" placeholder="model or provider…" />
    <div class="filter-sep"></div>
    <button id="btn-export" class="export-btn">↓ CSV</button>
    <div style="flex:1"></div>
    <button id="preset-all" class="preset-btn">All</button>
    <button id="preset-strong" class="preset-btn">Top 25%</button>
    <button id="preset-elite" class="preset-btn">Top 10%</button>
  </div>

  <div class="tabs" id="tabs"></div>
  <div id="tab-panels"></div>

  <div id="detail-panel" class="detail-panel">
    <button id="detail-close" class="detail-close">×</button>
    <div id="detail-panel-body"></div>
    <button id="detail-add-compare" class="detail-add-btn">+ Add to Compare</button>
  </div>

  <div id="py-status" class="py-status">warming up interactivity…</div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `docs/app.js` (static render only this task)**

```javascript
// ---- Tab + panel definitions (chart container ids match figures/<id>.json) ----
const TABS = [
  { id: "overview",  label: "Overview",    charts: ["pareto"] },
  { id: "recommend", label: "Agent Stack", charts: [] },           // cards, Phase 4
  { id: "landscape", label: "Landscape",   charts: ["treemap", "provider_leaderboard"] },
  { id: "rankings",  label: "Rankings",    charts: ["rankings", "value_leaders"] },
  { id: "compare",   label: "Compare",     charts: ["radar"] },
  { id: "budget",    label: "Budget",      charts: ["cost_calc"] },
  { id: "table",     label: "Table",       charts: [] },           // table, Phase 4
  { id: "local",     label: "Run Local",   charts: ["local_scatter", "local_compat"] },
  { id: "image",     label: "Image Gen",   charts: ["image_faceted"] },
  { id: "video",     label: "Video Gen",   charts: ["video_rankings", "video_scatter"] },
];

window.AF = { pyReady: false, figCache: {}, manifest: null, state: {
  providers: [], minQuality: 0, search: "", tab: "overview" } };

const PLOT_CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] };

function buildTabsAndPanels() {
  const tabsEl = document.getElementById("tabs");
  const panelsEl = document.getElementById("tab-panels");
  TABS.forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === 0 ? " tab--selected" : "");
    b.textContent = t.label; b.dataset.tab = t.id;
    b.onclick = () => switchTab(t.id);
    tabsEl.appendChild(b);

    const panel = document.createElement("div");
    panel.id = "panel-" + t.id;
    panel.style.display = i === 0 ? "block" : "none";
    panel.innerHTML = t.charts.map(c =>
      `<div class="chart-card"><div id="chart-${c}" style="min-height:400px"></div></div>`
    ).join("");
    panelsEl.appendChild(panel);
  });
}

function switchTab(id) {
  window.AF.state.tab = id;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("tab--selected", b.dataset.tab === id));
  TABS.forEach(t => {
    document.getElementById("panel-" + t.id).style.display = t.id === id ? "block" : "none";
  });
  // Plotly needs a resize when a hidden plot becomes visible.
  setTimeout(() => document.querySelectorAll("#panel-" + id + " .js-plotly-plot")
    .forEach(el => { try { Plotly.Plots.resize(el); } catch (e) {} }), 60);
}

async function renderFigure(divId, figId) {
  let fig = window.AF.figCache[figId];
  if (!fig) {
    const r = await fetch(`figures/${figId}.json`);
    fig = await r.json();
    window.AF.figCache[figId] = fig;
  }
  Plotly.react(divId, fig.data, fig.layout, PLOT_CONFIG);
}

async function loadManifest() {
  const m = await (await fetch("figures/manifest.json")).json();
  window.AF.manifest = m;
  document.getElementById("stat-model-count").textContent = m.model_count;
  document.getElementById("stat-provider-count").textContent = m.provider_count;
  document.getElementById("stat-floor-price").textContent = m.floor_price;
  document.getElementById("stat-peak-quality").textContent = m.peak_quality;
  const sel = document.getElementById("filter-provider");
  m.provider_options.forEach(o => {
    const opt = document.createElement("option");
    opt.value = o.value; opt.textContent = o.label; sel.appendChild(opt);
  });
}

async function init() {
  buildTabsAndPanels();
  await loadManifest();
  for (const t of TABS) for (const c of t.charts) await renderFigure("chart-" + c, c);
}
init();
```

- [ ] **Step 3: Serve and verify in the browser**

Run: `python -m http.server -d docs 8000`
Open `http://localhost:8000/`. Verify with a screenshot (Playwright `browser_navigate` + `browser_take_screenshot`):
- Header, stat bar populated from manifest, all 10 tabs clickable.
- Overview shows the Pareto chart; Landscape shows treemap + leaderboard; each tab's default charts render and are hover/zoom interactive.
Expected: a styled, correct-looking dashboard identical in appearance to the Dash app's default state.

- [ ] **Step 4: Commit**

```bash
git add docs/index.html docs/app.js
git commit -m "feat(static): index shell renders all default figures statically"
git push origin main
```

---

## Phase 2 — Make `stack_recommender` Pyodide-safe

### Task 3: Split selection from rendering in `stack_recommender.py`

**Files:**
- Modify: `components/stack_recommender.py`
- Create: `tests/test_stack_recommender_html.py`

**Interfaces:**
- Produces: `build_stack_cards_html(df, providers, mode="api", local_df=None) -> str` returning a self-contained HTML string with the same tier cards and inline styles as the dash version. Existing `build_stack_cards(...) -> dash component` keeps working unchanged.
- Internal: extract `select_stack(df, providers, mode, local_df) -> dict` (pure data: chosen tiers/rows) and have BOTH `build_stack_cards` and `build_stack_cards_html` consume it, so the two renderers never drift.

- [ ] **Step 1: Read the current file**

Run: open `components/stack_recommender.py`. Identify (a) the model-selection logic inside `build_stack_cards` and the `_tier_card`/`_api_row`/`_local_row`/`_chip` helpers, (b) every `html.Div/Span/Table` call.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_stack_recommender_html.py
import pandas as pd
from data.ingest import get_models
from components.stack_recommender import build_stack_cards, build_stack_cards_html

def test_html_renderer_returns_string_with_same_tiers():
    df = get_models()
    html = build_stack_cards_html(df, ["Anthropic", "Google", "OpenAI"], mode="api")
    assert isinstance(html, str) and len(html) > 200
    # Tier labels present in both renderers.
    for tier in ["Fast", "Balanced", "Reasoning"]:
        assert tier.lower() in html.lower()

def test_dash_renderer_still_works():
    df = get_models()
    comp = build_stack_cards(df, ["Anthropic"], mode="api")
    assert comp is not None  # unchanged dash component path
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_stack_recommender_html.py -v`
Expected: FAIL (`build_stack_cards_html` not defined).

- [ ] **Step 4: Refactor**

In `components/stack_recommender.py`:
1. Extract a `select_stack(df, providers, mode="api", local_df=None) -> dict` function containing the existing per-tier model picking logic (move it out of `build_stack_cards`). It returns plain data, e.g. `{"tiers": [{"name","picks": DataFrame,"source","use_cases","advice": {...}}, ...]}`.
2. Rewrite `build_stack_cards` to call `select_stack` then render via the existing `dash.html` helpers (behavior unchanged).
3. Add HTML-string mirrors of the renderers. Implement a tiny helper to avoid drift:

```python
def _h(tag, inner="", style=None, cls=None):
    s = f' style="{style}"' if style else ""
    c = f' class="{cls}"' if cls else ""
    return f"<{tag}{c}{s}>{inner}</{tag}>"

def build_stack_cards_html(df, providers, mode="api", local_df=None) -> str:
    data = select_stack(df, providers, mode=mode, local_df=local_df)
    cards = []
    for tier in data["tiers"]:
        rows_html = "".join(_row_html(r, is_top=(i == 0), source=tier["source"])
                            for i, (_, r) in enumerate(tier["picks"].iterrows()))
        cards.append(_h("div", _h("div", tier["name"], cls="stack-tier-title") + rows_html,
                        cls="stack-tier-card"))
    return _h("div", "".join(cards), cls="stack-cards")
```

Mirror the exact inline styles/markup the dash helpers used (copy the style dicts into inline `style="k:v;..."` strings). `_row_html` mirrors `_api_row`/`_local_row` depending on `source`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_stack_recommender_html.py -v`
Expected: PASS.

- [ ] **Step 6: Verify the Dash app is unbroken**

Run: `DEBUG=false python -c "import app"` (imports the module; should not error). Then briefly `python app.py` and confirm the Agent Stack tab still renders (screenshot via Playwright at `http://localhost:8050`).
Expected: Agent Stack identical to before.

- [ ] **Step 7: Commit**

```bash
git add components/stack_recommender.py tests/test_stack_recommender_html.py
git commit -m "refactor(stack): split selection from rendering; add HTML-string renderer"
git push origin main
```

---

## Phase 3 — `static_api.py` bridge + Pyodide boot + Overview interactivity

### Task 4: `static_api.py` — global filter + Overview/Landscape/Rankings/Compare/Budget/Table

**Files:**
- Create: `static_api.py`
- Create: `tests/test_static_api.py`

**Interfaces:**
- Produces (all return JSON **strings**):
  - `update_overview(providers, min_quality, search, xaxis) -> str` (figure)
  - `update_treemap(providers, min_quality, search) -> str`
  - `update_provider_leaderboard(providers, min_quality, search) -> str`
  - `update_rankings(providers, min_quality, search, sort_by) -> str`
  - `update_value_leaders(providers, min_quality, search) -> str`
  - `update_compare(providers, min_quality, search, selected_models, triggered) -> str` → `{"figure":..,"options":[..],"value":[..],"raw_table_html":".."}`
  - `update_cost_calc(monthly_tokens_m, providers, min_quality, search) -> str` (figure)
  - `update_table(providers, min_quality, search, sort_col, sort_dir) -> str` (records)
  - `export_csv(providers, min_quality, search) -> str` (CSV text)
  - `model_detail(model_name, provider) -> str` (HTML string for detail body)
- Consumes: `apply_filters`, `compute_diverse5`, `ctx_to_k`, `quality_label`, `model_options`, `provider_options` — **imported from `static_helpers`** (Task 0), never re-defined. Only `_build_raw_table_html` and `_detail_html` are new here (HTML-string mirrors of `app.py`'s dash-component `_build_raw_table` / detail body).

`static_api.py` skeleton (port of the callbacks; mirror `app.py` logic exactly):

```python
"""Browser bridge: ports the Dash callbacks to JSON-returning functions.
Runs inside Pyodide. No dash/requests/flask imports."""
import json, re
import pandas as pd

from data.ingest import get_models
from data.local_models import get_local_df, get_gpu_options, GPU_BY_NAME, QUANT_LEVELS
from data.image_models import get_image_df, get_image_providers
from data.video_models import get_video_df, get_video_providers
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap import build_treemap
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders
from components.charts.radar import build_radar
from components.charts.cost_calc import build_cost_calc
from components.charts.local_scatter import build_local_scatter
from components.charts.local_compat import build_local_compat
from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings, build_video_scatter
from components.stack_recommender import build_stack_cards_html
from static_helpers import (apply_filters, compute_diverse5, ctx_to_k,
                            quality_label, model_options, provider_options)

_DF = get_models()

def _apply_filters(providers, min_quality, search=""):
    return apply_filters(_DF, providers, min_quality, search)

# Local additions only — HTML-string mirrors of app.py's dash builders:
#   _build_raw_table_html(df, selected) ↔ app.py _build_raw_table (201-260)
#   _detail_html(row, provider)         ↔ app.py detail body (1330-1389)
# All other helpers (compute_diverse5, ctx_to_k, quality_label, model_options,
# provider_options) come from static_helpers — do NOT redefine them.

# ---- ported callbacks ----
def update_overview(providers, min_quality, search, xaxis):
    f = _apply_filters(providers, min_quality, search or "")
    fig = build_quadrant(f) if xaxis == "speed" else build_pareto_scatter(f)
    return fig.to_json()

def update_treemap(providers, min_quality, search):
    return build_treemap(_apply_filters(providers, min_quality, search or "")).to_json()

def update_provider_leaderboard(providers, min_quality, search):
    return build_provider_leaderboard(_apply_filters(providers, min_quality, search or "")).to_json()

def update_rankings(providers, min_quality, search, sort_by):
    f = _apply_filters(providers, min_quality, search or "")
    return build_rankings(f, top_n=min(25, len(f)), metric=sort_by or "intelligence").to_json()

def update_value_leaders(providers, min_quality, search):
    return build_value_leaders(_apply_filters(providers, min_quality, search or "")).to_json()

def update_compare(providers, min_quality, search, selected_models, triggered):
    f = _apply_filters(providers, min_quality, search or "")
    options = model_options(f)
    if triggered in ("filter-provider", "filter-quality", "model-search"):
        capped = compute_diverse5(f)
    else:
        capped = (list(selected_models) or [])[:5]
    return json.dumps({
        "figure": json.loads(build_radar(f, capped).to_json()),
        "options": options, "value": capped,
        "raw_table_html": _build_raw_table_html(f, capped),
    })

def update_cost_calc(monthly_tokens_m, providers, min_quality, search):
    f = _apply_filters(providers, min_quality, search or "")
    tokens = float(monthly_tokens_m) if monthly_tokens_m else 1.0
    return build_cost_calc(f, monthly_tokens_m=tokens).to_json()

def update_table(providers, min_quality, search, sort_col, sort_dir):
    f = _apply_filters(providers, min_quality, search or "").copy()
    f["value"] = f.apply(lambda r: r["quality"]/r["price"] if r["price"] > 0 else None, axis=1)
    col = sort_col or "quality"; asc = (sort_dir or "desc") == "asc"
    if col == "context":
        f["_ctx_k"] = f["context"].map(_ctx_to_k)
        f = f.sort_values("_ctx_k", ascending=asc, na_position="last")
    else:
        f = f.sort_values(col, ascending=asc, na_position="last")
    cols = ["model","provider","quality","value","price","speed","latency","context"]
    return json.dumps(f[cols].to_dict("records"))

def export_csv(providers, min_quality, search):
    return _apply_filters(providers, min_quality, search or "").to_csv(index=False)

def model_detail(model_name, provider):
    # Mirror toggle_detail_panel's body construction, emitting an HTML string.
    rows = _DF[_DF["model"] == model_name]
    if rows.empty:
        return ""
    # ... build the same provider/model/intelligence-bar/metrics markup as
    # app.py lines 1330-1389, as an HTML string. (full markup in implementation)
    return _detail_html(rows.iloc[0], provider)
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_static_api.py
import json
import static_api as api

def test_overview_returns_figure_json():
    fig = json.loads(api.update_overview([], 0, "", "price"))
    assert "data" in fig and "layout" in fig

def test_overview_speed_mode_uses_quadrant():
    # quadrant + pareto differ; just assert valid figure for speed axis.
    fig = json.loads(api.update_overview([], 0, "", "speed"))
    assert "data" in fig

def test_filters_reduce_rows_via_table():
    full = json.loads(api.update_table([], 0, "", "quality", "desc"))
    anthropic = json.loads(api.update_table(["Anthropic"], 0, "", "quality", "desc"))
    assert 0 < len(anthropic) < len(full)
    assert all(r["provider"] == "Anthropic" for r in anthropic)

def test_compare_caps_at_five_and_returns_parts():
    out = json.loads(api.update_compare([], 0, "", [], "filter-provider"))
    assert len(out["value"]) <= 5
    assert "figure" in out and "raw_table_html" in out and out["options"]

def test_export_csv_is_text():
    csv = api.export_csv(["Anthropic"], 0, "")
    assert "model" in csv.splitlines()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_static_api.py -v`
Expected: FAIL (`static_api` missing functions).

- [ ] **Step 3: Implement `static_api.py`** (complete the skeleton above; import `compute_diverse5`, `ctx_to_k`, `quality_label`, `model_options`, `provider_options` from `static_helpers` — do not redefine; implement only `_build_raw_table_html` and `_detail_html` as HTML-string mirrors of `app.py`'s `_build_raw_table` and the detail body).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_static_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add static_api.py tests/test_static_api.py
git commit -m "feat(static): static_api bridge for filters/overview/compare/table/budget"
git push origin main
```

### Task 5: `static_api.py` — Local, Image, Video, Agent Stack

**Files:**
- Modify: `static_api.py`
- Modify: `tests/test_static_api.py`

**Interfaces:**
- Produces (JSON strings):
  - `update_local(vram_per_gpu, num_gpus, quant, bandwidth_gbps, hw_type, tags) -> str` → `{"scatter":..,"compat":..}`
  - `local_hw_for_gpu(gpu_name) -> str` → `{"vram_gb":..,"bandwidth_gbps":..,"hw_type":..}`
  - `gpu_options() -> str`, `quant_levels() -> str`
  - `update_image(providers, tags) -> str` (figure)
  - `update_video(providers, tags) -> str` → `{"rankings":..,"scatter":..}`
  - `update_recommend(selected, mode, gpu_preset, vram_per_gpu, num_gpus, quant) -> str` → `{"cards_html":..,"show_providers":bool,"show_hw":bool}`

- [ ] **Step 1: Write failing tests**

```python
def test_local_returns_two_figs():
    import json, static_api as api
    out = json.loads(api.update_local(32, 1, "Q4", 1792, "nvidia", None))
    assert "scatter" in out and "compat" in out

def test_recommend_modes_toggle_rows():
    import json, static_api as api
    api_only = json.loads(api.update_recommend(["Anthropic"], "api", "NVIDIA RTX 5090", 32, 1, "Q4"))
    local = json.loads(api.update_recommend([], "local", "NVIDIA RTX 5090", 32, 1, "Q4"))
    assert api_only["show_hw"] is False and api_only["show_providers"] is True
    assert local["show_hw"] is True and local["show_providers"] is False
    assert "<" in api_only["cards_html"]

def test_video_and_image():
    import json, static_api as api
    assert "data" in json.loads(api.update_image(None, None))
    assert "rankings" in json.loads(api.update_video(None, None))
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_static_api.py -k "local or recommend or video" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (port `update_local_charts`, `update_local_hw`, `update_image_charts`, `update_video_charts`, `update_recommend` from `app.py`, including the `eff_bw` multi-GPU bandwidth math at lines 1282-1283 and the provider-resolution logic at 1106-1129; cards via `build_stack_cards_html`):

```python
def update_local(vram_per_gpu, num_gpus, quant, bandwidth_gbps, hw_type, tags):
    vram_gb = float(vram_per_gpu or 8) * int(num_gpus or 1)
    gpu_count = int(num_gpus or 1)
    eff_bw = (bandwidth_gbps or 1792) * (1 + (gpu_count - 1) * 0.85) if gpu_count > 1 else (bandwidth_gbps or 1792)
    ldf = get_local_df(quant=quant or "Q4", vram_gb=vram_gb, bandwidth_gbps=eff_bw,
                       hw_type=hw_type or "nvidia", tags=list(tags) if tags else None)
    return json.dumps({
        "scatter": json.loads(build_local_scatter(ldf, vram_gb=vram_gb, quant=quant or "Q4").to_json()),
        "compat":  json.loads(build_local_compat(ldf, quant=quant or "Q4").to_json()),
    })

def local_hw_for_gpu(gpu_name):
    g = GPU_BY_NAME.get(gpu_name)
    if not g: return json.dumps(None)
    return json.dumps({"vram_gb": g["vram_gb"], "bandwidth_gbps": g["bandwidth_gbps"], "hw_type": g["hw_type"]})

def gpu_options(): return json.dumps(get_gpu_options())
def quant_levels(): return json.dumps(list(QUANT_LEVELS))

def update_image(providers, tags):
    d = get_image_df()
    if providers: d = d[d["provider"].isin(list(providers))]
    if tags:
        for tag in tags:
            d = d[d["open_weights"] == True] if tag == "open_weights" else d[d["tags"].apply(lambda t: tag in t)]
    return build_image_faceted(d).to_json()

def update_video(providers, tags):
    d = get_video_df()
    if providers: d = d[d["provider"].isin(list(providers))]
    if tags:
        for tag in tags:
            d = d[d["open_weights"] == True] if tag == "open-weights" else d[d["tags"].apply(lambda t: tag in t)]
    paid = d[d["price_per_sec"] > 0] if not d.empty else d
    return json.dumps({"rankings": json.loads(build_video_rankings(d).to_json()),
                       "scatter": json.loads(build_video_scatter(paid if not paid.empty else d).to_json())})

def update_recommend(selected, mode, gpu_preset, vram_per_gpu, num_gpus, quant):
    mode = mode or "api"
    show_providers = mode != "local"
    show_hw = mode in ("hybrid", "hybrid2", "local")
    if mode == "local": providers = None
    elif not selected: providers = []
    elif "__all__" in selected: providers = None
    else: providers = list(selected)
    local_df = None
    if show_hw:
        meta = GPU_BY_NAME.get(gpu_preset or "", {})
        vram_gb = float(vram_per_gpu or 32) * int(num_gpus or 1)
        gpu_count = int(num_gpus or 1)
        bw = meta.get("bandwidth_gbps", 1792)
        eff_bw = bw * (1 + (gpu_count - 1) * 0.85) if gpu_count > 1 else bw
        local_df = get_local_df(quant=quant or "Q4", vram_gb=vram_gb, bandwidth_gbps=eff_bw,
                                hw_type=meta.get("hw_type", "nvidia"))
    cards = build_stack_cards_html(_DF, providers, mode=mode, local_df=local_df)
    return json.dumps({"cards_html": cards, "show_providers": show_providers, "show_hw": show_hw})
```

Note the `open_weights` vs `open-weights` tag-value difference between image and video (matches `app.py`).

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_static_api.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add static_api.py tests/test_static_api.py
git commit -m "feat(static): static_api for local/image/video/agent-stack tabs"
git push origin main
```

### Task 6: Verify whole bundle imports under a clean (dash-free) environment

**Files:** none (verification task).

- [ ] **Step 1: Check `__init__.py` files don't pull server-only deps**

Run: `grep -nE "import|from" data/__init__.py components/__init__.py components/charts/__init__.py`
If any import a scraper or `requests`/`dash`, note it; the bundle (Task 9) will ship trimmed `__init__.py` files that import nothing.

- [ ] **Step 2: Simulate the Pyodide import surface**

Run:
```bash
python - <<'PY'
import importlib, sys
# Fail if importing static_api drags in dash/requests/flask.
import static_api
bad = [m for m in sys.modules if m.split('.')[0] in ("dash","flask","requests","gunicorn","werkzeug")]
assert not bad, f"server-only modules imported: {bad}"
print("clean import surface OK")
PY
```
Expected: `clean import surface OK`. If it fails, make the offending import lazy or remove it from the modules the bundle ships.

- [ ] **Step 3: Commit** (only if `__init__.py` or imports changed)

```bash
git add -A && git commit -m "chore(static): ensure dash-free import surface for Pyodide bundle"
git push origin main
```

### Task 7: Pyodide boot + bundle build + wire global filters & all filter-driven charts

**Files:**
- Modify: `build_static.py` (add `build_pybundle()`)
- Modify: `docs/app.js` (add Pyodide boot + control wiring)

**Interfaces:**
- Consumes: `static_api.*` (JSON strings); `docs/pybundle.zip`.
- Produces: `window.AF.pyReady = true` once booted; `window.AF.callPy(fn, ...args) -> Promise<any>` (parses JSON); live updates on every control change.

- [ ] **Step 1: Add `build_pybundle()` to `build_static.py`**

```python
def build_pybundle():
    bundle = DOCS / "pybundle.zip"
    include = [
        "static_api.py", "static_helpers.py",
        "components/__init__.py", "components/stack_recommender.py",
        "components/charts",                      # whole dir
        "data/__init__.py", "data/ingest.py", "data/local_models.py",
        "data/image_models.py", "data/video_models.py", "data/embedding_models.py",
        "data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv",
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in include:
            p = ROOT / rel
            if p.is_dir():
                for f in p.rglob("*.py"):
                    z.write(f, f.relative_to(ROOT))
            else:
                z.write(p, rel)
    print("pybundle.zip:", bundle.stat().st_size, "bytes")
```
Call `build_pybundle()` from `main()`. (If `update_table` needs video CSV history it does not — only the three raw CSVs above are required.)

- [ ] **Step 2: Run the build, verify the zip**

Run: `python build_static.py && unzip -l docs/pybundle.zip | tail -5`
Expected: zip lists `static_api.py`, `components/charts/*.py`, `data/raw/aa_models.csv`, etc.

- [ ] **Step 3: Add Pyodide boot + wiring to `docs/app.js`**

```javascript
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";

async function bootPyodide() {
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = PYODIDE_URL; s.onload = res; s.onerror = rej; document.head.appendChild(s);
  });
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["pandas", "numpy", "micropip"]);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("plotly")
`);
  const buf = await (await fetch("pybundle.zip")).arrayBuffer();
  pyodide.unpackArchive(buf, "zip", { extractDir: "/bundle" });
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/bundle")
import static_api
`);
  window.AF.pyodide = pyodide;
  window.AF.callPy = async (fn, ...args) => {
    const py = pyodide.globals.get("static_api");
    const res = py[fn](...args);   // JS args auto-convert; returns JSON string
    const s = res.toString();
    if (typeof res.destroy === "function") res.destroy();
    return JSON.parse(s);
  };
  window.AF.pyReady = true;
  document.getElementById("py-status").style.display = "none";
  rerenderActiveFilterCharts();   // refresh with live (identical) data once ready
}

// Debounce helper
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

function readGlobalFilters() {
  const providers = Array.from(document.getElementById("filter-provider").selectedOptions).map(o => o.value);
  const minQuality = Number(document.getElementById("filter-quality").value);
  const search = document.getElementById("model-search").value;
  Object.assign(window.AF.state, { providers, minQuality, search });
  return [providers, minQuality, search];
}

async function renderJsonFig(divId, figObj) { Plotly.react(divId, figObj.data, figObj.layout, PLOT_CONFIG); }

async function rerenderActiveFilterCharts() {
  if (!window.AF.pyReady) return;
  const [p, q, s] = readGlobalFilters();
  const tab = window.AF.state.tab;
  if (tab === "overview") {
    const x = document.querySelector('input[name="overview-xaxis"]:checked')?.value || "price";
    renderJsonFig("chart-pareto", await window.AF.callPy("update_overview", p, q, s, x));
  } else if (tab === "landscape") {
    renderJsonFig("chart-treemap", await window.AF.callPy("update_treemap", p, q, s));
    renderJsonFig("chart-provider_leaderboard", await window.AF.callPy("update_provider_leaderboard", p, q, s));
  } else if (tab === "rankings") {
    const sort = document.querySelector('input[name="rankings-sort"]:checked')?.value || "intelligence";
    renderJsonFig("chart-rankings", await window.AF.callPy("update_rankings", p, q, s, sort));
    renderJsonFig("chart-value_leaders", await window.AF.callPy("update_value_leaders", p, q, s));
  } else if (tab === "compare") {
    await refreshCompare("filter-provider");
  } else if (tab === "budget") {
    const tok = Number(document.getElementById("budget-tokens")?.value || 1);
    renderJsonFig("chart-cost_calc", await window.AF.callPy("update_cost_calc", tok, p, q, s));
  } else if (tab === "table") {
    await refreshTable();
  }
}

function wireGlobalControls() {
  const trigger = debounce(rerenderActiveFilterCharts, 200);
  document.getElementById("filter-provider").onchange = trigger;
  document.getElementById("filter-quality").onchange = trigger;
  document.getElementById("model-search").oninput = trigger;
  document.getElementById("preset-all").onclick = () => setPreset(0, []);
  document.getElementById("preset-strong").onclick = () => setPreset(window.AF.manifest.p75, []);
  document.getElementById("preset-elite").onclick = () => setPreset(window.AF.manifest.p90, []);
}

function setPreset(minQ, providers) {
  document.getElementById("filter-quality").value = String(minQ);
  const sel = document.getElementById("filter-provider");
  Array.from(sel.options).forEach(o => { o.selected = providers.includes(o.value); });
  rerenderActiveFilterCharts();
}
```

Extend `switchTab` to call `rerenderActiveFilterCharts()` after switching (so a tab shows filtered data when entered). Call `wireGlobalControls()` and `bootPyodide()` at the end of `init()`.

- [ ] **Step 4: Verify end-to-end in browser**

Run: `python build_static.py && python -m http.server -d docs 8000`. With Playwright:
1. Load page; confirm default Pareto shows immediately and "warming up…" badge visible.
2. Wait for badge to disappear (Pyodide ready).
3. Select provider "Anthropic" → Pareto updates to Anthropic-only. Set MIN SCORE ≥ 40 → updates. Type in search → updates.
4. Switch to Rankings, toggle sort (add the radio in Task 8 if not present yet) → updates.
Compare screenshots against the live Dash app (`python app.py`) for the same filter states.
Expected: identical figures for identical filters.

- [ ] **Step 5: Commit**

```bash
git add build_static.py docs/app.js docs/pybundle.zip
git commit -m "feat(static): pyodide boot + live global-filter wiring"
git push origin main
```

---

## Phase 4 — Per-tab local controls + remaining interactivity

### Task 8: Add per-tab control rows + wire their handlers

**Files:**
- Modify: `docs/index.html` (add the per-tab control rows inside each panel — see below)
- Modify: `docs/app.js` (render Table & Agent Stack content; wire local controls)

**Interfaces:**
- Consumes: `static_api.update_compare/update_cost_calc/update_table/update_local/update_image/update_video/update_recommend/local_hw_for_gpu/gpu_options/quant_levels`.

Per-tab controls to add (match the Dash layout exactly — ids in parentheses):
- Overview: X-axis radio `price|speed` (`overview-xaxis`).
- Rankings: sort radio `intelligence|value|speed` (`rankings-sort`).
- Compare: model multiselect (`radar-model-select`, options from manifest, default `diverse5`, cap 5) + raw-table container (`compare-raw-table`).
- Budget: number input (`budget-tokens`, default 1).
- Table: sort-col select + sort-dir select (`table-sort-col`, `table-sort-dir`) + `<table id="model-table">`.
- Run Local: GPU select (`local-gpu-preset`, from `gpu_options()`), VRAM number (`local-vram`), GPUs select `1|2|4|8` (`local-num-gpus`), QUANT select (`local-quant`, from `quant_levels()`), tags multiselect (`local-tags`).
- Image Gen: provider multiselect (`image-provider-filter`) + tags multiselect (`image-tag-filter`).
- Video Gen: provider multiselect (`video-provider-filter`) + tags multiselect (`video-tag-filter`).
- Agent Stack: workflow radio (`recommend-mode`), providers checklist (`recommend-providers`), hardware row (`recommend-gpu-preset/vram/num-gpus/quant`) + `<div id="recommend-cards">`.

- [ ] **Step 1: Add the control HTML** to each panel in `index.html` (use the same `.filters` / `.filter-label` markup as Phase 1; selects/inputs with the ids above). For options sourced from Python (GPU list, quant levels), leave the `<select>` empty and populate in JS after boot.

- [ ] **Step 2: Wire handlers in `app.js`**

```javascript
async function refreshCompare(triggered) {
  const [p, q, s] = readGlobalFilters();
  const sel = document.getElementById("radar-model-select");
  const selected = Array.from(sel.selectedOptions).map(o => o.value);
  const out = await window.AF.callPy("update_compare", p, q, s, selected, triggered);
  renderJsonFig("chart-radar", out.figure);
  document.getElementById("compare-raw-table").innerHTML = out.raw_table_html;
  // sync select options/value
  if (triggered !== "radar-model-select") {
    sel.innerHTML = out.options.map(o => `<option value="${o.value}">${o.label}</option>`).join("");
    Array.from(sel.options).forEach(o => { o.selected = out.value.includes(o.value); });
  }
}

async function refreshTable() {
  const [p, q, s] = readGlobalFilters();
  const col = document.getElementById("table-sort-col").value;
  const dir = document.getElementById("table-sort-dir").value;
  const rows = await window.AF.callPy("update_table", p, q, s, col, dir);
  renderTableRows(rows);   // builds <tr> from records, provider color from a JS map
}

async function refreshLocal() {
  const hw = window.AF.localHwMeta || { bandwidth_gbps: 1792, hw_type: "nvidia" };
  const vram = Number(document.getElementById("local-vram").value || 8);
  const gpus = Number(document.getElementById("local-num-gpus").value || 1);
  const quant = document.getElementById("local-quant").value || "Q4";
  const tags = Array.from(document.getElementById("local-tags").selectedOptions).map(o => o.value);
  const out = await window.AF.callPy("update_local", vram, gpus, quant, hw.bandwidth_gbps, hw.hw_type, tags.length ? tags : null);
  renderJsonFig("chart-local_scatter", out.scatter);
  renderJsonFig("chart-local_compat", out.compat);
}
// On local-gpu-preset change: const hw = await callPy("local_hw_for_gpu", name);
//   window.AF.localHwMeta = hw; set vram input to hw.vram_gb; refreshLocal().

async function refreshImage() {
  const providers = multiVals("image-provider-filter");
  const tags = multiVals("image-tag-filter");
  renderJsonFig("chart-image_faceted", await window.AF.callPy("update_image", providers.length?providers:null, tags.length?tags:null));
}
async function refreshVideo() {
  const providers = multiVals("video-provider-filter");
  const tags = multiVals("video-tag-filter");
  const out = await window.AF.callPy("update_video", providers.length?providers:null, tags.length?tags:null);
  renderJsonFig("chart-video_rankings", out.rankings);
  renderJsonFig("chart-video_scatter", out.scatter);
}
async function refreshRecommend() {
  const mode = document.querySelector('input[name="recommend-mode"]:checked')?.value || "api";
  const providers = Array.from(document.querySelectorAll('input[name="recommend-providers"]:checked')).map(c => c.value);
  const gpu = document.getElementById("recommend-gpu-preset").value;
  const vram = Number(document.getElementById("recommend-vram").value || 32);
  const gpus = Number(document.getElementById("recommend-num-gpus").value || 1);
  const quant = document.getElementById("recommend-quant").value || "Q4";
  const out = await window.AF.callPy("update_recommend", providers, mode, gpu, vram, gpus, quant);
  document.getElementById("recommend-cards").innerHTML = out.cards_html;
  document.getElementById("recommend-providers-row").style.display = out.show_providers ? "" : "none";
  document.getElementById("recommend-hw-row").style.display = out.show_hw ? "" : "none";
}

function multiVals(id) { return Array.from(document.getElementById(id).selectedOptions).map(o => o.value); }
```

Add a `populateDynamicSelects()` that, after boot, fills `local-gpu-preset`/`recommend-gpu-preset` from `gpu_options()` and `local-quant`/`recommend-quant` from `quant_levels()`, sets defaults (`NVIDIA RTX 5090`, `Q4`), then renders each tab's first live state. Bind each tab control's `onchange`/`oninput` to its `refresh*` handler.

- [ ] **Step 3: Verify each tab in browser vs. live app**

Serve `docs/`, run `python app.py`. For each tab (Compare, Budget, Table, Run Local, Image, Video, Agent Stack), change each control and screenshot-compare against the Dash app at the same settings.
Expected: matching charts/cards/table for matching inputs.

- [ ] **Step 4: Commit**

```bash
git add docs/index.html docs/app.js
git commit -m "feat(static): per-tab controls wired to static_api"
git push origin main
```

### Task 9: Pure-JS extras — CSV export, share URL, detail panel, refresh

**Files:**
- Modify: `docs/app.js`

- [ ] **Step 1: Implement**

```javascript
// CSV export — get filtered CSV text from Python, trigger download.
document.getElementById("btn-export").onclick = async () => {
  const [p, q, s] = readGlobalFilters();
  const csv = await window.AF.callPy("export_csv", p, q, s);   // returns plain string (not JSON)
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "ai_frontier_export.csv"; a.click();
};
// NOTE: export_csv returns raw text; add a callPyRaw that does NOT JSON.parse.

// Share — copy URL with ?tab=&p=&q=
document.getElementById("btn-share").onclick = () => {
  const { tab, providers, minQuality } = window.AF.state;
  const params = new URLSearchParams();
  if (tab) params.set("tab", tab);
  if (providers.length) params.set("p", providers.join(","));
  if (minQuality > 0) params.set("q", minQuality);
  const url = location.origin + location.pathname + (params.toString() ? "?" + params : "");
  navigator.clipboard?.writeText(url);
  history.replaceState(null, "", url);
};
document.getElementById("btn-refresh").onclick = () => location.reload();

// Restore state from URL on load (tab/providers/min-score), then rerender.
function applyUrlState() {
  const u = new URLSearchParams(location.search);
  if (u.get("q")) document.getElementById("filter-quality").value = u.get("q");
  if (u.get("p")) { const set = new Set(u.get("p").split(",")); Array.from(document.getElementById("filter-provider").options).forEach(o => o.selected = set.has(o.value)); }
  if (u.get("tab")) switchTab(u.get("tab"));
}

// Detail panel — Plotly click on pareto → model_detail HTML.
function wireDetailPanel() {
  const div = document.getElementById("chart-pareto");
  div.on && div.on("plotly_click", async (ev) => {
    const cd = ev.points?.[0]?.customdata; if (!cd) return;
    const html = await window.AF.callPyRaw("model_detail", cd[0], cd[1]);
    if (!html) return;
    document.getElementById("detail-panel-body").innerHTML = html;
    document.getElementById("detail-panel").className = "detail-panel open";
    window.AF.detailModel = cd[0];
  });
  document.getElementById("detail-close").onclick = () =>
    document.getElementById("detail-panel").className = "detail-panel";
  document.getElementById("detail-add-compare").onclick = () => {
    const m = window.AF.detailModel; if (!m) return;
    const sel = document.getElementById("radar-model-select");
    const chosen = Array.from(sel.selectedOptions).map(o => o.value);
    if (!chosen.includes(m) && chosen.length < 5) {
      Array.from(sel.options).forEach(o => { if (o.value === m) o.selected = true; });
    }
    switchTab("compare"); refreshCompare("radar-model-select");
  };
}
```
Add `callPyRaw` (returns the string without `JSON.parse`). Re-wire `plotly_click` after each Pareto re-render (Plotly drops handlers on `react`). Call `applyUrlState()` and `wireDetailPanel()` post-boot.

- [ ] **Step 2: Verify**

Browser: click a Pareto bubble → detail panel opens with correct metrics; "Add to Compare" jumps to Compare with the model selected; CSV downloads and matches a filtered export from the Dash app; share copies a URL that, when reopened, restores tab + filters.

- [ ] **Step 3: Commit**

```bash
git add docs/app.js
git commit -m "feat(static): CSV export, share URL, detail panel, refresh"
git push origin main
```

---

## Phase 5 — Deploy, refresh workflow, verification, cleanup

### Task 10: Enable GitHub Pages + update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Enable Pages from `main:/docs`**

Try via API (requires `gh auth` with repo admin):
```bash
gh api -X POST repos/Avo-Sarkissian/AI-Frontier-Database/pages \
  -f 'source[branch]=main' -f 'source[path]=/docs' 2>&1 || \
echo "If this errors, enable manually: repo Settings → Pages → Source: Deploy from a branch → main → /docs → Save."
```
Verify: `gh api repos/Avo-Sarkissian/AI-Frontier-Database/pages --jq '.html_url'`
Expected: `https://avo-sarkissian.github.io/AI-Frontier-Database/`.

- [ ] **Step 2: Document in `README.md`** the static site URL, that `python build_static.py` regenerates `docs/`, and that the Dash app still runs locally with `python app.py`.

- [ ] **Step 3: Commit & push, then confirm the deploy**

```bash
git add README.md
git commit -m "docs: document static GitHub Pages site + build command"
git push origin main
```
Wait ~1–2 min, then load `https://avo-sarkissian.github.io/AI-Frontier-Database/` and confirm it renders (Playwright screenshot).

### Task 11: Point `refresh-models` at the static build

**Files:**
- Modify: `.claude/skills/refresh-models/SKILL.md` (and any script it calls)

- [ ] **Step 1: Read the skill** to find its final "commit + push so Render redeploys" step.

- [ ] **Step 2: Change the deploy step** so that after the catalogs are refreshed and verified it runs `python build_static.py`, then `git add data docs && git commit && git push origin main`. Update the wording from "Render redeploys" to "GitHub Pages auto-deploys (~1–2 min)". Keep the scrape + verify steps unchanged.

- [ ] **Step 3: Dry-run** `python build_static.py` after a no-op data change; confirm `docs/figures/*` and `docs/pybundle.zip` regenerate and the diff is sensible.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/refresh-models
git commit -m "chore(refresh): rebuild static docs + Pages deploy instead of Render"
git push origin main
```

### Task 12: Full per-tab verification vs. live Dash app

**Files:** none (verification).

- [ ] **Step 1:** Run `python app.py` (oracle) and serve `docs/`. For all 10 tabs, with the same global filter (`Anthropic`, MIN SCORE ≥ 40, a search term) and each tab's local controls, screenshot both and confirm the figures match. Record any mismatch as a bug and fix in the relevant `static_api`/`app.js` task before closing.

- [ ] **Step 2:** Confirm first paint < 1s (default figures) and that interactivity becomes live after Pyodide boots (badge disappears). Confirm a hard refresh uses cached Pyodide (faster second boot).

- [ ] **Step 3:** Note in the spec/README that the user should delete the Render service once satisfied (manual, one click in Render dashboard). Do not delete anything automatically.

---

## Self-Review

**Spec coverage:**
- Static `/docs` site + URL → Tasks 2, 10. ✓
- Instant first paint + background hydration → Tasks 1–2 (defaults), 7 (boot). ✓
- `static_api` bridge (all 19 callbacks) → Tasks 4, 5, 9 (export/detail). ✓ (URL-sync/refresh/resize clientside callbacks → Task 9 / switchTab.)
- `build_static.py` + bundle + CSS copy → Tasks 1, 7. ✓
- `stack_recommender` refactor → Task 3. ✓
- Engine compatibility / dash-free surface → Task 6. ✓
- Deployment + refresh workflow → Tasks 10, 11. ✓
- Verification vs. live app → Tasks 7, 8, 12. ✓
- Render left running until verified → Task 12 Step 3. ✓

**Code reuse (resolved):** Pure helpers live once in `static_helpers.py` (Task 0), imported by `app.py`/`static_api.py`/`build_static.py` — no duplication. `app.py` change is delegation-only (call sites unchanged).

**Placeholder scan:** Browser-render tasks use explicit "serve + screenshot-compare vs. live app" verification (no pytest possible there); `model_detail`/`_build_raw_table_html`/`_detail_html` reference `app.py` exact line ranges to mirror — implementer copies that markup. No "TBD/TODO".

**Type consistency:** `window.AF.callPy` returns parsed JSON; `callPyRaw` returns the string (used by `export_csv`, `model_detail`). `static_api` functions all return `str`. Figure-bearing returns are objects with `.data`/`.layout`. `update_compare`/`update_local`/`update_video`/`update_recommend` return composite objects with the keys consumed in `app.js`. Tab ids and chart-container ids (`chart-<figId>`) are consistent between `TABS`, `build_static.py` figure ids, and the wiring.
