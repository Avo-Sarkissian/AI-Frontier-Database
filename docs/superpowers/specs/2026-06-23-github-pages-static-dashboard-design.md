# AI Frontier — Static GitHub Pages Migration (Pyodide)

**Date:** 2026-06-23
**Status:** Design — awaiting approval
**Author:** Avo Sarkissian (with Claude Code)

## Problem

The dashboard is a Plotly **Dash** app — a Python server (gunicorn, 19 server-side
`@callback`s, hourly background scrapers). It is currently hosted on Render. The
goal is to host it at a **github.io URL specifically** (portfolio/showcase). GitHub
Pages serves only static files — no Python, no callbacks, no scrapers — so a direct
"move the server" is impossible.

## Decisions (locked)

1. **Target:** GitHub Pages on the existing `AI-Frontier-Database` repo. URL:
   `https://avo-sarkissian.github.io/AI-Frontier-Database/` (project pages). A
   cleaner root URL would require a repo rename to `Avo-Sarkissian.github.io` —
   out of scope unless requested.
2. **Fidelity:** Faithful rebuild — keep all 10 tabs, global filters, and the
   calculator-style tools.
3. **Engine:** **Pyodide** (CPython → WebAssembly). Run the *actual* existing
   Python chart/data code in the browser so charts are guaranteed identical to the
   current app. Only the thin UI shell is rebuilt.

## Architecture

A static single-page app in `/docs`, served by GitHub Pages from `main:/docs`.

### Key idea: instant static first paint + background Pyodide hydration

- **Build time:** `build_static.py` pre-renders every chart's *default* state to
  figure JSON. These ship as static assets.
- **Page load:** `app.js` renders the default figures with **Plotly.js** from a CDN
  immediately → full dashboard visible in < 1s, no waiting on Pyodide.
- **Background:** Pyodide boots quietly. When ready, every interactive control
  becomes "live." Until then, controls show a subtle "warming up…" state.

This removes Pyodide's only real downside (multi-second cold boot + ~30–50 MB
one-time, cached download) from the critical path.

### Components / files (all new files under `/docs` unless noted)

| File | Purpose |
|------|---------|
| `docs/index.html` | UI shell: header, stat bar, global filters, tab nav, empty chart containers. References `assets/style.css`. |
| `docs/app.js` | Renders default figures via Plotly.js; boots Pyodide; mounts Python modules + CSVs into the Pyodide FS; binds controls → `static_api` calls → `Plotly.react`. Also: tab switching, CSV export, share-URL, click-for-detail panel (all pure JS). |
| `docs/assets/style.css` | Copied verbatim from existing `assets/style.css`. |
| `docs/figures/*.json` | Pre-rendered default figure JSON per chart (generated). |
| `docs/py/` | Bundled copy of needed Python: `components/charts/*`, `components/stack_recommender.py` (refactored), `data/{ingest,local_models,image_models,video_models,embedding_models}.py`, and `static_api.py`. |
| `docs/data/raw/*.csv` | Data the loaders read, mounted into the Pyodide FS. |
| `static_api.py` (source, copied into `docs/py/`) | **The bridge.** Ports the 19 Dash callbacks into plain functions: filter dataframe → call existing `build_*` → return `fig.to_dict()`. No Dash dependency. |
| `build_static.py` (repo root) | Build: pre-render default figures, copy Python modules + CSVs + CSS into `/docs`. Run locally and by `refresh-models`. |

### The one required refactor

`components/stack_recommender.py` builds Agent-Stack cards with `from dash import
html`, which is unavailable in Pyodide. Refactor by **separating selection from
rendering**:

- Extract the pure logic that picks models per tier into a dash-free function
  (returns plain data: the chosen rows/tiers).
- Keep the existing `dash.html`-based renderer for `app.py` (unchanged behavior).
- Add an HTML-string renderer (same markup + inline styles, emitted as a string)
  used by `static_api`, injected into the page via `innerHTML`.

Result: `app.py` is untouched functionally; the static site renders identical cards
without importing dash.

### Engine compatibility (verified)

- All `components/charts/*.py` import only `plotly`, `pandas`, `numpy`, local
  constants/data — **no dash/requests**. Run in Pyodide as-is.
- Data loaders (`ingest`, `local_models`, `image_models`, `video_models`,
  `embedding_models`) do **not** import the `requests`-based scrapers. The scrapers
  stay server-side (refresh only) and never run in-browser.
- Pyodide packages: `pandas` + `numpy` are prebuilt; `plotly` installs via
  `micropip` (pure-Python figure construction; no `kaleido` needed). Plotly.js
  (CDN) renders the figure dicts; pin a version compatible with the installed
  plotly.py major version.

## Data flow

1. CSVs in `data/raw/` are the source of truth (unchanged; refreshed by scrapers
   locally as today).
2. `build_static.py` reads them, pre-renders default figures, and copies
   data + modules + CSS into `/docs`.
3. In the browser: default figures paint instantly; Pyodide loads the same CSVs and
   recomputes figures on filter changes via `static_api`.

## Deployment & refresh

- **Enable Pages:** `main` branch, `/docs` folder (via `gh api` if authenticated,
  else 2-click in repo Settings → Pages). Auto-deploys on every push to `main`.
- **`refresh-models` workflow** changes its last step from "push so Render
  redeploys" to "run `build_static.py`, commit `/docs`, push → Pages auto-deploys."
- **Render:** left running until the new site is verified, then the user deletes
  the Render service (one click). The Dash app + repo remain the local source of
  truth and still run with `python app.py`.

## Verification strategy

Run the live Dash app locally (`python app.py`, :8050) as the oracle. For each of
the 10 tabs, drive the static site with Playwright and compare against the Dash app:
default render + at least one filter interaction per tab. Tabs: Overview, Agent
Stack, Landscape, Rankings, Compare, Budget, Table, Run Local, Image Gen, Video Gen.

## Scope / non-goals

- No real-time scraping in the browser (data is build-time baked; refreshed by the
  existing workflow).
- No repo rename / custom domain (unless requested later).
- No visual redesign — reuse `style.css` verbatim.

## Risks

| Risk | Mitigation |
|------|-----------|
| Pyodide bundle size / cold boot | Instant static first paint; Pyodide hydrates in background; assets cached after first visit. |
| plotly.py ↔ Plotly.js version mismatch | Pin compatible CDN version; verify rendering in the per-tab checks. |
| `stack_recommender` refactor regresses Agent Stack | Split selection vs. rendering; keep dash-component path for `app.py`; verify both. |
| Pyodide can't find CSVs | Mount `docs/data/raw/*` into Pyodide FS at the path the loaders expect. |

## Execution phases (for the implementation plan)

1. Scaffold `/docs` shell (`index.html`, `app.js` skeleton, copied `style.css`) +
   `build_static.py` that pre-renders default figures and renders the full dashboard
   statically (no Pyodide yet). Ship a visible, correct static dashboard first.
2. Refactor `stack_recommender.py` to support an HTML-string path.
3. Boot Pyodide, mount modules + CSVs, write `static_api.py`, wire global filters +
   Overview tab interactivity end-to-end.
4. Wire remaining tabs' interactivity (Rankings, Compare, Budget, Table, Run Local,
   Image Gen, Video Gen, Agent Stack, Landscape).
5. Pure-JS extras: CSV export, share-URL, click-for-detail panel, presets, search.
6. Enable GitHub Pages; update `refresh-models`; per-tab verification vs. live app.
7. Docs/README note + (user) delete Render service.
