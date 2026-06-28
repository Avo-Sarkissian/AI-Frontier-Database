# Live Data Refresh — Design Spec

**Date:** 2026-06-28
**Status:** Approved for planning
**Topic:** Auto-pull latest model data on page open + a working Refresh button, with an hourly server-side refresh bot keeping the published snapshot fresh.

---

## 1. Goal

Make the public dashboard show the **latest model data every time it is opened**, and give visitors a **Refresh** button that pulls the newest published data on demand — *without standing up any proxy or per-visitor live connection to Artificial Analysis (AA)*.

Two user-facing promises:

1. **Open the site → see the latest data.** A freshly loaded page always reflects the most recent published snapshot, never a stale browser/CDN cache.
2. **Press Refresh → pull the latest.** The button fetches the newest published snapshot and updates the page; if nothing is newer, it says so.

---

## 2. Key constraint (why there is no proxy)

A browser running on `avo-sarkissian.github.io` **cannot fetch from AA directly.** Verified on 2026-06-28: the AA endpoint the scrapers use
(`https://artificialanalysis.ai/api/data/website/host-models/performance`) returns **no `Access-Control-Allow-Origin` header**, so any cross-origin `fetch()` from the site is blocked by the browser's same-origin policy. (`curl` succeeds because curl is not a browser and ignores CORS.) There is no client-side workaround — `mode: 'no-cors'` yields an unreadable opaque response, and AA supports no JSONP.

**Consequence:** the *only* component allowed to talk to AA is a **server-side job** (GitHub Actions runner), where CORS does not apply. Everything a visitor's browser fetches is served from the site's **own GitHub Pages origin** (same-origin, no proxy). Therefore "live data" means **the latest hourly publish**, re-fetched from our own origin on page open and on button press — not a per-click hit to AA. For benchmark data that moves on the order of days, an hourly republish is effectively live.

No API key is involved — the scrapers authenticate only with a browser `User-Agent`, so nothing secret is exposed in either component.

---

## 3. Non-goals (explicitly out of scope)

- **No CORS proxy / no per-visitor live AA fetch.** Deliberately excluded per the requirement to avoid extra infrastructure. Section 11 documents how to add it later without reworking this design.
- **No change to scrape sources or chart logic.** We reuse the existing scrapers, `build_static.py`, and chart builders unchanged in behavior.
- **No in-place Pyodide hot-swap of the dataset.** The Refresh button uses a clean cache-busted reload (Section 6.3) rather than mutating `static_api._DF` in a live interpreter — simpler and immune to half-updated states.
- **Video / embedding catalogs are static** (`data/video_models.py`, `data/embedding_models.py` are hardcoded, not scraped). "Everything" in scope = the three *scraped* catalogs (hosted LLMs, open-weight/local, image). Video charts are rebuilt but their data only changes via code.

---

## 4. Architecture overview

```
┌─────────────────────────── GitHub (server side) ───────────────────────────┐
│  .github/workflows/refresh.yml   (cron: hourly + manual dispatch)           │
│     1. scrape AA  → data/raw/*.csv   (3 scrapers; fall back on failure)     │
│     2. CHANGE GUARD: did data/raw/ actually change?  ── no ──▶ stop (no-op) │
│     3. build_static.py --data-only  → docs/figures/*.json, manifest.json,   │
│                                        CSVs swapped into docs/pybundle.zip  │
│     4. commit + push  (only when changed)                                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                    │ push to main → GitHub Pages auto-deploys
                                    ▼
┌─────────────────────────── Visitor's browser ──────────────────────────────┐
│  docs/app.js                                                                │
│     • on load: fetch manifest.json (no-store) → version + generated_iso     │
│                load figures + pybundle with ?v=<version> (always latest)    │
│                render "Updated X ago" badge                                 │
│     • Refresh button: re-fetch manifest (no-store);                         │
│                newer version → reload at ?v=<version> (fresh everything)    │
│                same version  → toast "Already up to date"                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Two independent components, connected only by the published files in `docs/`:

- **Component 1 — the refresh bot** is the sole AA caller; it keeps `docs/` fresh.
- **Component 2 — the site** never calls AA; it just always loads the freshest published `docs/`.

---

## 5. Component 1 — Hourly refresh bot

### 5.1 File: `.github/workflows/refresh.yml`

```yaml
name: Refresh model data
on:
  schedule:
    - cron: "23 * * * *"        # hourly at :23 (off the top of the hour — GitHub delays :00 jobs)
  workflow_dispatch:            # manual "Run workflow" button
permissions:
  contents: write               # required to push regenerated docs/
concurrency:
  group: refresh-data
  cancel-in-progress: false     # let an in-flight push finish before the next run
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      # plotly is pinned for reproducible figure JSON (matches the version shim in app.js)
      - run: pip install "plotly==6.5.2"
      - name: Scrape AA (failures fall back to cache, never fail the job)
        run: |
          python -m data.scraper       || echo "::warning::hosted scrape failed; kept cache"
          python -m data.local_scraper || echo "::warning::local scrape failed; kept cache"
          python -m data.image_scraper || echo "::warning::image scrape failed; kept cache"
      - name: Change guard — stop if no data moved
        id: guard
        run: |
          if git diff --quiet -- data/raw/; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
            echo "No data change — skipping rebuild and commit."
          else
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi
      - name: Sanity check (only if changed)
        if: steps.guard.outputs.changed == 'true'
        run: |
          python -c "
          import pandas as pd, sys
          for f in ['data/raw/aa_models.csv','data/raw/aa_local_models.csv','data/raw/aa_image_models.csv']:
              df = pd.read_csv(f)
              assert len(df) > 50, f'{f} only {len(df)} rows — refusing to publish'
          print('sanity OK')
          "
      - name: Rebuild static site (data-only) (only if changed)
        if: steps.guard.outputs.changed == 'true'
        run: python build_static.py --data-only
      - name: Commit + push (only if changed)
        if: steps.guard.outputs.changed == 'true'
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/raw docs
          git commit -m "data: hourly AA refresh ($(date -u +%Y-%m-%dT%H:%MZ))"
          git push
```

### 5.2 Behavior notes

- **Trigger:** hourly cron + `workflow_dispatch` (a "Run workflow" button in the Actions tab for on-demand server refresh).
- **Change guard:** gates on `git diff -- data/raw/`. If the scrape produced identical CSVs (AA unchanged, or all scrapers fell back), the job stops with **zero commits**. This is why "hourly" is cheap — quiet hours are no-ops.
  - The scrapers write one **daily history snapshot** (`data/raw/history/aa_models_<date>.csv`) the first time they run each calendar day. That guarantees ≥1 commit/day even on identical data, which is intended — the Trends tab needs daily points.
- **Sanity check:** refuses to publish a catastrophic scrape (any catalog under 50 rows or unparseable). On failure the job errors (red ✗ surfaces to the maintainer) **before** committing, so bad data is never published.
- **Failure tolerance:** each scraper already swallows network errors and leaves its CSV untouched (returns non-zero only on hard error; the `|| echo ::warning::` keeps the job green for transient AA hiccups). A failed scrape simply means "no change" for that catalog.
- **Free:** public repo ⇒ unlimited Actions minutes.
- **Pages redeploy:** pushing to `main` (even with the default `GITHUB_TOKEN`) triggers GitHub Pages' own build-and-deploy for the `docs/` folder. (Note: token pushes intentionally do *not* re-trigger `on: push` workflows, so there is no refresh loop.)

### 5.3 File: `build_static.py` — add `--data-only` mode

**Why:** the existing `main()` rebuilds `pybundle.zip` by re-vendoring `plotly` from site-packages. Re-vendoring on every hourly run risks a ~4 MB bundle rewrite whose bytes drift with the installed plotly version (known gotcha). The bot does **not** need to re-vendor — only the data changed.

Add a data-only path that:

1. `export_default_figures(FIG)` — regenerate `docs/figures/*.json` + `manifest.json` from the fresh CSVs (uses runtime plotly only to call `fig.to_json()`; nothing is vendored).
2. `copy_css()` — cheap, idempotent.
3. **`swap_bundle_csvs()`** (new helper) — open the existing committed `docs/pybundle.zip`, copy every entry through to a new zip **except** the three `data/raw/*.csv` members, then add the fresh CSVs. Leaves the vendored plotly bytes byte-for-byte identical. (Python `zipfile` cannot replace members in place, so rewrite-with-substitution is the standard approach.)

```python
# build_static.py
import sys

DATA_CSVS = ["data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv"]

def swap_bundle_csvs():
    """Replace the 3 data CSVs inside the existing docs/pybundle.zip without re-vendoring plotly."""
    bundle = DOCS / "pybundle.zip"
    if not bundle.exists():
        raise RuntimeError("pybundle.zip missing — run a full `python build_static.py` first.")
    tmp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(bundle) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in DATA_CSVS:
                continue                                   # drop stale copy
            zout.writestr(item, zin.read(item.filename))   # pass everything else through
        for rel in DATA_CSVS:
            zout.write(ROOT / rel, rel)                    # add fresh copy
    tmp.replace(bundle)

def rebuild_data_only():
    export_default_figures(FIG)   # figures/*.json + manifest.json from fresh CSVs
    copy_css()
    swap_bundle_csvs()
    print("Data-only rebuild complete →", DOCS)

if __name__ == "__main__":
    if "--data-only" in sys.argv:
        rebuild_data_only()
    else:
        main()                    # full build re-vendors plotly (run locally / on code changes)
```

`main()` is unchanged and remains the full rebuild for local use and whenever chart code or plotly changes.

### 5.4 File: `build_static.py` — manifest version fields

`export_default_figures()` already writes `manifest.json`. Add two fields so the site can cache-bust and show freshness:

```python
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
manifest["version"]       = now.strftime("%Y%m%dT%H%M%SZ")   # URL-safe cache-bust token
manifest["generated_iso"] = now.isoformat()                  # for client-side "X ago"
# existing "generated" (human "Jun 28 14:17") is kept for display
```

Using a UTC token avoids the spaces in the existing `generated` string. Because the bot only rebuilds when `data/raw/` changed, `version` advances only on real data changes — so version-stamped asset URLs stay cache-stable between real updates.

---

## 6. Component 2 — Site autopull + Refresh button

All changes are in `docs/app.js` and `docs/index.html`. **No `static_api.py` changes are required.**

### 6.1 Cache-busting + always-fresh load

`docs/app.js`:

- `loadManifest()` fetches `figures/manifest.json` with **`{ cache: "no-store" }`** so a freshly opened page always learns the true latest `version` (bypassing browser/CDN cache on this one tiny file). Store `window.AF.version = m.version` and `window.AF.generatedIso = m.generated_iso`.
- Append `?v=${window.AF.version}` to every subsequent data-asset fetch:
  - `renderFigure()` → `fetch(\`figures/${figId}.json?v=${window.AF.version}\`)`
  - Pyodide bundle → `fetch(\`pybundle.zip?v=${window.AF.version}\`)`
- `init()` already loads the manifest **before** figures and pybundle, so the ordering is correct: learn version → load everything stamped to that version. A freshly opened page therefore renders the latest published snapshot with no reload.

> Cache note: GitHub Pages (Fastly) varies/revalidates on a changed query string, and the underlying file content also changes on each real publish (ETag revalidation returns fresh). `no-store` on the manifest is the authoritative signal. Confirm in the rollout step that a new publish is visible on hard-and-soft loads.

### 6.2 "Updated X ago" badge

`docs/index.html`: add a small element next to the existing Refresh button, e.g. `<span id="data-freshness" class="freshness"></span>`.

`docs/app.js`: a `renderFreshness()` helper computes the delta from `window.AF.generatedIso` → "Updated 12 min ago" / "Updated 2 hr ago", with the absolute UTC time as a `title` tooltip. Call it after `loadManifest()` and on a 60 s `setInterval`. Style via `assets/style.css` (muted, matches the minimalist theme — small uppercase caption, low-contrast text).

### 6.3 Refresh button (replaces the current `location.reload()`)

Today `btn-refresh` is `onclick = () => location.reload()` (app.js:561). Replace with a smart check:

```js
async function doRefresh() {
  const btn = document.getElementById("btn-refresh");
  btn.classList.add("is-loading");
  try {
    const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
    if (m.version && m.version !== window.AF.version) {
      // New data published — reload everything cleanly, stamped to the new version.
      const u = new URL(location.href);
      u.searchParams.set("v", m.version);
      location.replace(u.toString());          // fresh figures + pybundle + Pyodide reboot
    } else {
      toast("Already up to date — " + relativeTime(window.AF.generatedIso));
      btn.classList.remove("is-loading");
    }
  } catch (e) {
    console.error("refresh failed:", e);
    location.reload();                          // safe fallback
  }
}
document.getElementById("btn-refresh").onclick = doRefresh;
```

**Why a reload instead of in-place swap:** a cache-busted reload re-fetches the version-stamped figures **and** the pybundle (whose embedded CSVs carry the fresh data for *all* catalogs), then re-boots Pyodide — guaranteeing every tab (LLM, image, local, video) reflects the new snapshot with zero risk of a partially-updated interpreter. It only happens when there is genuinely new data; otherwise the button is an instant "up to date" toast. This matches the user's mental model: Refresh = refresh.

`toast()` is a tiny transient message element (add to `index.html` + minimal CSS); `relativeTime()` is shared with the freshness badge.

### 6.4 The `?v=` URL param

`applyUrlState()` already ignores unknown params, so a `?v=<version>` left in the URL after a refresh is harmless. On the next load, `loadManifest()` (no-store) re-establishes the true latest version regardless of the URL value, so the param never pins a stale version.

---

## 7. Data flow

**Page open**
1. `index.html` loads `app.js`.
2. `loadManifest()` → `manifest.json` (no-store) → `version`, `generated_iso`, stat counts.
3. Figures + `pybundle.zip` fetched with `?v=<version>` → latest snapshot painted.
4. Freshness badge renders "Updated X ago". Pyodide boots for live filtering using the version-stamped bundle's CSVs.

**Refresh button**
1. `doRefresh()` → `manifest.json` (no-store).
2. Newer `version` → `location.replace(?v=<version>)` → full fresh load (flow above).
3. Same `version` → "Already up to date" toast, no reload.

**Hourly bot**
1. Cron → scrape → change-guard → (if changed) sanity → `build_static.py --data-only` → commit + push → Pages redeploys → new `manifest.version` is live within ~1 min.

---

## 8. File-by-file change list

| File | Change | Type |
|---|---|---|
| `.github/workflows/refresh.yml` | New hourly + dispatch workflow (scrape → guard → sanity → data-only build → commit) | new |
| `build_static.py` | Add `--data-only` mode, `swap_bundle_csvs()`, and `version`/`generated_iso` manifest fields | edit |
| `docs/app.js` | `no-store` manifest fetch; `?v=` cache-busting on figures + pybundle; freshness badge; `doRefresh()`; `relativeTime()`/`toast()` helpers | edit |
| `docs/index.html` | Add `#data-freshness` badge + toast container next to Refresh button | edit |
| `docs/assets/style.css` (+ source `assets/style.css`) | Styles for freshness caption, button loading state, toast | edit |
| `tests/test_build_static.py` | Assert manifest has `version` + `generated_iso`; assert `--data-only` swaps CSVs and preserves non-CSV bundle members | edit |

---

## 9. Error handling

- **Bot — scrape failure:** scraper keeps the old CSV; change-guard sees no diff for that catalog; job is a no-op (green). Logged as a `::warning::`.
- **Bot — bad data:** sanity check (<50 rows / unparseable) errors the job **before** commit; nothing is published.
- **Bot — push race:** `concurrency: cancel-in-progress: false` lets a running push finish; hourly runs (~2-3 min) do not overlap in practice.
- **Site — manifest fetch fails on load:** fall back to current behavior (cached/bundled snapshot renders); badge shows the last known time or hides.
- **Site — refresh fetch fails:** `doRefresh()` catches and falls back to a plain `location.reload()`; the page is never left broken.

---

## 10. Testing strategy

**Automated (pytest, extends existing suite):**
- `build_static.py`: after `rebuild_data_only()`, `manifest.json` contains URL-safe `version` (matches `^\d{8}T\d{6}Z$`) and ISO `generated_iso`.
- `swap_bundle_csvs()`: the rebuilt `pybundle.zip` (a) contains the three CSVs with the new bytes, (b) preserves every non-CSV member unchanged (compare name set + a sampled member's bytes against the original), (c) stays a valid zip Pyodide can open.
- Guard logic: a small unit check that identical CSV input ⇒ `swap_bundle_csvs` produces a bundle whose non-CSV members are byte-identical (proves no plotly drift).

**Manual verification (rollout):**
1. Run the workflow via `workflow_dispatch`; confirm a commit appears **only** when `data/raw/` changed, and authored by `github-actions[bot]`.
2. Force a data change locally, push, and confirm the live site's badge advances and figures update after a normal open (no manual cache clear).
3. With the site open on an old version, publish a new version, click **Refresh** → page reloads to the new data. Click again immediately → "Already up to date" toast, no reload.
4. Confirm Actions stays green across an hour with no AA changes (no-op runs, zero commits).

---

## 11. Future extension — true on-click live (if ever wanted)

If literal click-straight-from-AA (not "latest hourly publish") is ever desired, add **only** a tiny CORS proxy (~20-line Cloudflare Worker / Vercel / Deno function) that re-serves the AA endpoint with `Access-Control-Allow-Origin`. Then `doRefresh()` could optionally `fetch` AA through it, hand the ~20 MB JSON to Pyodide, and call a new `static_api.reload_models(json_text)` that re-parses via the existing `data.ingest.load_from_raw` (after relocating the pure `_parse_api_response` out of `data/scraper.py`, which imports `requests`, into the requests-free `data/ingest.py`). **This design requires none of that** and is forward-compatible with it.

---

## 12. Confirmed preconditions

- **GitHub Pages source confirmed:** `build_type: legacy`, serving `main` → `/docs`. A branch push (including one by the workflow's default `GITHUB_TOKEN`) auto-triggers the Pages "build-and-deploy" — so the bot needs no separate Pages Actions workflow.
- **plotly pin confirmed:** the local `ai-frontier` venv runs `plotly 6.5.2` (matches the app.js version shim), so CI pins `plotly==6.5.2` for reproducible figure JSON.
