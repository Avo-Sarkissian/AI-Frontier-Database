# Live Data Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the published dashboard data fresh via an hourly GitHub Actions bot, and make the site auto-load the latest snapshot on open plus pull it on demand via the Refresh button.

**Architecture:** Two independent components joined only by the files in `docs/`. (1) A scheduled Action is the *sole* caller of the Artificial Analysis API (server-side, where CORS doesn't block it); it scrapes, rebuilds `docs/` data-only, and commits only when data changed. (2) The static site fetches its own origin's `manifest.json` with `no-store`, version-stamps all data assets so a fresh open always shows the latest publish, displays an "Updated X ago" badge, and rewires the Refresh button to reload to a newer version (or toast "up to date").

**Tech Stack:** Python 3.11, pandas, plotly 6.5.2 (pinned), GitHub Actions, vanilla browser JS (Pyodide), pytest.

## Global Constraints

- **No CORS proxy and no per-visitor AA fetch.** The browser never calls AA. Only the Action does. (Verified: AA sends no `Access-Control-Allow-Origin`.)
- **No `static_api.py` changes.** The refresh path is a cache-busted reload, not an in-interpreter dataset swap.
- **plotly pinned to `6.5.2`** everywhere it matters (CI install + the app.js version shim already says `6.5.2`).
- **The hourly bot must commit only when `data/raw/` actually changed** (change-guard).
- **The bot must NOT re-vendor plotly** — it uses `build_static.py --data-only`, which swaps CSVs into the existing `pybundle.zip`, leaving vendored plotly bytes identical.
- **Video & embedding catalogs are static** (hardcoded Python); only the three scraped catalogs (hosted/`aa_models`, local/`aa_local_models`, image/`aa_image_models`) change.
- **GitHub Pages** serves `main` → `/docs` (legacy branch build); a `GITHUB_TOKEN` push auto-deploys it.
- **Aggressively minimalist UI** (project aesthetic): badge/toast must be muted, low-contrast, no clutter.
- **Per project CLAUDE.md, each task's commit also pushes to `main`** (`git push origin main`). Each task leaves the site working.
- Tasks 1–2 legitimately regenerate `docs/figures/*.json`, `docs/figures/manifest.json`, and `docs/pybundle.zip`; commit those regenerated artifacts. The local `ai-frontier` venv has plotly 6.5.2, so the re-vendored bundle is byte-stable.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `build_static.py` | Add `version`/`generated_iso` to manifest; add `swap_bundle_csvs()`, `rebuild_data_only()`, and `--data-only` CLI | 1, 2 |
| `.github/workflows/refresh.yml` | Hourly + manual bot: scrape → guard → sanity → data-only build → commit/push | 3 |
| `docs/app.js` | `no-store` manifest fetch + version capture; `?v=` cache-busting; freshness badge; `doRefresh()`; `relativeTime()`/`toast()` | 4, 5, 6 |
| `docs/index.html` | `#data-freshness` badge + `#toast` container | 5, 6 |
| `docs/assets/shell.css` | Styles for freshness caption, button loading, toast | 5, 6 |
| `tests/test_build_static.py` | Manifest fields + data-only swap tests | 1, 2 |
| `tests/test_refresh_workflow.py` | Workflow wiring guard (dependency-free text asserts) | 3 |
| `tests/test_static_site_wiring.py` | app.js / index.html wiring guards | 4, 5, 6 |

> **Note on JS testing:** the repo has no JS test runner; tests are pytest. The `*_wiring.py` tests are structural regression guards (assert the wiring strings are present). Actual browser behavior is verified with the Playwright MCP / a manual load in each JS task's verification step.

---

## Task 1: Manifest `version` + `generated_iso`

**Files:**
- Modify: `build_static.py` (function `export_default_figures`, the `manifest = {...}` dict ~lines 63–77)
- Test: `tests/test_build_static.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `docs/figures/manifest.json` now contains `version` (string, regex `^\d{8}T\d{6}Z$`, UTC) and `generated_iso` (ISO-8601 UTC string). Task 4/5 read both.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_static.py`:

```python
import re
from datetime import datetime

def test_manifest_has_version_and_iso():
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    manifest = json.loads((ROOT / "docs" / "figures" / "manifest.json").read_text())
    assert re.fullmatch(r"\d{8}T\d{6}Z", manifest.get("version", "")), manifest.get("version")
    # generated_iso must parse as ISO-8601
    datetime.fromisoformat(manifest["generated_iso"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_build_static.py::test_manifest_has_version_and_iso -v`
Expected: FAIL — `KeyError: 'generated_iso'` (or version assertion fails).

- [ ] **Step 3: Write minimal implementation**

In `build_static.py`, change the top of `export_default_figures` to import the UTC clock and stamp the manifest. Replace the existing `"generated": datetime.now().strftime("%b %d  %H:%M"),` line inside the `manifest` dict and add the two new fields:

```python
# at top of build_static.py, ensure timezone is imported:
from datetime import datetime, timezone
```

```python
    # inside export_default_figures, replace the single "generated" entry with:
    _now = datetime.now(timezone.utc)
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
        "image_providers":  get_image_providers(),
        "video_providers":  get_video_providers(),
        "generated":        _now.strftime("%b %d  %H:%M"),   # human display (kept)
        "version":          _now.strftime("%Y%m%dT%H%M%SZ"),  # URL-safe cache-bust token
        "generated_iso":    _now.isoformat(),                 # for client-side "X ago"
    }
```

(The existing top-of-file import is `from datetime import datetime`; widen it to include `timezone`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_build_static.py -v`
Expected: PASS (both the existing test and the new one).

- [ ] **Step 5: Commit**

```bash
git add build_static.py tests/test_build_static.py docs/figures/manifest.json docs/figures
git commit -m "feat(static): add version + generated_iso to manifest"
git push origin main
```

---

## Task 2: `--data-only` rebuild (CSV swap, no plotly re-vendor)

**Files:**
- Modify: `build_static.py` (add module constant + two functions + `__main__` dispatch)
- Test: `tests/test_build_static.py`

**Interfaces:**
- Consumes: Task 1's manifest stamping (runs inside `export_default_figures`).
- Produces: `python build_static.py --data-only` regenerates `docs/figures/*` + `manifest.json` and replaces the three `data/raw/*.csv` members inside `docs/pybundle.zip` while leaving every other (plotly) member byte-identical. Task 3's workflow calls it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_static.py`:

```python
import zipfile

DATA_CSVS = ["data/raw/aa_models.csv", "data/raw/aa_local_models.csv", "data/raw/aa_image_models.csv"]

def test_data_only_swaps_csvs_and_preserves_plotly(tmp_path):
    # Full build first so a bundle exists.
    subprocess.run([sys.executable, "build_static.py"], cwd=ROOT, check=True)
    bundle = ROOT / "docs" / "pybundle.zip"
    with zipfile.ZipFile(bundle) as z:
        before = {i.filename: z.read(i.filename) for i in z.infolist()}
    sample_py = next(n for n in before if n.endswith(".py") and not n.startswith("data/raw/"))

    # Data-only rebuild.
    subprocess.run([sys.executable, "build_static.py", "--data-only"], cwd=ROOT, check=True)
    with zipfile.ZipFile(bundle) as z:
        after = set(z.namelist())
        # membership unchanged
        assert after == set(before)
        # a plotly/source member is byte-identical (no re-vendor drift)
        assert z.read(sample_py) == before[sample_py]
        # the 3 CSVs match the live data/raw files
        for csv in DATA_CSVS:
            assert csv in after
            assert z.read(csv) == (ROOT / csv).read_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_build_static.py::test_data_only_swaps_csvs_and_preserves_plotly -v`
Expected: FAIL — `--data-only` is ignored by current `__main__`, so the bundle's CSVs are re-vendored copies (still pass membership but `build_static.py --data-only` currently runs the FULL build; the `sample_py` byte-equality may still hold, but the run is wrong). To make the red unambiguous, this test will fail because the current `__main__` has no `--data-only` branch and a fresh full build re-vendors plotly — assert fails on `sample_py` equality is not guaranteed; the reliable failure is the missing function. Confirm failure is reported; if it passes accidentally, proceed to Step 3 anyway (implementation makes intent explicit).

- [ ] **Step 3: Write minimal implementation**

In `build_static.py`, add near the top (after `ROOT/DOCS/FIG` definitions):

```python
import sys

DATA_CSVS = [
    "data/raw/aa_models.csv",
    "data/raw/aa_local_models.csv",
    "data/raw/aa_image_models.csv",
]
```

Add these two functions (above `main()`):

```python
def swap_bundle_csvs():
    """Replace the 3 data CSVs inside docs/pybundle.zip without re-vendoring plotly."""
    bundle = DOCS / "pybundle.zip"
    if not bundle.exists():
        raise RuntimeError("pybundle.zip missing — run a full `python build_static.py` first.")
    tmp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(bundle) as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in DATA_CSVS:
                continue                                    # drop stale copy
            zout.writestr(item, zin.read(item.filename))    # pass everything else through
        for rel in DATA_CSVS:
            zout.write(ROOT / rel, rel)                     # add fresh copy
    tmp.replace(bundle)
    print("swapped CSVs into pybundle.zip")


def rebuild_data_only():
    """Data-only refresh for the hourly bot: figures + manifest + CSV swap, no plotly re-vendor."""
    export_default_figures(FIG)
    copy_css()
    swap_bundle_csvs()
    print("Data-only rebuild complete →", DOCS)
```

Replace the `__main__` block at the bottom:

```python
if __name__ == "__main__":
    if "--data-only" in sys.argv:
        rebuild_data_only()
    else:
        main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_build_static.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add build_static.py tests/test_build_static.py docs/figures docs/pybundle.zip
git commit -m "feat(static): add --data-only rebuild that swaps CSVs without re-vendoring plotly"
git push origin main
```

---

## Task 3: Hourly refresh workflow

**Files:**
- Create: `.github/workflows/refresh.yml`
- Test: `tests/test_refresh_workflow.py`

**Interfaces:**
- Consumes: Task 2's `build_static.py --data-only`; the existing scrapers `python -m data.{scraper,local_scraper,image_scraper}`.
- Produces: an automated hourly commit to `main` when `data/raw/` changes. No code depends on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_refresh_workflow.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows" / "refresh.yml"

def test_workflow_present_and_wired():
    assert WF.exists(), "refresh.yml not created"
    txt = WF.read_text()
    for needle in [
        "schedule:",
        "workflow_dispatch:",
        "contents: write",
        "pip install -r requirements.txt",
        'pip install "plotly==6.5.2"',
        "python -m data.scraper",
        "python -m data.local_scraper",
        "python -m data.image_scraper",
        "git diff --quiet -- data/raw/",
        "build_static.py --data-only",
        "git push",
    ]:
        assert needle in txt, f"workflow missing: {needle}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_refresh_workflow.py -v`
Expected: FAIL — `assert WF.exists()` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

Create `.github/workflows/refresh.yml`:

```yaml
name: Refresh model data
on:
  schedule:
    - cron: "23 * * * *"        # hourly at :23 (off the top of the hour)
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: refresh-data
  cancel-in-progress: false
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
      - name: Sanity check
        if: steps.guard.outputs.changed == 'true'
        run: |
          python -c "
          import pandas as pd
          for f in ['data/raw/aa_models.csv','data/raw/aa_local_models.csv','data/raw/aa_image_models.csv']:
              df = pd.read_csv(f)
              assert len(df) > 50, f'{f} only {len(df)} rows — refusing to publish'
          print('sanity OK')
          "
      - name: Rebuild static site (data-only)
        if: steps.guard.outputs.changed == 'true'
        run: python build_static.py --data-only
      - name: Commit + push
        if: steps.guard.outputs.changed == 'true'
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/raw docs
          git commit -m "data: hourly AA refresh ($(date -u +%Y-%m-%dT%H:%MZ))"
          git push
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_refresh_workflow.py -v`
Expected: PASS.

- [ ] **Step 5: Commit, push, and trigger one manual run**

```bash
git add .github/workflows/refresh.yml tests/test_refresh_workflow.py
git commit -m "ci: hourly AA refresh bot (scrape, guard, data-only build, commit)"
git push origin main
gh workflow run "Refresh model data"
```

- [ ] **Step 6: Verify the manual run**

Run: `gh run list --workflow "Refresh model data" --limit 1` then `gh run watch <run-id>` (or view in the Actions tab).
Expected: job succeeds (green). It either commits a refresh (if AA moved) or logs "No data change — skipping rebuild and commit." with zero commits. Confirm no error emails for transient AA hiccups.

---

## Task 4: Site — always-fresh load (no-store manifest + `?v=` cache-busting)

**Files:**
- Modify: `docs/app.js` (`loadManifest` ~line 145, `renderFigure` ~line 135, `bootPyodide` pybundle fetch ~line 201)
- Test: `tests/test_static_site_wiring.py`

**Interfaces:**
- Consumes: Task 1's `manifest.version` + `manifest.generated_iso`.
- Produces: `window.AF.version` (string) and `window.AF.generatedIso` (string) set during `loadManifest`; all data-asset fetches carry `?v=<version>`. Task 5/6 read `window.AF.version` and `window.AF.generatedIso`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_static_site_wiring.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "docs" / "app.js").read_text()
HTML = (ROOT / "docs" / "index.html").read_text()

def test_manifest_fetched_no_store_and_version_captured():
    assert 'fetch("figures/manifest.json", { cache: "no-store" })' in APP
    assert "window.AF.version = m.version" in APP
    assert "window.AF.generatedIso = m.generated_iso" in APP

def test_data_assets_are_version_busted():
    assert "figures/${figId}.json?v=" in APP
    assert "pybundle.zip?v=" in APP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py -v`
Expected: FAIL — strings not present in `app.js`.

- [ ] **Step 3: Write minimal implementation**

In `docs/app.js`, in `loadManifest()` replace the first line:

```javascript
// OLD: const m = await (await fetch("figures/manifest.json")).json();
const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
window.AF.version = m.version || "";
window.AF.generatedIso = m.generated_iso || null;
```

In `renderFigure()` replace the fetch line:

```javascript
// OLD: const r = await fetch(`figures/${figId}.json`);
const r = await fetch(`figures/${figId}.json?v=${window.AF.version || ""}`);
```

In `bootPyodide()` replace the pybundle fetch:

```javascript
// OLD: const buf = await (await fetch("pybundle.zip")).arrayBuffer();
const buf = await (await fetch(`pybundle.zip?v=${window.AF.version || ""}`)).arrayBuffer();
```

(`init()` already `await loadManifest()` before any figure/pybundle load, so `window.AF.version` is set first.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Verify in a browser**

Serve and load locally: `./ai-frontier/bin/python -m http.server -d docs 8051` then open `http://localhost:8051`. In DevTools → Network, confirm `manifest.json` is requested with `no-store` and that `figures/*.json` and `pybundle.zip` requests carry `?v=<timestamp>`. The dashboard renders normally. (Or drive it with the Playwright MCP and check `browser_network_requests`.)

- [ ] **Step 6: Commit**

```bash
git add docs/app.js tests/test_static_site_wiring.py
git commit -m "feat(site): no-store manifest + version-busted data assets for fresh loads"
git push origin main
```

---

## Task 5: Site — "Updated X ago" freshness badge

**Files:**
- Modify: `docs/index.html` (header actions, ~line 16–19), `docs/app.js` (new helpers + `init`), `docs/assets/shell.css`
- Test: `tests/test_static_site_wiring.py`

**Interfaces:**
- Consumes: `window.AF.generatedIso` from Task 4.
- Produces: a `relativeTime(iso)` helper and a `renderFreshness()` updater (also used by Task 6's toast). DOM element `#data-freshness`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_static_site_wiring.py`:

```python
def test_freshness_badge_present():
    assert 'id="data-freshness"' in HTML
    assert "function relativeTime(" in APP
    assert "function renderFreshness(" in APP
    assert "setInterval(renderFreshness, 60000)" in APP
```

(Re-read the files at top of module is fine since pytest imports the module fresh per run.)

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py::test_freshness_badge_present -v`
Expected: FAIL — strings absent.

- [ ] **Step 3: Write the implementation**

In `docs/index.html`, add the badge inside `.header-right-actions` before the buttons:

```html
    <div class="header-right-actions">
      <span id="data-freshness" class="freshness" title=""></span>
      <button id="btn-refresh" class="header-icon-btn" title="Pull latest data">⟳</button>
      <button id="btn-share" class="header-icon-btn" title="Copy URL">↗</button>
    </div>
```

In `docs/app.js`, add these helpers near the top (after the `debounce` helper is fine):

```javascript
// ---- Relative time + freshness badge ----
function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  const days = Math.round(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function renderFreshness() {
  const el = document.getElementById("data-freshness");
  if (!el) return;
  const iso = window.AF.generatedIso;
  if (!iso) { el.textContent = ""; el.title = ""; return; }
  el.textContent = "Updated " + relativeTime(iso);
  el.title = new Date(iso).toUTCString();
}
```

In `init()`, right after `await loadManifest();`, add:

```javascript
  renderFreshness();
  setInterval(renderFreshness, 60000);
```

In `docs/assets/shell.css`, add:

```css
.freshness {
  font-size: 10px;
  letter-spacing: 0.06em;
  color: var(--text-3, #666);
  font-family: Inter, -apple-system, sans-serif;
  margin-right: 4px;
  white-space: nowrap;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Verify in a browser**

Reload `http://localhost:8051`. Confirm a muted "Updated just now" (or "X min ago") sits left of the ⟳ button, with the absolute UTC time on hover. Confirm it doesn't break the header layout on a narrow window.

- [ ] **Step 6: Commit**

```bash
git add docs/index.html docs/app.js docs/assets/shell.css tests/test_static_site_wiring.py
git commit -m "feat(site): 'Updated X ago' data-freshness badge"
git push origin main
```

---

## Task 6: Site — Refresh button pulls the latest

**Files:**
- Modify: `docs/app.js` (`doRefresh`/`toast` helpers + rewire `btn-refresh` in `wireGlobalControls` ~line 561), `docs/index.html` (toast container), `docs/assets/shell.css`
- Test: `tests/test_static_site_wiring.py`

**Interfaces:**
- Consumes: `window.AF.version`, `window.AF.generatedIso`, `relativeTime()` (Task 4/5).
- Produces: final user-facing behavior. Nothing depends on it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_static_site_wiring.py`:

```python
def test_refresh_button_rewired():
    assert "async function doRefresh(" in APP
    assert '{ cache: "no-store" }' in APP            # manifest re-check
    assert 'document.getElementById("btn-refresh").onclick = doRefresh' in APP
    # old bare reload wiring is gone
    assert 'getElementById("btn-refresh").onclick = () => location.reload()' not in APP
    assert 'id="toast"' in HTML
    assert "function toast(" in APP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py::test_refresh_button_rewired -v`
Expected: FAIL — `doRefresh`/`toast` absent and old wiring still present.

- [ ] **Step 3: Write the implementation**

In `docs/index.html`, add a toast container just before the closing `</body>` (after `#py-status`):

```html
  <div id="toast" class="toast"></div>
```

In `docs/app.js`, add the helpers (near `renderFreshness`):

```javascript
// ---- Toast ----
let _toastTimer;
function toast(msg) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

// ---- Refresh: pull latest published snapshot ----
async function doRefresh() {
  const btn = document.getElementById("btn-refresh");
  btn.classList.add("is-loading");
  try {
    const m = await (await fetch("figures/manifest.json", { cache: "no-store" })).json();
    if (m.version && m.version !== window.AF.version) {
      const u = new URL(location.href);
      u.searchParams.set("v", m.version);
      location.replace(u.toString());   // fresh figures + pybundle + Pyodide reboot
      return;                            // navigating away; leave spinner on
    }
    toast("Already up to date — updated " + relativeTime(window.AF.generatedIso));
  } catch (e) {
    console.error("refresh failed:", e);
    location.reload();                   // safe fallback
    return;
  }
  btn.classList.remove("is-loading");
}
```

In `wireGlobalControls()`, replace the refresh wiring:

```javascript
// OLD: document.getElementById("btn-refresh").onclick = () => location.reload();
document.getElementById("btn-refresh").onclick = doRefresh;
```

In `docs/assets/shell.css`, add:

```css
.header-icon-btn.is-loading { opacity: 0.5; pointer-events: none; }
.toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%) translateY(8px);
  background: var(--bg-card, #15171c);
  border: 1px solid var(--border, #2a2d34);
  color: #e6e6e6;
  font-family: Inter, -apple-system, sans-serif;
  font-size: 12px;
  padding: 8px 16px;
  border-radius: 6px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.25s ease, transform 0.25s ease;
  z-index: 1000;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./ai-frontier/bin/python -m pytest tests/test_static_site_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Verify in a browser (both branches)**

With `http://localhost:8051` open:
1. Click ⟳ when no new version exists → a toast "Already up to date — updated just now" appears, no reload.
2. Simulate a new publish: edit `docs/figures/manifest.json`, bump `version` to a higher timestamp, save; click ⟳ → page reloads, URL gains `?v=<new>`, badge resets to "just now". (Restore the manifest afterward, or just rebuild.)
   (Optionally drive these with the Playwright MCP: `browser_click` the button, `browser_snapshot` for the toast, `browser_network_requests` for the reload.)

- [ ] **Step 6: Commit**

```bash
git add docs/index.html docs/app.js docs/assets/shell.css tests/test_static_site_wiring.py
git commit -m "feat(site): Refresh button pulls latest published snapshot (reload-on-newer)"
git push origin main
```

---

## Task 7: Full-suite + end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `./ai-frontier/bin/python -m pytest -q`
Expected: all tests pass (existing + the new build/workflow/wiring tests).

- [ ] **Step 2: Full local rebuild sanity**

Run: `./ai-frontier/bin/python build_static.py && ./ai-frontier/bin/python build_static.py --data-only`
Expected: both succeed; `git status` shows only expected `docs/` data changes (no plotly churn in `pybundle.zip` beyond the CSV members — verify with `git diff --stat docs/pybundle.zip` showing a small delta).

- [ ] **Step 3: Confirm the live deploy**

After the Task 3–6 pushes have deployed, open `https://avo-sarkissian.github.io/AI-Frontier-Database/`. Confirm: badge shows "Updated X ago", ⟳ behaves per Task 6, and DevTools shows `?v=` on data assets. Confirm the hourly bot's most recent run in the Actions tab is green.

- [ ] **Step 4: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "chore: live-data-refresh verification fixups"
git push origin main
```
(Skip if nothing changed.)

---

## Self-Review

- **Spec §5 (bot):** Task 3 (workflow) + Task 2 (`--data-only`) + Task 1 (manifest). ✓
- **Spec §6.1 (cache-bust/no-store):** Task 4. ✓
- **Spec §6.2 (badge):** Task 5. ✓
- **Spec §6.3 (Refresh button):** Task 6. ✓
- **Spec §5.3 (no plotly re-vendor):** Task 2 `swap_bundle_csvs` + its byte-equality test. ✓
- **Spec §9 (error handling):** workflow `|| echo ::warning::` + sanity gate (Task 3); `doRefresh` catch→reload fallback (Task 6); manifest load failure tolerated via `|| ""`/`|| null` (Task 4). ✓
- **Spec §10 (tests):** manifest fields (T1), swap preservation (T2), workflow wiring (T3), site wiring (T4–6), full suite (T7). ✓
- **Type consistency:** `window.AF.version` (string) and `window.AF.generatedIso` (string|null) defined in Task 4, consumed identically in Tasks 5–6; `relativeTime`/`renderFreshness`/`toast`/`doRefresh` names consistent across tasks. ✓
- **No proxy / no static_api change:** honored throughout (Section 11 of spec is future-only). ✓
