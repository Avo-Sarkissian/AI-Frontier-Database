"""Real-browser smoke checks for the published static site (docs/).

The unit suite proves the Python that builds the site; it cannot see the page.
Every failure this file looks for has shipped at least once behind a green
suite: a freshness badge warning about a dataset that was fine, a badge that
stayed calm over a failing one, a tab left blank by a Pyodide error that only
the console showed, a CSV button that did nothing.

Opt-in: skipped unless AIF_E2E=1 and Playwright is importable. The project
.venv has Playwright installed, so an import check alone made every local
`pytest` run boot Pyodide from the CDN 20 times and fail offline. To run it:

    uv venv <scratch>/pw-venv && uv pip install --python <scratch>/pw-venv/bin/python \\
        playwright pytest pandas
    PLAYWRIGHT_BROWSERS_PATH=<scratch>/browsers <scratch>/pw-venv/bin/python -m playwright install chromium
    PLAYWRIGHT_BROWSERS_PATH=<scratch>/browsers AIF_SMOKE_SCREENSHOTS=<dir> \\
        AIF_E2E=1 <scratch>/pw-venv/bin/python -m pytest tests/e2e -q -p no:cacheprovider

Pyodide is fetched from the jsDelivr CDN, exactly as for a visitor, so this
needs network access. AIF_SMOKE_REQUIRE_FRESH=1 additionally fails when the
SHIPPED manifest would show the warning — the pre-publish check that the data
behind the site really did refresh.
"""
from __future__ import annotations

import csv
import functools
import http.server
import io
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

if os.environ.get("AIF_E2E") != "1":
    pytest.skip("real-browser smoke checks are opt-in: set AIF_E2E=1",
                allow_module_level=True)
pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BOOT_TIMEOUT_MS = 240_000
AMBER = "rgb(251, 191, 36)"          # #fbbf24, renderFreshness's warning colour

# Tab id -> chart div ids that must hold a drawn Plotly figure. Mirrors TABS in
# docs/app.js; the tabs without charts are checked by their own containers.
TAB_CHARTS = {
    "overview":  ["pareto"],
    "landscape": ["treemap", "provider_leaderboard"],
    "rankings":  ["rankings", "value_leaders"],
    "compare":   ["radar"],
    "budget":    ["cost_calc"],
    "local":     ["local_scatter", "local_compat"],
    "image":     ["image_faceted"],
    "video":     ["video_rankings", "video_scatter"],
}


def _shots_dir(tmp_path_factory) -> Path:
    env = os.environ.get("AIF_SMOKE_SCREENSHOTS")
    out = Path(env) if env else tmp_path_factory.mktemp("smoke-shots")
    out.mkdir(parents=True, exist_ok=True)
    return out


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def site():
    if not (DOCS / "index.html").exists():
        pytest.skip("docs/ not built — run build_static.py first")
    handler = functools.partial(_Quiet, directory=str(DOCS))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def shots(tmp_path_factory):
    return _shots_dir(tmp_path_factory)


def _open(browser, site, manifest_patch=None):
    """New page with console/network problems recorded on `page.problems`."""
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000},
                              accept_downloads=True)
    page = ctx.new_page()
    problems: list[str] = []
    page.on("console", lambda m: m.type == "error" and problems.append(
        f"console: {m.text}"))
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: problems.append(
        f"requestfailed: {r.url} ({r.failure})"))
    page.on("response", lambda r: r.status >= 400 and problems.append(
        f"http {r.status}: {r.url}"))
    page.problems = problems

    if manifest_patch is not None:
        def handle(route):
            resp = route.fetch()
            body = manifest_patch(json.loads(resp.text()))
            route.fulfill(response=resp, body=json.dumps(body),
                          headers={**resp.headers, "content-type": "application/json"})
        page.route("**/figures/manifest.json*", handle)

    page.goto(f"{site}/index.html", wait_until="load")
    page.wait_for_function(
        "() => document.getElementById('data-freshness')?.textContent.startsWith('Updated')",
        timeout=30_000)
    return page


def _badge(page) -> dict:
    return page.evaluate("""() => {
        const el = document.getElementById('data-freshness');
        return {text: el.textContent, title: el.title,
                color: getComputedStyle(el).color, inline: el.style.color};
    }""")


def _wait_drawn(page, div_id: str):
    page.wait_for_function(
        """id => { const d = document.getElementById(id);
                   return d && d.data && d.data.length > 0
                       && d.querySelector('.main-svg'); }""",
        arg=div_id, timeout=60_000)


# Points Plotly actually drew. Not `trace.x.length`: plotly.py ships numeric
# arrays as {dtype, bdata} objects, so that reads 0 for every real chart.
# calcdata is Plotly's own decoded per-point list for every trace type.
_COUNT_JS = ("(id => (document.getElementById(id).calcdata || [])"
             ".reduce((n, c) => n + (c ? c.length : 0), 0))")


def _points(page, div_id: str) -> int:
    return page.evaluate(f"id => {_COUNT_JS}(id)", div_id)


# ── Live page: one Pyodide boot shared by the functional checks ──────────────

@pytest.fixture(scope="module")
def live(browser, site, shots):
    page = _open(browser, site)
    page.wait_for_function("() => window.AF && window.AF.pyReady",
                           timeout=BOOT_TIMEOUT_MS)
    status = page.evaluate("() => document.getElementById('py-status')?.textContent || ''")
    assert "unavailable" not in status, f"Pyodide failed to boot: {status}"
    _wait_drawn(page, "chart-pareto")
    page.wait_for_function("() => window.AF.drawn['chart-pareto'] === 'py'",
                           timeout=60_000)
    page.screenshot(path=str(shots / "overview.png"))
    yield page
    page.context.close()


def test_pyodide_boots_and_overview_renders_live(live):
    assert live.evaluate("() => window.AF.pyReady") is True
    assert _points(live, "chart-pareto") > 50
    count = live.inner_text("#stat-model-count")
    assert count.strip().isdigit() and int(count) > 50, count


@pytest.mark.parametrize("tab", list(TAB_CHARTS) + ["recommend", "table"])
def test_every_tab_renders(live, shots, tab):
    live.evaluate("id => switchTab(id)", tab)
    if tab == "table":
        live.wait_for_function(
            "() => document.querySelectorAll('#model-table-body tr').length > 10",
            timeout=60_000)
    elif tab == "recommend":
        live.wait_for_function(
            "() => (document.getElementById('recommend-cards')?.innerText || '').length > 40",
            timeout=60_000)
    else:
        for chart in TAB_CHARTS[tab]:
            _wait_drawn(live, f"chart-{chart}")
            assert _points(live, f"chart-{chart}") > 0, f"{tab}/{chart} drew nothing"
    live.wait_for_timeout(400)
    live.screenshot(path=str(shots / f"tab-{tab}.png"))


def test_provider_filter_narrows_the_overview(live):
    live.evaluate("() => switchTab('overview')")
    _wait_drawn(live, "chart-pareto")
    before = _points(live, "chart-pareto")
    live.select_option("#filter-provider", ["OpenAI"])
    live.wait_for_function(f"n => {_COUNT_JS}('chart-pareto') < n",
                           arg=before, timeout=60_000)
    after = _points(live, "chart-pareto")
    assert 0 < after < before
    live.click("#preset-all")
    live.wait_for_function(f"n => {_COUNT_JS}('chart-pareto') === n",
                           arg=before, timeout=60_000)


def test_csv_export_downloads_the_filtered_catalogue(live, shots):
    live.evaluate("() => switchTab('overview')")
    live.select_option("#filter-provider", ["OpenAI"])
    live.wait_for_timeout(1500)
    with live.expect_download(timeout=60_000) as dl:
        live.click("#btn-export")
    path = shots / (dl.value.suggested_filename or "export.csv")
    dl.value.save_as(str(path))
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert rows, "CSV export was empty"
    assert {"model", "provider"} <= set(rows[0]), list(rows[0])
    assert {r["provider"] for r in rows} == {"OpenAI"}, "export ignored the filter"
    live.click("#preset-all")


def test_video_modes_are_separate_arenas(live, shots):
    """t2v is the re-anchored 30-model pool; i2v is still the 83-model one.

    The ranked chart caps at 30 bars, so the subtitle — which says "all N" or
    "top 30 of N" — is what tells the two pools apart.
    """
    import re

    title = "id => (document.getElementById(id).layout.title || {}).text || ''"
    live.evaluate("() => switchTab('video')")
    _wait_drawn(live, "chart-video_rankings")
    live.wait_for_function(f"() => ({title})('chart-video_rankings').includes('text to video')",
                           timeout=60_000)
    t2v_title = live.evaluate(title, "chart-video_rankings")
    t2v_scatter = _points(live, "chart-video_scatter")

    live.select_option("#video-mode-filter", "i2v")
    live.wait_for_function(f"() => ({title})('chart-video_rankings').includes('image to video')",
                           timeout=60_000)
    _wait_drawn(live, "chart-video_scatter")
    i2v_title = live.evaluate(title, "chart-video_rankings")
    i2v_scatter = _points(live, "chart-video_scatter")
    live.screenshot(path=str(shots / "tab-video-i2v.png"))
    live.select_option("#video-mode-filter", "t2v")

    def shown(t):
        m = re.search(r"(?:all|of) (\d+)", t)
        assert m, t
        return int(m.group(1))

    assert 0 < shown(t2v_title) <= 30, f"text-to-video claims {t2v_title!r}"
    assert shown(i2v_title) > 30, f"image-to-video claims {i2v_title!r}"
    assert 0 < t2v_scatter < i2v_scatter


def test_no_console_errors_or_failed_requests(live):
    # favicon is the browser's own probe, not something the page asked for.
    real = [p for p in live.problems if "favicon" not in p]
    assert not real, "\n".join(real)


# ── ⟳ pulls the newest published snapshot ─────────────────────────────────────

def test_refresh_button_loads_a_newer_published_snapshot(browser, site):
    """A newer manifest version must navigate to it (fresh figures, bundles and
    Pyodide); the same version must say so rather than silently doing nothing."""
    page = _open(browser, site)
    shipped = page.evaluate("() => window.AF.version")
    assert shipped

    newer = "29991231T000000Z"

    def bump(route):
        resp = route.fetch()
        body = json.loads(resp.text())
        body["version"] = newer
        route.fulfill(response=resp, body=json.dumps(body))

    page.route("**/figures/manifest.json*", bump)
    with page.expect_navigation(timeout=30_000):
        page.click("#btn-refresh")
    page.wait_for_function("v => window.AF && window.AF.version === v",
                           arg=newer, timeout=30_000)
    assert f"v={newer}" in page.url
    page.context.close()


def test_refresh_button_when_current_reports_the_data_age(browser, site):
    page = _open(browser, site)
    page.click("#btn-refresh")
    page.wait_for_function(
        "() => (document.getElementById('toast')?.textContent || '').includes('up to date')",
        timeout=15_000)
    toast = page.inner_text("#toast")
    badge = _badge(page)["text"]
    page.context.close()
    # Same relative time as the badge: both describe the data, not the build.
    assert toast.split("updated ", 1)[1].strip() in badge, (toast, badge)


# ── Freshness badge: healthy, honestly failing, and as shipped ───────────────

def _now_iso(hours_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(
        timespec="seconds")


def test_badge_is_calm_when_every_dataset_refreshed(browser, site, shots):
    def healthy(m):
        stamp = _now_iso(0.2)
        for e in m["datasets"].values():
            e.update(ok=True, fetched_at=stamp, checked_at=stamp)
            e.pop("error", None)
        m["stale_datasets"], m["data_fetched_iso"] = [], stamp
        return m

    page = _open(browser, site, healthy)
    b = _badge(page)
    page.screenshot(path=str(shots / "badge-healthy.png"),
                    clip={"x": 900, "y": 0, "width": 700, "height": 120})
    page.context.close()
    assert "⚠" not in b["text"], b
    assert b["inline"] == "" and b["color"] != AMBER, b
    assert "Stale or failing" not in b["title"] and "FAILED" not in b["title"], b


def test_badge_warns_and_says_why_when_a_scrape_failed(browser, site, shots):
    reason = "textToVideo: 86 -> 30 scored models — looks truncated, not re-scaled"

    def failing(m):
        v = m["datasets"]["video"]
        v.update(ok=False, fetched_at=_now_iso(30), error=reason)
        m["stale_datasets"] = ["video"]
        return m

    page = _open(browser, site, failing)
    b = _badge(page)
    page.screenshot(path=str(shots / "badge-failing.png"),
                    clip={"x": 900, "y": 0, "width": 700, "height": 120})
    page.context.close()
    assert "⚠" in b["text"] and b["color"] == AMBER, b
    assert "Stale or failing: video" in b["title"], b["title"]
    assert "Video arena:" in b["title"] and "last scrape FAILED: " + reason in b["title"]


def test_badge_matches_the_shipped_manifest(browser, site, shots):
    """No patching: the badge must agree with what was actually published."""
    m = json.loads((DOCS / "figures" / "manifest.json").read_text())
    limit = float(m.get("stale_after_hours") or 12)
    now = datetime.now(timezone.utc)
    bad = set(m.get("stale_datasets") or [])
    for name, e in (m.get("datasets") or {}).items():
        when = e.get("fetched_at")
        if e.get("ok") is not True or not when or (
                now - datetime.fromisoformat(when)).total_seconds() >= limit * 3600:
            bad.add(name)

    page = _open(browser, site)
    b = _badge(page)
    page.screenshot(path=str(shots / "badge-shipped.png"),
                    clip={"x": 900, "y": 0, "width": 700, "height": 120})
    page.context.close()
    assert ("⚠" in b["text"]) == bool(bad), (b, sorted(bad))
    if os.environ.get("AIF_SMOKE_REQUIRE_FRESH") == "1":
        assert not bad, f"the shipped site would warn about: {sorted(bad)}"
