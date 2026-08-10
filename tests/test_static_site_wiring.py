import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "docs" / "app.js").read_text()
HTML = (ROOT / "docs" / "index.html").read_text()
WORKER = (ROOT / "docs" / "pyworker.js").read_text()

def test_manifest_fetched_no_store_and_version_captured():
    assert 'fetch("figures/manifest.json", { cache: "no-store" })' in APP
    assert "window.AF.version = m.version" in APP
    assert "window.AF.generatedIso = m.generated_iso" in APP

def test_data_assets_are_version_busted():
    assert "figures/${figId}.json?v=" in APP
    # The bundle is fetched by the Pyodide worker, and the app hands it the
    # manifest version so a refresh still busts the cache.
    assert "pybundle.zip?v=" in WORKER
    assert 'worker.postMessage({ type: "boot", version: window.AF.version' in APP

def test_pyodide_runs_off_the_main_thread():
    """Booting Pyodide inline blocked the UI for ~3.8s of a 4.7s startup, so
    hovering a bubble during load felt like the mouse had frozen."""
    assert 'new Worker("pyworker.js")' in APP
    # no inline pyodide bootstrap left on the main thread
    assert "loadPyodide(" not in APP
    assert "unpackArchive" not in APP
    assert "loadPyodide(" in WORKER and "unpackArchive" in WORKER

def test_js_provider_colors_mirror_python():
    """docs/app.js keeps its own copy of the palette for the Table tab's
    provider dot. Drift there means a model is one color in the chart and
    another in the table."""
    from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR

    block = re.search(r"const PROVIDER_COLORS = \{(.*?)\n\};", APP, re.S)
    assert block, "PROVIDER_COLORS block not found in docs/app.js"
    js = dict(re.findall(r'"([^"]+)":\s*"(#[0-9a-fA-F]{6})"', block.group(1)))
    assert js == PROVIDER_COLORS, (
        f"palette drift — only in JS: {set(js) - set(PROVIDER_COLORS)}, "
        f"only in Python: {set(PROVIDER_COLORS) - set(js)}, "
        f"different hex: {{k for k in set(js) & set(PROVIDER_COLORS) if js[k] != PROVIDER_COLORS[k]}}"
    )
    assert f'const DEFAULT_PROVIDER_COLOR = "{DEFAULT_COLOR}";' in APP


def test_worker_rpc_is_wired_both_ways():
    assert "window.AF.callPy = " in APP and "window.AF.callPyRaw = " in APP
    assert 'worker.postMessage({ type: "call"' in APP
    assert '"result"' in WORKER and '"ready"' in WORKER and '"bootError"' in WORKER

def test_freshness_badge_present():
    assert 'id="data-freshness"' in HTML
    assert "function relativeTime(" in APP
    assert "function renderFreshness(" in APP
    assert "setInterval(renderFreshness, 60000)" in APP

def test_refresh_button_rewired():
    assert "async function doRefresh(" in APP
    assert '{ cache: "no-store" }' in APP            # manifest re-check
    assert 'document.getElementById("btn-refresh").onclick = doRefresh' in APP
    # old bare reload wiring is gone
    assert 'getElementById("btn-refresh").onclick = () => location.reload()' not in APP
    assert 'id="toast"' in HTML
    assert "function toast(" in APP

def test_table_rows_escape_scraped_text():
    """docs/app.js:539 spliced ${r.model} straight into a template literal that
    line 549 assigns to tbody.innerHTML — a second XSS sink independent of the
    Python-rendered HTML."""
    assert "${escapeHtml(r.model)}" in APP
    assert "${escapeHtml(r.provider)}" in APP
    # the raw interpolations are gone
    assert "${r.model}</td>" not in APP
    assert "${r.provider}</td>" not in APP


def test_compare_option_list_is_built_with_dom_api():
    """The compare <select> was rebuilt from an innerHTML template, so a scraped
    model label was parsed as markup. createElement keeps option.value byte-exact
    for the selection match on the next line."""
    assert '`<option value="${o.value}">${o.label}</option>`' not in APP
    assert "sel.replaceChildren(" in APP
    assert "opt.textContent = o.label" in APP


def test_escape_html_helper_actually_escapes():
    """Run the real helper in node rather than trusting a grep."""
    import json, shutil, subprocess
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("node not available")

    block = re.search(r"const ESCAPE_MAP = .*?\n\}", APP, re.S)
    assert block, "escapeHtml helper not found in docs/app.js"
    payload = '<img src=x onerror=alert(1)>&"\''
    script = block.group(0) + "\nprocess.stdout.write(escapeHtml(" + json.dumps(payload) + "));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout
    assert out == "&lt;img src=x onerror=alert(1)&gt;&amp;&quot;&#39;"
    # ampersand first, so nothing is double-encoded
    assert subprocess.run(
        [node, "-e", block.group(0) + '\nprocess.stdout.write(escapeHtml("a & b"));'],
        capture_output=True, text=True, check=True,
    ).stdout == "a &amp; b"


def test_budget_min_intelligence_is_wired():
    """The Budget tab's own intelligence floor: control, readout, and the call
    that carries it through to Python."""
    assert 'id="budget-min-intelligence"' in HTML
    assert 'id="budget-min-intelligence-value"' in HTML
    assert 'id="budget-answer"' in HTML
    # the floor is passed as the 5th argument of update_cost_calc
    assert 'callPy("update_cost_calc", tok, p, q, s, floor)' in APP
    assert "function refreshBudget(" in APP


def test_budget_answer_is_built_with_dom_apis():
    """The answer card shows a scraped model name, so it must not be assembled
    as an HTML string (see the XSS fixes in static_api.py / renderTableRows)."""
    block = APP[APP.index("function renderBudgetAnswer("):APP.index("async function refreshBudget(")]
    assert "innerHTML" not in block
    assert "name.textContent = best.model" in block
