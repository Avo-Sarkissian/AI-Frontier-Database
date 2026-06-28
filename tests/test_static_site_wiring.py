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
