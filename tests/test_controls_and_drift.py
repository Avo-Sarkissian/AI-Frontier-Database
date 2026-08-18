"""Controls must do what they say, and both renderings must say the same thing.

Themes 4 and 5 of audit/2026-08-12. Two patterns:

* A control that silently does something other than what it is labelled — the
  Compare selection being discarded rather than pruned, a quality preset also
  clearing the provider filter, a filter bar live over content it does not
  touch, ↓CSV exporting a different dataset from the one on screen.
* app.py and docs/app.js drifting, always in the same direction: the fix lands
  on the Dash side, which nobody deploys, and the public site keeps the bug.
"""
import json
import re
from pathlib import Path

import pytest

from captions import CAPTIONS
from data.ingest import get_models
from static_helpers import (
    apply_filters, cap_compare_selection, compute_diverse5, quality_options,
    export_frame_for_tab, TABS_WITHOUT_GLOBAL_FILTERS, COMPARE_MAX,
)

ROOT = Path(__file__).resolve().parent.parent
DF = get_models()
APP_JS = (ROOT / "docs" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()
INDEX = (ROOT / "docs" / "index.html").read_text()


def _manifest():
    p = ROOT / "docs" / "figures" / "manifest.json"
    if not p.exists():
        pytest.skip("no built manifest")
    return json.loads(p.read_text())


# ── 4.4 — a filter must prune the Compare selection, not replace it ──────────

def test_a_selection_that_still_passes_the_filter_survives_it():
    """A curated three-model comparison vanished on one click of "Top 25%"
    while every one of the three still cleared the new floor."""
    strong = DF.nlargest(3, "quality")["model"].tolist()
    floor = float(DF["quality"].quantile(0.75))
    filtered = apply_filters(DF, None, floor, "")
    kept = cap_compare_selection(strong, filtered, "filter-quality")
    assert set(kept) == set(strong), (
        f"models that still pass a >= {floor} floor were discarded: "
        f"{set(strong) - set(kept)}"
    )


def test_a_selection_excluded_by_the_filter_is_dropped_not_kept():
    """Pruning, not ignoring — a model the filter excludes must go."""
    weakest = DF.nsmallest(1, "quality")["model"].tolist()
    filtered = apply_filters(DF, None, float(DF["quality"].quantile(0.9)), "")
    kept = cap_compare_selection(weakest, filtered, "filter-quality")
    assert weakest[0] not in kept


def test_a_tab_switch_does_not_touch_the_selection():
    """The static site routed every tab change through the filter path, so
    going to Table and back replaced the user's picks with the defaults. Dash
    has no tabs input on this callback and never did it."""
    picked = DF.nlargest(3, "quality")["model"].tolist()
    assert cap_compare_selection(picked, DF, "tab-switch") == picked


def test_an_empty_selection_still_falls_back_to_the_defaults():
    assert cap_compare_selection([], DF, "tab-switch") == compute_diverse5(DF)
    assert cap_compare_selection(None, DF, None) == compute_diverse5(DF)


def test_the_static_site_passes_a_tab_switch_trigger():
    assert 'refreshCompare("filter-provider")' not in APP_JS, (
        "a tab switch still claims to be a provider-filter change"
    )
    assert "tab-switch" in APP_JS


# ── 4.8 — the cap evicts the oldest pick, not the newest ─────────────────────

def test_the_sixth_pick_is_kept_and_the_oldest_is_evicted():
    """`[:5]` threw away the model the user had just clicked, which reads as
    the click doing nothing at all."""
    six = DF.nlargest(6, "quality")["model"].tolist()
    kept = cap_compare_selection(six, DF, None)
    assert len(kept) == COMPARE_MAX
    assert six[-1] in kept, "the newest pick was discarded"
    assert six[0] not in kept, "the oldest pick was not evicted"


def test_the_static_site_trims_an_over_selection_back_into_the_control():
    """Shift-selecting 9 left 8 highlighted while 5 were charted."""
    assert "selected.slice(-COMPARE_MAX)" in APP_JS


def test_the_add_to_compare_button_is_not_a_no_op_when_full():
    """Both renderings default to exactly 5 models, so the detail panel's only
    call to action did nothing on a fresh page load — and switched tabs
    anyway."""
    assert "chosen.length < 5" not in APP_JS, (
        "the button still refuses to act when the selection is already full"
    )
    assert "COMPARE_MAX - 1" in APP_JS


# ── 4.7 — a quality preset touches quality ───────────────────────────────────

def test_quality_presets_leave_the_provider_selection_alone():
    block = re.search(r"def apply_preset\(.*?\n\n\n", APP_PY, re.S)
    assert block, "could not find apply_preset"
    body = block.group(0)
    assert 'return _P75, no_update, no_update' in body
    assert 'return _P90, no_update, no_update' in body


def test_the_static_presets_leave_the_provider_selection_alone():
    assert "setPreset(window.AF.manifest.p75, null, false)" in APP_JS
    assert "setPreset(window.AF.manifest.p90, null, false)" in APP_JS


def test_the_button_that_clears_everything_says_so():
    """"All" cleared SEARCH and PROVIDER too — three buttons, three scopes,
    one label that named none of it."""
    assert "Reset filters" in APP_PY and "Reset filters" in INDEX


# ── 4.9 — the filter bar and ↓CSV must match what is on screen ───────────────

@pytest.mark.parametrize("tab", TABS_WITHOUT_GLOBAL_FILTERS)
def test_the_global_filter_bar_is_hidden_where_it_does_nothing(tab):
    assert "global-filters" in APP_PY, "the Dash filter bar has no id to hide by"
    assert "TABS_WITHOUT_GLOBAL_FILTERS" in APP_JS
    assert f'"{tab}"' in APP_JS


@pytest.mark.parametrize("tab,expect_col", [
    ("image", "elo"),
    ("video", "elo_t2v"),
    ("local", "params_b"),
])
def test_csv_export_returns_the_dataset_on_screen(tab, expect_col):
    """From Image Gen, export_csv(['Anthropic'], 50, '') returned the LLM
    header and seven text models."""
    frame, name = export_frame_for_tab(tab, DF, ["Anthropic"], 50, "")
    assert expect_col in frame.columns, f"{tab} exported the wrong dataset: {list(frame.columns)[:5]}"
    assert tab in name, f"{tab} export is named {name!r}"


def test_csv_export_still_honours_the_filters_where_the_bar_is_live():
    frame, name = export_frame_for_tab("overview", DF, ["Anthropic"], 0, "")
    assert set(frame["provider"]) == {"Anthropic"}
    assert name == "ai_frontier_export.csv"


# ── 5.1 — an unknown ?tab= must not blank the page ───────────────────────────

def test_an_unknown_tab_id_falls_back_instead_of_hiding_every_panel():
    """`insights`, `performance` and `embeddings` were all live tab values this
    app once emitted into share URLs. Dash has guarded this since 82effa0; the
    static site never got the same fix."""
    assert "VALID_TABS" in APP_JS
    assert "if (!VALID_TABS.includes(id)) id = VALID_TABS[0];" in APP_JS


def test_dash_still_validates_the_tab_too():
    assert "_VALID_TABS" in APP_PY


# ── 5.3 — one quality value, one representation ──────────────────────────────

def test_the_quality_control_can_hold_the_value_its_presets_set():
    """42.1 into a round-ladder dropdown left the Dash control visually blank
    and made the static site snap 52.7 down to 50 — so "Top 10%" returned
    15.5% of the catalogue under a label that is a precise numeric claim."""
    p75 = round(float(DF["quality"].quantile(0.75)), 1)
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    values = [o["value"] for o in quality_options(p75, p90)]
    assert p75 in values and p90 in values
    assert values == sorted(values)


def test_the_preset_percentiles_are_labelled_not_bare_numbers():
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    labels = {o["value"]: o["label"] for o in quality_options(None, p90)}
    assert "top 10%" in labels[p90]


def test_a_fractional_q_param_is_not_silently_dropped():
    """`int("42.1")` raised, the error was swallowed, and the filter became 0 —
    a shared "Top 25%" link opened as no filter at all."""
    import app as dash_app

    p75 = round(float(DF["quality"].quantile(0.75)), 1)
    _tab, _providers, quality = dash_app.init_from_url(f"?q={p75}")
    assert quality == pytest.approx(p75), (
        f"?q={p75} parsed as {quality}; the filter was silently dropped"
    )


def test_the_static_site_inserts_an_option_rather_than_snapping():
    assert "ensureQualityOption" in APP_JS
    assert "snapped" not in APP_JS, "the nearest-option snap is still there"


def test_the_manifest_ships_the_quality_options():
    opts = _manifest().get("quality_options")
    assert opts, "manifest has no quality_options — re-run build_static.py"
    p90 = round(float(DF["quality"].quantile(0.90)), 1)
    assert p90 in [o["value"] for o in opts]


# ── 5.4 — the deployed site must explain itself too ──────────────────────────

def test_every_caption_has_one_source():
    """app.py rendered 14 captions; docs/app.js rendered 1, so nine tabs of the
    site that actually deploys explained nothing."""
    assert len(CAPTIONS) >= 12
    assert "from captions import CAPTIONS" in APP_PY
    literal_descs = re.findall(r'_desc\(\s*\n?\s*"', APP_PY)
    assert not literal_descs, (
        f"{len(literal_descs)} captions are still inline string literals in app.py"
    )


def test_the_manifest_ships_the_captions():
    caps = _manifest().get("captions")
    assert caps, "manifest has no captions — re-run build_static.py"
    assert caps == dict(CAPTIONS), "manifest captions have drifted from captions.py"


def test_the_static_site_renders_a_caption_on_every_tab():
    assert "renderCaptions" in APP_JS
    block = re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};", APP_JS, re.S)
    assert block, "TAB_CAPTIONS not found"
    covered = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
    tabs = set(re.findall(r'\{ id: "(\w+)"', APP_JS))
    assert tabs <= covered, f"tabs with no caption: {sorted(tabs - covered)}"


def test_every_caption_key_the_site_asks_for_exists():
    block = re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};", APP_JS, re.S)
    wanted = set(re.findall(r'"(\w+)"', block.group(1)))
    missing = wanted - set(CAPTIONS)
    assert not missing, f"docs/app.js asks for captions that do not exist: {missing}"


# ── 5.5 — the six smaller drift rows ─────────────────────────────────────────

def test_the_refresh_signal_is_content_addressed_not_a_counter():
    """_cache_mtime is a module global while data-version is per-session, and
    _reload_if_stale clears the "changed" flag as a side effect — so whichever
    session ticked first consumed the signal and the others got a fresh stat bar
    above stale charts."""
    assert "new_version = _cache_mtime" in APP_PY, (
        "the refresh signal is still an incrementing per-session counter"
    )


def test_the_table_headers_do_not_invite_a_click_that_does_nothing():
    block = re.search(r"sort_action=\"none\".*?style_cell=", APP_PY, re.S)
    assert block, "could not find the table component"
    assert '"cursor": "pointer"' not in block.group(0), (
        "inert headers still show a pointer cursor"
    )


def test_filter_state_is_read_before_the_pyodide_guard():
    """window.AF.state is written only by readGlobalFilters, which sat one line
    BELOW the `if (!pyReady) return;` guard — so pre-boot the state said no
    filters while the DOM said Anthropic / >= 40, and Share copied a filter-less
    link and rewrote the address bar with it."""
    body = APP_JS.split("async function rerenderActiveFilterCharts")[1][:600]
    read_at = body.index("readGlobalFilters()")
    guard_at = body.index("if (!window.AF.pyReady) return;")
    assert read_at < guard_at, "the filter state is still populated after the guard"


def test_the_export_button_says_when_it_cannot_act_yet():
    """↓CSV returned silently pre-boot: no download, no message, no console
    line — and Pyodide is CDN-loaded, so a cold cache widens that window."""
    assert "setExportPending" in APP_JS


def test_the_detail_panel_closes_on_a_tab_switch():
    """It was cleared only by its own close button, so it stayed open — still
    armed to an LLM with a live "Add to Compare" — over the Video Gen tab."""
    body = APP_JS.split("function switchTab")[1][:800]
    assert "detail-panel" in body, "switchTab does not close the detail panel"


def test_the_speed_column_does_not_use_the_visitors_locale():
    """A bare toLocaleString renders 1560 as "1.560" in de-DE — a 1000x
    ambiguity in a column next to $0.1580."""
    assert ".toLocaleString()" not in APP_JS
    assert 'toLocaleString("en-US")' in APP_JS


def test_the_pre_boot_speed_view_uses_the_prebuilt_quadrant_figure():
    """docs/figures/quadrant.json is built (29 KB) and was never fetched, so
    switching to Speed before boot left the Price chart under a Speed caption —
    an exactly inverted reading of the same picture."""
    assert '"quadrant"' in APP_JS, "the pre-built quadrant figure is still unused"


# ── 9.5 / 9.6 / 8.2 — the numbers a caption implies ──────────────────────────

def test_the_overview_charts_state_how_many_rows_they_collapse():
    """Prose alone left the cross-tab contradiction unexplained: the header tile
    says 155, the treemap says OpenAI ships 25, and Overview draws 8 of them."""
    import json as _json
    from components.charts.pareto import build_pareto_scatter
    from components.charts.quadrant import build_quadrant

    for build in (lambda: build_pareto_scatter(DF, full_df=DF),
                  lambda: build_quadrant(DF, full_df=DF)):
        title = _json.loads(build().to_json())["layout"]["title"]["text"]
        assert re.search(r"\d+ of \d+ rows", title), (
            f"subtitle does not quantify the family collapse: {title!r}"
        )


def test_the_budget_chart_states_the_token_mix_it_assumes():
    """The whole cost model is `tokens * price`, and price is the 3:1
    output-weighted blend — so every figure assumes 75% generated tokens.
    Claude Opus 5 at 100M charts $2000 where an all-input workload pays $500."""
    import json as _json
    from components.charts.cost_calc import build_cost_calc

    title = _json.loads(build_cost_calc(DF).to_json())["layout"]["title"]["text"]
    assert "75% output" in title, f"the Budget chart hides its assumption: {title!r}"


def test_the_run_local_caption_says_what_the_vram_figure_covers():
    caption = CAPTIONS["local"].lower()
    assert "kv cache" in caption and "weights" in caption


def test_image_providers_are_not_split_across_spellings():
    """"Bytedance" (6) and "ByteDance Seed" (3) counted as two providers and
    fragmented the leaderboard."""
    from data.image_models import get_image_df

    names = set(get_image_df()["provider"].dropna())
    normalised = {}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        normalised.setdefault(key, []).append(n)
    dupes = {k: v for k, v in normalised.items() if len(v) > 1}
    assert not dupes, f"one provider under several spellings: {dupes}"


def test_every_quant_select_is_filled_through_the_one_helper():
    """A {label, value} option assigned whole becomes "[object Object]".

    quant_levels became quant_options so Q3/Q2 could be marked lossy. docs/app.js
    had TWO copies of the population loop; only local-quant was updated, so
    recommend-quant shipped <option value="[object Object]">. Agent Stack sent
    that string as the quantisation, calc_vram_gb raised
    KeyError('[object Object]') inside Pyodide, and every workflow with a local
    tier silently kept showing the previous API cards — while the radio and the
    hardware row both moved, so the control looked like it had worked.

    Nothing in the suite touched it: the Python was correct at every layer and
    the defect lived entirely in the browser. This greps the shipped JS instead.
    """
    js = (ROOT / "docs" / "app.js").read_text()

    # Every quant <select> in the shell must be filled by the shared helper.
    shell = (ROOT / "docs" / "index.html").read_text()
    quant_selects = re.findall(r'<select id="([a-z-]*quant[a-z-]*)"', shell)
    assert quant_selects, "no quant selects found in docs/index.html"
    for sel in quant_selects:
        assert f'fillQuantSelect("{sel}"' in js, (
            f"#{sel} is not populated by fillQuantSelect — a second copy of the "
            f"loop is exactly how recommend-quant shipped [object Object]"
        )

    # And nothing may assign a whole option object to .value.
    assert not re.search(r'opt\.value\s*=\s*q\s*;', js), (
        "an option loop still assigns the raw item to .value; with {label, value} "
        "options that serialises to the string '[object Object]'"
    )


def test_the_quant_values_the_browser_sends_are_real_quant_levels():
    """The end of the contract the test above guards: whatever the select can
    emit has to be a key calc_vram_gb accepts."""
    from data.local_models import QUANT_BYTES, quant_options

    for opt in quant_options():
        assert opt["value"] in QUANT_BYTES, (
            f"{opt['value']!r} is offered by the control but is not a "
            f"quantisation calc_vram_gb can price"
        )
