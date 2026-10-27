"""Economics › Labs: the lab ledger, Epoch AI slices and the four figures
(data/lab_economics.py, components/charts/lab_economics.py)."""
from __future__ import annotations

import inspect
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import build_static
import static_api as api
from captions import CAPTIONS
from components.charts import lab_economics as C
from data import lab_economics as L
from data.econ_common import STATUS_VALUES
from static_helpers import econ_export, export_frame_for_tab

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()


@pytest.fixture(autouse=True)
def _fresh_caches():
    L.reset_cache()
    yield
    L.reset_cache()


def _ledger_csv() -> pd.DataFrame:
    return pd.read_csv(ROOT / L.LEDGER_FILE, dtype=str, keep_default_na=False)


def _data_traces(fig) -> list:
    """Figure traces that carry data (legend-only key traces have x=[None])."""
    return [t for t in fig.data if t.x is not None and any(v is not None for v in t.x)]


# ── The ledger file ──────────────────────────────────────────────────────────

def test_ledger_schema_and_provenance():
    df = _ledger_csv()
    assert tuple(df.columns) == L.LEDGER_COLUMNS
    assert df["row_id"].is_unique
    for _, r in df.iterrows():
        where = r["row_id"]
        assert r["badge"] in L.BADGES, where
        assert r["filing_status"] in L.FILING_STATUSES, where
        assert r["ledger_state"] in L.LEDGER_STATES, where
        assert r["view"] in L.LEDGER_VIEWS, where
        assert r["status"] in STATUS_VALUES, where
        assert r["source_url"].startswith("https://"), where
        for col in ("source_date", "retrieved", "as_of"):
            assert date.fromisoformat(r[col]) <= date.today(), (where, col)
        if r["ledger_state"] == "excluded":
            assert len(r["note"].strip()) > 30, where
        if r["view"] == "commitment" and r["ledger_state"] == "ship":
            assert r["denominator_basis"] in L.DENOMINATOR_BASES, where
        if r["view"] == "token":
            assert r["per"] in L.TOKENS_PER_DAY_FACTOR and r["scope"] in L.SCOPE_LABELS, where


def test_the_ledger_ships_to_the_browser():
    assert L.LEDGER_FILE in build_static.DATA_CSVS
    assert {L.DERIVED_REVENUE, L.DERIVED_TOKENS} <= set(build_static.DERIVED_DATA)
    src = inspect.getsource(build_static.build_pybundle)
    for mod in ("data/lab_economics.py", "components/charts/lab_economics.py"):
        assert f'"{mod}"' in src
    # Pyodide has no requests; nothing in the Labs modules may fetch.
    for mod in ("data/lab_economics.py", "components/charts/lab_economics.py"):
        assert not re.search(r"^\s*(import|from)\s+requests", (ROOT / mod).read_text(), re.M)


def test_leaked_and_misdefined_figures_never_ship():
    df = L.load_ledger(include_excluded=True)
    shipped = df[df["ledger_state"] == "ship"]
    assert not shipped["value"].isin([518e9, 665e9]).any()
    excluded = df[df["ledger_state"] == "excluded"]
    assert set(excluded["value"]) == {518e9, 665e9}
    figs = C.build_labs_figures()
    blob = json.dumps([json.loads(f.to_json())["data"] for f in figs.values()])
    for bad in ("518000000000", "665000000000", "5.18e+11", "6.65e+11", "$518B", "$665B"):
        assert bad not in blob
    com = figs["econ_labs_commitments"]
    assert all("OpenAI" not in str(t.y) for t in _data_traces(com))
    # ... but the ledger shows them, greyed, with the reason
    h = C.labs_ledger_html()
    assert h.count('class="lab-ledger__excluded"') == 2
    assert "public S-1" in h and "not contracted commitments" in h


def test_filed_margins_recompute_from_their_inputs():
    m = L.load_ledger()
    m = m[(m["view"] == "margin") & m["numerator"].notna() & m["denominator"].notna()]
    assert len(m) >= 16
    # As filed, Zhipu's FY24 cloud margin reads 3.3%; the rounded RMB-thousand
    # inputs give 3.37%, so its tolerance is one rounding step wider.
    wider = {"zhipu_cloud_gm_fy24": 0.1}
    for _, r in m.iterrows():
        got = r["numerator"] / r["denominator"] * 100
        assert abs(got - r["value"]) <= wider.get(r["row_id"], 0.06), (r["row_id"], got)
    by_id = m.set_index("row_id")["value"]
    assert by_id["zhipu_gm_h1_26"] == 26.4 and by_id["zhipu_cloud_gm_h1_26"] == 24.6
    assert by_id["minimax_gm_h1_26"] == 17.9 and by_id["minimax_gm_fy25"] == 25.4
    assert by_id["zhipu_gm_fy25"] == 41.0


def test_anthropic_derived_margin_matches_the_s1():
    m = L.load_ledger().set_index("row_id")
    assert abs(m.loc["anthropic_rev_less_compute_fy25", "value"] - (-59.8)) <= 0.05
    assert abs(m.loc["anthropic_rev_less_compute_h1_26", "value"] - 45.1) <= 0.05
    rows = m[m.index.str.startswith("anthropic_rev_less_compute")]
    assert (rows["status"] == "derived").all() and (rows["derived"] == "1").all()
    assert all("not a gross margin" in d for d in rows["definition"])


def test_interim_hkex_rows_are_marked_unaudited():
    m = L.load_ledger()
    hk = m[m["source_url"].str.contains("hkexnews")]
    assert len(hk) == 12
    for _, r in hk.iterrows():
        want = "unaudited" if r["period_label"].startswith("H1") else "audited"
        assert r["filing_status"] == want, r["row_id"]


def test_epoch_badge_mapping():
    assert L.epoch_badge("Company disclosure", "Annualized run rate") == "disclosed"
    assert L.epoch_badge("Company disclosure,Media report",
                         "Annual recurring revenue (ARR)") == "disclosed"
    assert L.epoch_badge("Media report", "Annualized run rate") == "reported"
    assert L.epoch_badge("Company disclosure", "Period interpolation") == "estimated"
    assert L.epoch_badge("Other", "Annualized run rate") == "estimated"
    assert L.epoch_badge(float("nan"), None) == "estimated"


def test_token_normalisation_matches_epochs_convention():
    assert abs(L.tokens_per_day(9.7e12, "month") / 3.23e11 - 1) < 0.005
    assert abs(L.tokens_per_day(6e9, "minute") / 8.6e12 - 1) < 0.01
    assert L.tokens_per_day(3.2e15, "month") == 3.2e15 / 30
    assert L.tokens_per_day(128.4e12, "week") == 128.4e12 / 7
    assert L.tokens_per_day(1, "fortnight") is None


def test_token_scope_classification():
    assert L.token_scope("Google (all AI products)") == "all"
    assert L.token_scope("Full company") == "all"
    assert L.token_scope("Chat and API for V3 and R1") == "all"
    assert L.token_scope("API") == "api" and L.token_scope("Gemini API") == "api"
    assert L.token_scope("GPT-5-Codex") == "product"


def test_curated_token_rows_do_not_duplicate_epoch():
    t = L.token_rows()
    assert {"epoch", "curated"} <= set(t["origin"])
    for pal, g in t[t["palette_name"] != ""].groupby("palette_name"):
        d = pd.to_datetime(g["date"]).sort_values()
        cur = g[g["origin"] == "curated"]
        for _, r in cur.iterrows():
            others = g[(g["origin"] == "epoch")]
            gaps = (pd.to_datetime(others["date"]) - pd.Timestamp(r["date"])).abs().dt.days
            assert not (gaps <= L.TOKEN_DEDUP_DAYS).any(), pal


def test_a_curated_token_row_near_an_epoch_row_is_dropped(monkeypatch):
    led = L.load_ledger()
    g = led[led["row_id"] == "google_tokens_2026_05"].copy()
    ep = L.load_epoch_tokens()
    near = ep[ep["company"] == "Google"].iloc[0]
    g["as_of"] = near["date"]
    monkeypatch.setattr(L, "load_ledger", lambda include_excluded=False: g)
    t = L.token_rows()
    assert "curated" not in set(t[t["entity"] == "Google"]["origin"])


def test_commitment_pairing_rule(monkeypatch):
    c = L.commitment_rows().set_index("row_id")
    a = c.loc["anthropic_noncancellable_2025"]
    assert a["den_date"] == "2025-12-31" and a["den_usd"] == 9e9
    assert abs(a["multiple"] - 6.06) < 0.01
    b = c.loc["anthropic_fixed_min_2026_06"]
    assert b["den_date"] == "2026-05-15" and b["den_usd"] == 47e9
    assert abs(b["multiple"] - 5.65) < 0.01
    cw = c.loc["coreweave_rpo_2026_06"]
    assert cw["den_usd"] == 2575e6 * 4 and abs(cw["multiple"] - 10.07) < 0.01
    assert c["charted"].all()
    # a commitment with no run-rate within 92 days is flagged, not charted
    led = L.load_ledger()
    extra = led[led["row_id"] == "anthropic_noncancellable_2025"].copy()
    extra["row_id"], extra["as_of"] = "synthetic", "2023-06-30"
    both = pd.concat([led, extra], ignore_index=True)
    monkeypatch.setattr(L, "load_ledger", lambda include_excluded=False: both)
    c2 = L.commitment_rows().set_index("row_id")
    assert not c2.loc["synthetic", "charted"]
    assert "92 days" in c2.loc["synthetic", "flag"]
    fig = C.build_labs_commitments(c2.reset_index())
    assert len(_data_traces(fig)) == 3


# ── The Epoch slices ─────────────────────────────────────────────────────────

def _raw_revenue(**over) -> pd.DataFrame:
    row = {"Id": "X 1B", "Company": "OpenAI", "Date": "2026-01-31",
           "Annualized revenue (USD)": "1000000000", "Annualized revenue type":
           "Annualized run rate", "Scope": "Full company", "Confidence": "Likely",
           "Source Type": "Media report", "Source 1": "https://example.com/a, https://b.org"}
    row.update(over)
    return pd.DataFrame([row])


def test_slim_revenue_matches_columns_case_insensitively_and_keeps_https_only():
    out = L.slim_revenue(_raw_revenue(), today=date(2026, 10, 27))
    assert tuple(out.columns) == L.REVENUE_COLUMNS
    r = out.iloc[0]
    assert r["badge"] == "reported" and r["palette_name"] == "OpenAI"
    assert r["source_url"] == "https://example.com/a"
    assert L.slim_revenue(_raw_revenue(**{"Source 1": "http://x.com"}))["source_url"][0] == ""


def test_out_of_range_units_are_dropped_with_a_warning_never_raised(capsys):
    raw = pd.concat([_raw_revenue(), _raw_revenue(Id="drift", **{
        "Annualized revenue (USD)": "5e13"}), _raw_revenue(Id="future", Date="2031-01-01"),
        _raw_revenue(Id="old", Date="2010-01-01")], ignore_index=True)
    out = L.slim_revenue(raw, today=date(2026, 10, 27))
    assert list(out["epoch_id"]) == ["X 1B"]
    err = capsys.readouterr().out
    assert "drift" in err and "future" in err and "old" in err
    with pytest.raises(ValueError):
        L.validate_units(L.slim_revenue(_raw_revenue()).assign(annualized_usd=5e13), "revenue")


def test_a_renamed_required_column_is_refused_and_the_build_ships_empty(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        L.slim_revenue(_raw_revenue().drop(columns=["Annualized revenue (USD)"]))
    bad = tmp_path / "rev.csv"
    _raw_revenue().drop(columns=["Scope"]).to_csv(bad, index=False)
    monkeypatch.setattr(L, "EPOCH_REVENUE", str(bad))
    assert build_static.derived_text("x", lambda: L.revenue_csv(bad)) == ""


def test_the_committed_epoch_mirrors_give_in_range_slices():
    rev = pd.read_csv(pd.io.common.StringIO(L.revenue_csv()))
    tok = pd.read_csv(pd.io.common.StringIO(L.tokens_csv()))
    assert len(rev) >= 40 and len(tok) >= 8
    L.validate_units(rev, "revenue")
    L.validate_units(tok, "tokens")
    for rel in (L.DERIVED_REVENUE, L.DERIVED_TOKENS):
        assert len(build_static.DERIVED_DATA[rel]().encode()) < build_static.MAX_DERIVED_BYTES


# ── Figures ──────────────────────────────────────────────────────────────────

def test_runrate_figure():
    rr = L.runrate_frame()
    assert set(rr["scope"].str.lower()) == {"full company"}
    fig = C.build_labs_runrate()
    traces = _data_traces(fig)
    assert {t.meta["company"] for t in traces} == set(rr["company"])
    assert len(traces) == rr["company"].nunique()
    assert fig.layout.yaxis.type == "log"
    for t in traces:
        badges = [cd[7] for cd in t.customdata]
        assert list(t.marker.symbol) == [C.badge_symbol(b) for b in badges]
    notes = " ".join(a.text for a in fig.layout.annotations)
    assert "×13" in notes and "gross" in notes
    keys = [t.name for t in fig.data if t.legendgroup == "badge"]
    assert len(keys) == 3


def test_badge_symbols_follow_the_economics_convention():
    assert C.badge_symbol("disclosed") == "circle"
    assert C.badge_symbol("reported") == "circle-open"
    assert C.badge_symbol("estimated") == "diamond-open"
    assert C.badge_symbol("estimated", "square") == "square-open-dot"


def test_margins_figure_never_mixes_definitions():
    rows = L.margin_rows()
    fig = C.build_labs_margins()
    traces = _data_traces(fig)
    assert len(traces) == rows["label"].nunique()
    for t in traces:
        sub = rows[rows["label"] == t.meta["label"]]
        assert set(sub["definition"]) == {t.meta["definition"]}
        assert set(sub["entity"]) == {t.meta["entity"]}
        assert set(t.y) == {t.meta["label"]}
    ax = fig.layout.xaxis.range
    assert ax[0] <= C.MARGIN_RANGE[0] and ax[1] >= C.MARGIN_RANGE[1]
    off = [a.text for a in fig.layout.annotations if "off scale" in a.text]
    assert len(off) == 1 and "FY24" in off[0] and "−553" in off[0]
    xs = [x for t in traces for x in t.x]
    assert min(xs) >= C.MARGIN_RANGE[0] and max(xs) <= C.MARGIN_RANGE[1]


def test_coincident_margin_points_share_one_label():
    text, pos = C._period_labels([12.2, 12.1, 25.4, 17.9], ["FY24", "H1'25", "FY25", "H1'26"])
    assert sorted(t for t in text if t) == ["FY24 · H1'25", "FY25", "H1'26"]


def test_tokens_figure_is_log_scale_per_day_and_never_joins_scopes():
    fig = C.build_labs_tokens()
    assert fig.layout.yaxis.type == "log"
    traces = _data_traces(fig)
    hovers = [cd for t in traces for cd in t.customdata]
    assert any("over 3.2 quadrillion tokens per month" in cd[3] for cd in hovers)
    for t in traces:
        assert {cd[2] for cd in t.customdata} == {L.SCOPE_LABELS[t.meta["scope"]]}
    g = next(t for t in traces if t.meta == {"entity": "Google", "scope": "all"})
    assert max(g.y) == pytest.approx(3.2e15 / 30)
    # Epoch's own "exclude from graph view" rows are not drawn
    assert not any("Codex" in str(cd[1]) for cd in hovers)


def test_charts_read_no_clock():
    src = (ROOT / "components" / "charts" / "lab_economics.py").read_text()
    assert "datetime" not in src and "date.today" not in src and "now(" not in src


def test_empty_inputs_render_empty_states():
    empty = pd.DataFrame()
    for f in (C.build_labs_runrate, C.build_labs_margins, C.build_labs_tokens):
        fig = f(empty)
        assert fig.layout.annotations and not fig.data
    assert C.build_labs_commitments(L.commitment_rows().iloc[0:0]).layout.annotations


def test_ledger_html_escapes_and_links_safely():
    led = L.load_ledger(include_excluded=True).head(3).copy()
    led.loc[led.index[0], "entity"] = "<script>alert(1)</script>"
    led.loc[led.index[1], "source_url"] = "javascript:alert(1)"
    led.loc[led.index[2], "ledger_state"] = "excluded"
    led.loc[led.index[2], "note"] = "Excluded: <b>because</b>"
    h = C.labs_ledger_html(led, L.commitment_rows())
    assert "<script>" not in h and "&lt;script&gt;" in h
    assert 'href="javascript' not in h
    hrefs = re.findall(r'href="([^"]+)"', h)
    assert hrefs and all(u.startswith("https://") for u in hrefs)
    assert h.count('rel="noopener noreferrer"') == len(hrefs)
    assert "Excluded: &lt;b&gt;because&lt;/b&gt;" in h
    assert h.index("lab-ledger__excluded") > h.index("<tbody>")


# ── Wiring: pre-render, API, export, captions, app.js, Dash ─────────────────

def test_labs_prerender_matches_live(tmp_path):
    written = build_static.export_default_figures(tmp_path)
    live = json.loads(api.update_econ_labs())
    for fid, key in (("econ_labs_runrate", "runrate"), ("econ_labs_margins", "margins"),
                     ("econ_labs_commitments", "commitments"),
                     ("econ_labs_tokens", "tokens")):
        assert f"{fid}.json" in written
        pre = json.loads((tmp_path / f"{fid}.json").read_text())
        assert pre["data"] == live[key]["data"], fid
        assert pre["layout"] == live[key]["layout"], fid
    assert "econ_labs_ledger.json" in written
    ledger = json.loads((tmp_path / "econ_labs_ledger.json").read_text())
    assert ledger == {"html": live["ledger_html"]}


def test_caption_credits_sources():
    c = CAPTIONS["economics_labs"]
    for s in ("Epoch AI", "CC BY 4.0", "EDGAR", "HKEX", "across models", "×13", "÷ 30",
              "$518B", "$665B", "not a gross margin"):
        assert s in c, s
    assert not re.search(r"hir(e|ing)|job|career|applica", c, re.I)


def test_export_on_economics_labs_returns_the_ledger_and_ignores_filters():
    import static_api
    frame, name = export_frame_for_tab("economics", static_api._DF, ["Anthropic"], 50, "zzz",
                                       econ_args=["labs"])
    same, _ = econ_export(["labs"])
    assert name == "ai_frontier_labs.csv"
    assert frame.equals(same)
    for col in ("badge", "source_url", "credit", "ledger_state", "status"):
        assert col in frame.columns
    rr = frame[frame["chart"] == "runrate"]
    assert (rr["credit"] == L.EPOCH_CREDIT).all() and len(rr) == len(L.runrate_frame())
    led = frame[frame["chart"] == "ledger"]
    assert len(led) == len(_ledger_csv())


def test_app_js_places_labs_first_and_the_ledger_last():
    views = re.search(r"const ECON_VIEWS = \[(.*?)\n\];", APP_JS, re.S).group(1)
    labs = views[views.index('view: "labs"'):]
    order = [labs.index(k) for k in ('"economics_labs"', '"econ_labs_runrate"',
                                     '"econ_labs_tokens"', '"economics_open_gap"',
                                     '"econ_open_gap_price"', '"econ_labs_ledger"')]
    assert order == sorted(order)
    assert "sections" in labs
    assert "labs:" not in APP_JS.split("const ECON_REFRESHERS = {")[1].split("};")[0]
    assert "(v.sections || [v]).forEach" in APP_JS
    assert '"economics_labs"' in re.search(r"const TAB_CAPTIONS = \{(.*?)\n\};",
                                           APP_JS, re.S).group(1)


def test_dash_labs_view_draws_the_same_figures_and_ledger():
    import app as dash_app
    out = dash_app.toggle_econ_view("labs")
    live = json.loads(api.update_econ_labs())
    for fig, key in zip(out[9:13], ("runrate", "margins", "commitments", "tokens")):
        assert json.loads(fig.to_json())["data"] == live[key]["data"]
    ledger = out[13]
    assert "lab-ledger" in ledger.className
    serve = dash_app.toggle_econ_view("serve")
    assert len(serve) == len(out) == 14


def test_badge_css_is_neutral_grey():
    css = (ROOT / "assets" / "style.css").read_text()
    for cls in (".lab-badge--disclosed", ".lab-badge--reported", ".lab-badge--estimated"):
        assert cls in css
    block = css[css.index("Economics › Labs ledger"):css.index("Responsive: 1024px")]
    for hexc in re.findall(r"#([0-9a-fA-F]{6})\b", block):
        r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
        assert r == g == b, f"#{hexc} is not a neutral grey"


def test_the_short_commitments_box_matches_its_figure():
    """The commitments strip opts out of the site-wide chart clamp (shell.css)
    and holds the box app.js names, which must be the figure's own height."""
    m = re.search(r"const ECON_CHART_MIN_HEIGHT = \{ econ_labs_commitments: (\d+) \};", APP_JS)
    assert m and int(m.group(1)) == C.build_labs_commitments().layout.height
    shell = (ROOT / "docs" / "assets" / "shell.css").read_text()
    assert '[id^="chart-"].chart--own-height' in shell
