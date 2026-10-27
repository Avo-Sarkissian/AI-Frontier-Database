"""Economics › Compute: the three figures, their pre-render parity, the
static and Dash wiring, and the ↓CSV."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pandas as pd
import pytest

import build_static
import static_api as api
from captions import CAPTIONS
from components.charts.econ_compute import (
    build_econ_compute, build_rental_series, build_spread_ladder, build_tco_stack,
    MUTED_OPACITY,
)
from data import gpu_tco as T
from static_helpers import ECON_EXPORT_FILENAMES

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text(encoding="utf-8")
INDEX = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
FIG_IDS = ("econ_tco", "econ_ladder", "econ_rental_series")


def _live(*args) -> dict:
    return json.loads(api.update_econ_compute(*args))


# ── figures ───────────────────────────────────────────────────────────────────

def test_series_traces_are_one_source_each():
    r = T.compute("h100")
    fig = json.loads(build_rental_series(r, T.weekly_for("h100")).to_json())
    data_traces = [t for t in fig["data"] if t.get("meta") in T.SERIES]
    assert data_traces, "no weekly series drawn"
    assert len({t["meta"] for t in data_traces}) == len(data_traces)
    for t in data_traces:
        assert t["connectgaps"] is False
        assert t["name"] == T.SERIES_LABELS[t["meta"]]
    others = [t for t in fig["data"] if t.get("meta") not in T.SERIES]
    # only the legend key for the owned line; no index or price-point data
    assert all(t["x"] == [None] and "owned" in t["name"] for t in others)
    assert "GetDeploying (CC BY 4.0)" in json.dumps(fig["layout"]["annotations"])


def test_series_shows_a_gap_for_a_missing_week():
    wk = pd.DataFrame({"gpu_key": "h100", "series": "on_demand",
                       "date": ["2026-09-28", "2026-10-12", "2026-10-19"],
                       "median_usd": [3.5, 3.7, 3.9], "provider_count": [30, 31, 32]})
    fig = build_rental_series(T.compute("h100"), wk)
    tr = next(t for t in fig.data if t.meta == "on_demand")
    assert list(tr.x) == ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]
    assert list(tr.y) == [3.5, None, 3.7, 3.9]


def test_ladder_rungs_ordered_and_dated():
    r = T.compute("h100")
    lad = T.ladder_rows(r)
    fig = json.loads(build_spread_ladder(r, lad).to_json())
    order = fig["layout"]["yaxis"]["categoryarray"]
    rungs = [lab for _k, lab in T.LADDER_RUNGS]
    assert order == [x for x in rungs if x in order]
    assert fig["layout"]["yaxis"]["autorange"] == "reversed"
    assert fig["layout"]["xaxis"]["type"] == "log"
    assert len(fig["data"]) == len(lad)              # one trace per mark
    for t, row in zip(fig["data"], lad.to_dict("records")):
        assert t["mode"] == "markers+text" and len(t["x"]) == 1   # never a joined line
        hover, url = t["customdata"][0]
        assert row["period"] in hover and url == row["source_url"]
        assert url.startswith("https://")
    idx = [t for t in fig["data"] if t["meta"] == "contract_index"]
    assert len(idx) == 2


def test_coincident_marks_share_one_label():
    r = T.compute("h100")
    lad = T.ladder_rows(r)
    fig = build_spread_ladder(r, lad)
    neo = [t for t in fig.data if t.meta == "neocloud_list" and t.x[0] == 3.99]
    assert len(neo) == 2                                  # still two marks
    texts = sorted(t.text[0] for t in neo)
    assert texts[0] == "" and "Lambda · Together AI $3.99" == texts[1]


def test_ladder_owned_floor_and_spread_note():
    r = T.compute("h100")
    fig = build_spread_ladder(r, T.ladder_rows(r))
    owned = r["selected"]["breakdown"]["total_at_util"]
    assert any(s.x0 == owned and s.line.dash == "dash" for s in fig.layout.shapes)
    assert "× the owned cost" in fig.layout.title.text
    b300 = T.compute("b300")
    f2 = build_spread_ladder(b300, T.ladder_rows(b300))
    assert "no owned cost until CAPEX is typed" in f2.layout.title.text
    assert not [t for t in f2.data if t.meta == "owned"]


def test_tco_stack_selected_solid_others_muted_and_shares_in_title():
    r = T.compute("gb200")
    fig = build_tco_stack(r)
    labels = [x["spec"].label for x in r["rows"]]
    for tr in fig.data:
        assert list(tr.y) == labels
        assert tr.marker.opacity[labels.index("GB200 NVL72")] == 1.0
        assert tr.marker.opacity[labels.index("H100 SXM")] == MUTED_OPACITY
    title = fig.layout.title.text
    sh = r["selected"]["breakdown"]["shares"]
    for k in ("capital", "power", "colo"):
        assert f"{k} {sh[k] * 100:.0f}%" in title
    assert [tr.name for tr in fig.data] == ["capital", "power", "colocation"]


def test_tco_stack_handles_empty_and_unpriced(monkeypatch):
    fig = build_tco_stack(T.compute("mi355x"))
    texts = [a.text for a in fig.layout.annotations]
    assert sum("type CAPEX" in t for t in texts) == 4          # B300 + three AMD parts
    assert "type CAPEX" in fig.layout.title.text
    T.reset_cache()
    monkeypatch.setattr(T, "latest_sofr", lambda: None)
    try:
        empty = build_tco_stack(T.compute())
        assert "Nothing can be priced" in empty.layout.annotations[0].text
        assert build_spread_ladder(T.compute(), pd.DataFrame()).layout.annotations
        assert build_rental_series(T.compute(), pd.DataFrame()).layout.annotations
    finally:
        T.reset_cache()


def test_no_hollow_marks_in_economics_compute():
    figs = build_econ_compute(T.compute())
    for fig in figs.values():
        for tr in fig.data:
            sym = getattr(getattr(tr, "marker", None), "symbol", None)
            syms = sym if isinstance(sym, (list, tuple)) else [sym]
            assert not any(s and "open" in str(s) for s in syms)


# ── parity, arity, manifest ──────────────────────────────────────────────────

def test_compute_prerender_matches_live_default(tmp_path):
    figs = build_static.export_default_figures(tmp_path)
    live = _live()
    for fid, key in zip(FIG_IDS, ("tco", "ladder", "series")):
        assert f"{fid}.json" in figs
        pre = json.loads((tmp_path / f"{fid}.json").read_text())
        assert pre["data"] == live[key]["data"], fid
        assert pre["layout"] == live[key]["layout"], fid
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["econ_compute_options"] == json.loads(api.econ_compute_options())


def test_shipped_figures_match_live_default():
    """docs/figures as committed equal a fresh default render (a stale
    pre-render visibly changes a few seconds after the page loads)."""
    live = _live()
    for fid, key in zip(FIG_IDS, ("tco", "ladder", "series")):
        p = ROOT / "docs" / "figures" / f"{fid}.json"
        if not p.exists():
            pytest.skip("docs/figures not built")
        pre = json.loads(p.read_text())
        assert pre["data"] == live[key]["data"], fid


def test_update_econ_compute_accepts_all_nulls_and_appended_args():
    a = _live()
    b = _live(None, None, None, None, None, None, None, None, None, None)
    assert a == b
    assert a["gpu"] == "h100" and a["preset"] == "coreweave" and a["util_pct"] == 60
    typed = _live("h100", "coreweave", 60, None, None, 40000, None, None, None, None)
    assert typed["breakdown"]["total_100"] > a["breakdown"]["total_100"]
    sig = inspect.signature(api.update_econ_compute)
    assert list(sig.parameters) == ["gpu_key", "preset", "util_pct", "life_years", "rate_pct",
                                    "capex_usd", "it_kw", "pue", "usd_per_kwh",
                                    "colo_usd_kw_month"]


def test_options_carry_placeholders_and_provenance():
    o = T.compute_options()
    assert o["default_gpu"] == "h100" and o["default_preset"] == "coreweave"
    assert [g["value"] for g in o["gpus"]] == list(T.GPU_KEYS)
    assert o["defaults"]["h100"]["capex_k"] == 31.25
    assert o["defaults"]["mi300x"]["capex_k"] is None
    assert "unavailable" in o["defaults"]["mi300x"]["capex_title"]
    assert o["preset_defaults"]["lambda"]["rate_pct"] == 6.78
    for f in o["facility"].values():
        assert "https://" in f["title"]


# ── wiring ───────────────────────────────────────────────────────────────────

def test_app_js_sends_nulls_not_literal_defaults():
    fn = APP_JS[APP_JS.index("function econComputeArgs"):][:900]
    for box in ("econ-compute-life", "econ-compute-rate", "econ-compute-capex",
                "econ-compute-kw", "econ-compute-pue", "econ-compute-kwh",
                "econ-compute-colo", "econ-util"):
        assert f'numOrNull("{box}")' in fn, box
    assert "capexK * 1000" in fn
    for literal in ("31250", "31.25", "1.275", "0.0909", "195.94", "9.43", "6.78"):
        assert literal not in APP_JS, literal
    assert ('callPy("update_econ_compute", g, p, u, life, rate, capex, kw, pue, kwh, colo)'
            in APP_JS)


def test_compute_view_is_registered_between_serve_and_labs():
    views = re.search(r"const ECON_VIEWS = \[(.*?)\n\];", APP_JS, re.S).group(1)
    i_s, i_c, i_l = (views.index(f'view: "{v}"') for v in ("serve", "compute", "labs"))
    assert i_s < i_c < i_l
    assert '"econ_tco", "econ_ladder", "econ_rental_series"' in views
    assert '"econ-shared-controls", "econ-compute-controls"' in views
    assert '"econ-compute-controls"' in APP_JS.split("const ECON_CONTROL_SPANS")[1][:120]
    assert re.search(r'compute: makeRefresher\("economics-compute"', APP_JS)
    assert 'if (view === "compute") return [view, ...econComputeArgs()];' in APP_JS
    radios = re.findall(r'name="econ-view" value="(\w+)"', INDEX)
    assert radios == ["serve", "compute", "labs"]
    for el in ('id="econ-compute-controls"', 'id="econ-compute-gpu"', 'id="econ-compute-preset"',
               'id="econ-compute-life"', 'id="econ-compute-colo"'):
        assert el in INDEX
    assert "economics_compute" in CAPTIONS
    for credit in ("GetDeploying (CC BY 4.0", "Epoch AI", "CC BY 4.0", "New York",
                   "InferenceX", "Azure Retail Prices API", "AWS Price List"):
        assert credit in CAPTIONS["economics_compute"], credit


def test_compute_modules_ship_and_curated_data_rides_the_data_zip():
    src = inspect.getsource(build_static.build_pybundle)
    for mod in ("data/gpu_tco.py", "components/charts/econ_compute.py"):
        assert f'"{mod}"' in src
    for rel in (*T.CURATED_FILES, T.HYPERSCALER_FILE):
        assert rel in build_static.DATA_CSVS
    for rel in (T.DERIVED_WEEKLY, T.DERIVED_SOFR, T.DERIVED_EPOCH):
        assert rel in build_static.DERIVED_DATA
    shipped = {Path(p).name for p in build_static.DATA_CSVS}
    assert "gpu_prices_weekly.csv" not in shipped and "sofr.csv" not in shipped


def test_compute_export():
    assert ECON_EXPORT_FILENAMES["compute"] == "ai_frontier_gpu_tco.csv"
    assert api.export_csv_filename("economics", ["compute"]) == "ai_frontier_gpu_tco.csv"
    csv_text = api.export_csv(None, 0, "", "economics", None, None, None, None,
                              ["compute", "gb300", "lambda", 80, None, None, None, None,
                               None, None, None])
    df = pd.read_csv(pd.io.common.StringIO(csv_text))
    tco = df[df["chart"] == "tco"]
    assert len(tco) == len(T.GPUS) and set(tco["preset"]) == {"lambda"}
    assert (tco["util_pct"] == 80).all()
    lad = df[df["chart"] == "ladder"]
    assert len(lad) and set(lad["gpu_key"]) == {"gb300"}


def test_dash_compute_callbacks_draw_the_same_figures():
    import app as dash_app
    tco, ladder, series = dash_app.update_econ_compute("h100", "coreweave", 60)
    live = _live("h100", "coreweave", 60)
    assert json.loads(tco.to_json())["data"] == live["tco"]["data"]
    assert json.loads(ladder.to_json())["data"] == live["ladder"]["data"]
    assert json.loads(series.to_json())["data"] == live["series"]["data"]
    styles = dash_app.toggle_econ_view("compute")
    assert styles[-2] == {} and styles[-1]["display"] == "inline-flex"
    assert styles[3]["display"] == "inline-flex"          # shared USE slider shown
    assert dash_app.toggle_econ_view("serve")[-1] == {"display": "none"}
    cleared = dash_app.sync_econ_compute_gpu("mi300x")
    assert cleared[:2] == (None, None) and cleared[2] == "—"
    assert "Compute" in APP_PY and '"value": "compute"' in APP_PY
