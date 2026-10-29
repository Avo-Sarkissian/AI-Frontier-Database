"""Economics › Labs: the value-capture stack (data/lab_economics.stack_rows,
components/charts/lab_economics.build_labs_stack) and its ledger rows."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

import static_api as api
from captions import CAPTIONS, CAPTION_TEXT
from components.charts import lab_economics as C
from data import lab_economics as L
from static_helpers import econ_export

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "docs" / "app.js").read_text()


@pytest.fixture(autouse=True)
def _fresh_caches():
    L.reset_cache()
    yield
    L.reset_cache()


def _stack():
    return L.stack_rows().set_index("layer")


def _marks(fig):
    return [t for t in fig.data if t.x is not None and any(v is not None for v in t.x)]


# ── Ledger rows ──────────────────────────────────────────────────────────────

def test_stack_rows_are_well_formed():
    led = L.load_ledger(include_excluded=True)
    st = led[led["view"] == "stack"]
    assert set(st["layer"]) == {"chips", "capex", "apps", "required"}
    for _, r in st.iterrows():
        assert r["unit"] == "usd" and r["per"] in L.USD_PER_YEAR, r["row_id"]
        if pd.notna(r["value_high"]):
            assert r["value_high"] > r["value"], r["row_id"]
        assert r["note"].strip(), r["row_id"]
    # only stack rows carry a layer; only ranges carry value_high
    assert (led.loc[led["view"] != "stack", "layer"] == "").all()
    assert led.loc[led["view"] != "stack", "value_high"].isna().all()


def test_nvidia_quarter_is_the_filed_10q_figure_annualised():
    r = L.load_ledger().set_index("row_id").loc["nvidia_dc_q2_fy27"]
    assert r["value"] == 89_023e6 and r["per"] == "quarter"
    assert r["badge"] == "disclosed" and r["filing_status"] == "unaudited"
    assert r["source_url"].startswith("https://www.sec.gov/Archives/edgar/data/1045810/")
    assert r["period_end"] == "2026-07-26"
    s = _stack()
    assert s.loc["chips", "lo"] == s.loc["chips", "hi"] == 89_023e6 * 4


def test_capex_layer_sums_the_four_guidance_rows():
    s = _stack()
    assert s.loc["capex", "n_parts"] == 4
    assert s.loc["capex", "lo"] == (195 + 135 + 175 + 220) * 1e9
    assert s.loc["capex", "hi"] == (205 + 145 + 175 + 220) * 1e9
    names = {p["name"] for p in s.loc["capex", "parts"]}
    assert names == {"Alphabet", "Meta", "Microsoft", "Amazon"}
    led = L.load_ledger().set_index("row_id")
    assert "sec.gov" in led.loc["meta_capex_2026", "source_url"]
    # Amazon's is company-wide, Microsoft's calendar-year: the definitions say so
    assert "company-wide" in led.loc["amazon_capex_2026", "definition"]
    assert "calendar-2026" in led.loc["microsoft_capex_cy2026", "definition"]


def test_sequoia_rerun_is_four_times_the_chips_layer():
    s = _stack()
    assert L.SEQUOIA_DC_MULT * L.SEQUOIA_MARGIN_MULT == 4
    assert s.loc["required_now", "mid"] == pytest.approx(s.loc["chips", "mid"] * 4)
    assert s.loc["required_now", "badge"] == "estimated"
    assert s.loc["required_2024", "mid"] == 600e9
    assert s.loc["required_now", "kind"] == s.loc["required_2024", "kind"] == "reference"


def test_labs_layer_is_the_latest_snapshot_per_lab_inside_the_window():
    s = _stack()
    labs = L.stack_lab_runrates()
    assert len(labs) >= 3
    assert labs["company"].is_unique
    assert not labs["revenue_type"].str.lower().eq("period interpolation").any()
    assert s.loc["labs", "lo"] == pytest.approx(labs["annualized_usd"].sum())
    newest = pd.Timestamp(labs["date"].max())
    assert ((newest - pd.to_datetime(labs["date"])).dt.days <= L.STACK_LAB_WINDOW_DAYS).all()
    # each kept row is that company's newest snapshot
    rr = L.runrate_frame()
    snap = rr[rr["revenue_type"].str.lower() != "period interpolation"]
    for _, r in labs.iterrows():
        assert r["date"] == snap[snap["company"] == r["company"]]["date"].max()


def test_labs_window_drops_a_stale_lab(monkeypatch):
    rr = pd.DataFrame({
        "epoch_id": ["a1", "a2", "b1", "c1"], "company": ["A", "A", "B", "C"],
        "palette_name": "", "date": ["2026-01-01", "2026-09-01", "2024-01-01", "2026-08-01"],
        "annualized_usd": [1e9, 2e9, 5e9, 3e9],
        "revenue_type": ["Annualized run rate", "Annualized run rate", "Annualized run rate",
                         "Period interpolation"],
        "scope": "Full company", "confidence": "", "source_type": "", "badge": "reported",
        "source_url": "https://example.com/"})
    monkeypatch.setattr(L, "runrate_frame", lambda: rr)
    got = L.stack_lab_runrates()
    assert list(got["company"]) == ["A"] and float(got["annualized_usd"].iloc[0]) == 2e9


def test_weakest_badge_and_range_format():
    assert L.weakest_badge(["disclosed", "reported"]) == "reported"
    assert L.weakest_badge(["disclosed", "estimated", "reported"]) == "estimated"
    assert L.weakest_badge([]) == "estimated"
    assert L.usd_range(725e9, 745e9) == "$725–745B"
    assert L.usd_range(900e6, 1.1e9) == "$900M–$1.1B"
    led = L.load_ledger().set_index("row_id")
    assert L.fmt_value(led.loc["nvidia_dc_q2_fy27"]) == "$89.02B / quarter"
    assert L.fmt_value(led.loc["meta_capex_2026"]) == "$135–145B"
    assert L.fmt_value(led.loc["microsoft_capex_cy2026"]) == "~$175B"
    assert L.fmt_value(led.loc["cursor_arr_2026_06"]) == ">$4B"


def test_neolab_tally_is_ledger_only():
    led = L.load_ledger().set_index("row_id")
    r = led.loc["dealroom_neolabs_2026_10"]
    assert r["view"] == "ledger" and r["value"] == 23e9 and r["badge"] == "estimated"
    assert "35%" in r["note"]
    blob = json.dumps(json.loads(C.build_labs_stack().to_json())["data"])
    assert "23000000000" not in blob and "Dealroom" not in blob
    assert "29 neolabs (Dealroom)" in C.labs_ledger_html()


# ── The figure ───────────────────────────────────────────────────────────────

def test_stack_figure_layers_marks_and_references():
    fig = C.build_labs_stack()
    marks = _marks(fig)
    assert [t.meta["layer"] for t in marks] == list(L.STACK_LAYERS)
    s = _stack()
    for t in marks:
        r = s.loc[t.meta["layer"]]
        assert t.x[0] == pytest.approx(r["mid"])
        assert t.marker.symbol == C.badge_symbol(r["badge"])
    assert marks[3].text[0] == "$356.1B" and marks[2].text[0] == "$725–745B"
    assert marks[0].text[0] == ">$4B"
    xs = sorted(sh.x0 for sh in fig.layout.shapes)
    assert xs == pytest.approx(sorted([s.loc["required_2024", "mid"],
                                       s.loc["required_now", "mid"]]))
    lo, hi = fig.layout.xaxis.range
    assert lo < math.log10(s["lo"].min()) and hi > math.log10(s["hi"].max())
    assert fig.layout.height == C.STACK_HEIGHT
    assert re.search(r"≈ \d+×", fig.layout.title.text)


def test_reference_labels_point_away_from_each_other():
    fig = C.build_labs_stack()
    notes = {a.text: a for a in fig.layout.annotations}
    old = next(a for t, a in notes.items() if "Sequoia" in t)
    now = next(a for t, a in notes.items() if "needed now" in t)
    assert old.x < now.x
    assert old.xanchor == "right" and now.xanchor == "left"


def test_stack_narrow_patch_moves_labels_above_rows():
    fig = C.build_labs_stack()
    nar = fig.layout.meta["narrow"]
    assert nar["max_width"] == C.NARROW_PX
    assert nar["layout"]["yaxis"]["showticklabels"] is False
    texts = [a["text"] for a in nar["layout"]["annotations"]]
    for lab in (L.STACK_LAYER_LABELS[k] for k in L.STACK_LAYERS):
        assert any(lab in t for t in texts)
    # phone: short reference labels lifted above the top row's label, more
    # room right of the top line, a taller top margin for the wrapped title
    refs = [a for a in nar["layout"]["annotations"] if a["text"].startswith(("← now", "'24"))]
    assert len(refs) == 2 and all(a["yshift"] == 16 for a in refs)
    assert "← now $1.42T" in [a["text"] for a in refs]
    assert nar["layout"]["xaxis"]["range"][1] > fig.layout.xaxis.range[1]
    assert nar["layout"]["margin"]["t"] > fig.layout.margin.t


def test_stack_empty_state():
    assert C.build_labs_stack(L.stack_rows().iloc[0:0]).layout.annotations
    assert not C.build_labs_stack(L.stack_rows().iloc[0:0]).data


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_stack_ships_everywhere():
    assert "stack" in json.loads(api.update_econ_labs())
    labs = APP_JS[APP_JS.index('view: "labs"'):]
    assert labs.index('"econ_labs_tokens"') < labs.index('"econ_labs_stack"') \
        < labs.index('"economics_open_gap"')
    frame, _ = econ_export(["labs"])
    st = frame[frame["chart"] == "stack"]
    assert set(st["layer"]) == {"chips", "capex", "labs", "apps", "required_now",
                                "required_2024"}
    assert (st.loc[st["layer"] == "labs", "credit"] == L.EPOCH_CREDIT).all()
    assert (st.loc[st["layer"] != "labs", "credit"] == "").all()


def test_caption_names_the_stack_sources():
    c = CAPTION_TEXT["economics_labs"]
    for s in ("STACK", "Sequoia", "× 2", "Dealroom", "capex", "10-Q", "Cursor", "floor"):
        assert s in c, s
