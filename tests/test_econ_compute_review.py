"""Economics › Compute review fixes (F-gpu-hour review, 2026-10-27): one
$/GPU-hr basis on the ladder and the weekly chart, GetDeploying's
offering-level medians labelled as such, capex bases named per bar, ladder
labels that never overprint, typed values that say when they were clamped,
and Dash/static parity for the box tooltips, bounds and label pairing."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from components.charts.econ_compute import (
    build_rental_series, build_spread_ladder, build_tco_stack, label_layout, LABEL_PAD_PX,
    LABEL_PLOT_PX,
)
from data import gpu_tco as T

ROOT = Path(__file__).resolve().parent.parent
INDEX = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "docs" / "app.js").read_text(encoding="utf-8")
SOFR_FIXED = {"pct": 3.93, "date": "2026-10-26", "source_url": T.SOFR_URL}


@pytest.fixture
def sofr(monkeypatch):
    T.reset_cache()
    monkeypatch.setattr(T, "latest_sofr", lambda: dict(SOFR_FIXED))
    yield
    T.reset_cache()


# ── one basis ─────────────────────────────────────────────────────────────────

def test_ladder_owned_mark_is_per_paid_hour_like_every_rental(sofr):
    r = T.compute()
    lad = T.ladder_rows(r)
    owned = float(lad[lad["rung"] == "owned"]["usd_per_gpu_hr"].iloc[0])
    assert owned == pytest.approx(1.2983, abs=1e-4)          # full use, not ÷ 0.6
    assert "busy or idle" in lad[lad["rung"] == "owned"]["basis"].iloc[0]
    fig = build_spread_ladder(r, lad)
    title = fig.layout.title.text
    hyp = lad[lad["rung"] == "hyperscaler"]["usd_per_gpu_hr"].max()
    assert f"{hyp / owned:.1f}× the owned cost" in title       # 12.29 / 1.2983 = 9.5×
    assert "9.5×" in title
    # every rung moves together with USE, so the ladder does not move at all
    for u in (10, 60, 100):
        other = T.ladder_rows(T.compute(util_pct=u))
        assert list(other["usd_per_gpu_hr"]) == list(lad["usd_per_gpu_hr"])


def test_ladder_subtitle_is_a_complete_sentence():
    for g in T.GPU_KEYS:
        for u in (10, 60, 100):
            r = T.compute(g, util_pct=u)
            title = build_spread_ladder(r, T.ladder_rows(r)).layout.title.text
            assert "is not<br>" not in title and "and is not" not in title
            assert "per paid GPU-hour, busy or idle" in title
            assert not re.search(r"<br>\s*·", title)


def test_weekly_owned_line_is_per_paid_hour(sofr):
    r = T.compute(util_pct=40)
    fig = build_rental_series(r, T.weekly_for("h100"))
    o = r["selected"]["breakdown"]["total_100"]
    assert any(s.y0 == o and s.line.dash == "dot" for s in fig.layout.shapes)
    key = next(t for t in fig.data if t.name.startswith("owned"))
    assert f"${o:,.2f} per paid GPU-hour" in key.name
    # The stack prints the per-busy-hour figure; the key names it too, so the
    # two owned numbers on the Compute screen are never unlabelled.
    busy = r["selected"]["breakdown"]["total_at_util"]
    assert f"${busy:,.2f} per busy hour at 40% use" in key.name


# ── GetDeploying medians are over offerings ─────────────────────────────────

def test_marketplace_basis_names_offerings_and_providers():
    lad = T.ladder_rows(T.compute("h100"))
    wk = T.weekly_for("h100")
    newest = wk["date"].max()
    for row in lad[lad["rung"] == "marketplace"].to_dict("records"):
        assert "median across" not in row["basis"]
        s = next(k for k, v in T.SERIES_SHORT.items() if v == row["label"])
        hit = wk[(wk["series"] == s) & (wk["date"] == newest)].iloc[0]
        assert (f"median of {int(hit['offering_count'])} offering" in row["basis"]
                and f"from {int(hit['provider_count'])} provider" in row["basis"])


def test_slim_weekly_carries_offering_count_verbatim():
    arch = pd.read_csv(ROOT / "tests" / "fixtures" / "econ" / "gd_weekly_sample.csv",
                       dtype=str, keep_default_na=False)
    out = T.slim_weekly(arch)
    assert list(out.columns) == T.WEEKLY_COLUMNS and "offering_count" in out.columns
    slug = {g.key: g.gd_slug for g in T.GPUS}
    keyed = arch.set_index(["gpu_slug", "date", "median_price", "provider_count"])
    assert not out.empty
    for r in out.to_dict("records"):
        hit = keyed.loc[(slug[r["gpu_key"]], r["date"], r["median_usd"], r["provider_count"])]
        assert r["offering_count"] in set(pd.DataFrame(hit).T["offering_count"]
                                          if isinstance(hit, pd.Series)
                                          else hit["offering_count"])


def test_offerings_phrase():
    assert T.offerings_phrase(109, 39) == "median of 109 offerings from 39 providers"
    assert T.offerings_phrase("1", "1.0") == "median of 1 offering from 1 provider"
    assert T.offerings_phrase(None, 3) == "median of offerings from 3 providers"


def test_series_hover_says_offerings():
    fig = json.loads(build_rental_series(T.compute("h100"), T.weekly_for("h100")).to_json())
    tr = next(t for t in fig["data"] if t.get("meta") == "on_demand")
    assert any("offerings from" in str(c) for c in tr["customdata"])
    assert "providers<extra>" not in tr["hovertemplate"]


# ── capex bases ───────────────────────────────────────────────────────────────

def test_every_capex_default_names_its_basis():
    defaults = T.load_gpu_defaults()
    for key in T.GPU_KEYS:
        row = defaults[key]["capex_usd"]
        if row["value"]:
            assert row["basis"] in T.CAPEX_BASES, key
        else:
            assert row["basis"] == "", key
    bases = {k: defaults[k]["capex_usd"]["basis"] for k in ("h100", "gb200", "h200", "b200")}
    assert bases["h100"] == bases["gb200"] != bases["h200"] == bases["b200"]


def test_tco_bars_and_hover_name_the_basis_and_no_like_for_like_claim():
    r = T.compute("h200")
    fig = build_tco_stack(r)
    ticks = list(fig.layout.yaxis.ticktext)
    for row, tick in zip(r["rows"], ticks):
        b = row["inputs"]["capex_usd"]["basis"]
        if b:
            assert T.CAPEX_BASES[b][0] in tick
    sel = next(i for i, x in enumerate(r["rows"]) if x["selected"])
    assert "capex basis:" in fig.data[0].customdata[sel]
    assert "like with like" not in (T.compute.__doc__ or "")
    typed = T.compute("h200", capex_usd=50000)
    assert typed["selected"]["inputs"]["capex_usd"]["basis"] == "typed"
    exp = T.tco_export_frame(r)
    assert exp.set_index("gpu_key").at["h200", "capex_basis"].startswith("a 2026 street")


# ── ladder labels ─────────────────────────────────────────────────────────────

def test_ladder_labels_never_overprint_and_name_every_price():
    for g in T.GPU_KEYS:
        lad = T.ladder_rows(T.compute(g)).reset_index(drop=True)
        if lad.empty:
            continue
        v = lad["usd_per_gpu_hr"].astype(float)
        lo, hi = math.log10(v.min() * 0.75), math.log10(v.max() * 1.35)
        lay = label_layout(lad, lo, hi)
        for rung in set(lad["rung"]):
            mine = [x for x, rr in zip(lay, lad["rung"]) if rr == rung and x["text"]]
            for i, a in enumerate(mine):
                for b in mine[i + 1:]:
                    if a["slot"] == b["slot"]:
                        assert (a["x1"] + LABEL_PAD_PX <= b["x0"]
                                or b["x1"] + LABEL_PAD_PX <= a["x0"]), (g, a, b)
            text = " ".join(x["text"] for x in mine)
            for val in lad[lad["rung"] == rung]["usd_per_gpu_hr"]:
                assert f"${val:,.2f}" in text, (g, rung, val)
        # the dearest mark is labelled toward the inside, not cut at the edge
        top = lay[int(v.idxmax())]
        if top["text"]:
            assert top["pos"].endswith("left"), (g, top)


def test_ladder_labels_stay_off_the_owned_rule_when_a_side_is_free(sofr):
    """"IX own $1.17", centred on a mark just left of the owned $1.30 rule,
    was struck through by it. A label crosses the rule only when no side and
    no slot clears it."""
    for g in T.GPU_KEYS:
        r = T.compute(g)
        lad = T.ladder_rows(r).reset_index(drop=True)
        if lad.empty or not (lad["rung"] == "owned").any():
            continue
        v = lad["usd_per_gpu_hr"].astype(float)
        lo, hi = math.log10(v.min() * 0.75), math.log10(v.max() * 1.35)
        lay = label_layout(lad, lo, hi)
        own = float(lad[lad["rung"] == "owned"]["usd_per_gpu_hr"].iloc[0])
        ox = (math.log10(own) - lo) / (hi - lo) * LABEL_PLOT_PX
        for x, rung, val in zip(lay, lad["rung"], v):
            if rung == "owned" or not x["text"]:
                continue
            mx = (math.log10(val) - lo) / (hi - lo) * LABEL_PLOT_PX
            w = x["x1"] - x["x0"]
            crosses = x["x0"] < ox < x["x1"]
            # had the other side been inside the plot, it would have been used
            if crosses:
                left_ok = mx - w >= 0 and mx < ox
                right_ok = mx + w <= LABEL_PLOT_PX and mx > ox
                assert not (left_ok or right_ok), (g, x)


def test_owned_mark_names_both_bases_when_use_is_below_full(sofr):
    r = T.compute("h100", util_pct=60)
    fig = build_spread_ladder(r, T.ladder_rows(r))
    owned = next(t for t in fig.data if t.meta == "owned")
    bd = r["selected"]["breakdown"]
    assert owned.text[0] == (f"owned ${bd['total_100']:,.2f} per paid hr · "
                             f"${bd['total_at_util']:,.2f} per busy hr at 60% use")
    tco = build_tco_stack(r).layout.title.text
    assert (f"${bd['total_at_util']:,.2f} per busy GPU-hour at 60% use "
            f"(${bd['total_100']:,.2f} per paid hour" in tco)
    full = T.compute("h100", util_pct=100)
    f2 = build_spread_ladder(full, T.ladder_rows(full))
    assert next(t for t in f2.data if t.meta == "owned").text[0].endswith("per paid hr")


def test_neighbours_in_price_order_alternate_slots():
    df = pd.DataFrame({"rung": ["marketplace"] * 3,
                       "label": ["12-month reserved", "36-month reserved", "spot"],
                       "usd_per_gpu_hr": [5.50, 5.00, 5.27]})
    lay = label_layout(df, math.log10(1.0), math.log10(20.0))
    labelled = [x for x in lay if x["text"]]
    assert len(labelled) < 3                      # the crowded one is folded in
    assert all("$" in x["text"] for x in labelled)
    joined = " ".join(x["text"] for x in labelled)
    assert all(p in joined for p in ("$5.00", "$5.27", "$5.50"))


# ── clamped inputs ────────────────────────────────────────────────────────────

def test_clamped_values_are_named_in_the_subtitle(sofr):
    r = T.compute("h100", capex_usd=-10000, life_years=0, it_kw=99, pue=0.5)
    title = build_tco_stack(r).layout.title.text
    assert "capex $1,000 typed (clamped)" in title
    assert "kW 20.000 typed (clamped)" in title
    assert "1 yr typed (clamped)" in title
    assert "PUE 1 typed (clamped)" in title
    ok = build_tco_stack(T.compute("h100", capex_usd=40000)).layout.title.text
    assert "capex $40,000 typed" in ok and "(clamped)" not in ok
    rate = T.compute(rate_pct=80)["financing"]["rate"]
    assert rate["value"] == 0.5 and rate["clamped"] and "(clamped)" in rate["text"]


def test_box_bounds_come_from_the_manifest_not_the_markup():
    opts = T.compute_options()
    assert opts["bounds"]["life"] == T.LIFE_RANGE
    block = INDEX[INDEX.index('id="econ-compute-controls"'):]
    block = block[:block.index("</div>")]
    assert 'min="0"' not in block
    assert "el.min = String(r[0]); el.max = String(r[1]);" in APP_JS
    for k in ("life", "rate_pct", "capex_k", "it_kw", "pue", "usd_per_kwh", "colo"):
        assert f"b.{k}]" in APP_JS


def test_static_labels_pair_with_their_controls():
    block = INDEX[INDEX.index('id="econ-compute-controls"'):]
    # .econ-pair.fg: inline-flex + nowrap from docs/assets/shell.css (.fg).
    pairs = re.findall(r'<span class="econ-pair fg"[^>]*>(.*?)</span>\s*'
                       r'(?=<span class="econ-pair|<div class="filter-sep"|</span>)',
                       block, flags=re.S)
    ids = ("gpu", "preset", "life", "rate", "capex", "kw", "pue", "kwh", "colo")
    for k in ids:
        assert any(f'id="econ-compute-{k}"' in p and "filter-label" in p for p in pairs), k


# ── Dash parity ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dash_app():
    import app as module
    return module


def test_dash_label_tooltips_follow_gpu_and_preset(dash_app):
    out = dash_app.sync_econ_compute_preset("nebius")
    assert len(out) == 8
    assert out[4] == out[6] and out[5] == out[7]            # wrap == label, life / rate
    assert out[6] == dash_app._compute_placeholders(None, "nebius")["life"][1]
    assert out[6] != dash_app._compute_placeholders(None, "coreweave")["life"][1]
    g = dash_app.sync_econ_compute_gpu("b300")
    assert len(g) == 8 and g[4] == g[6] and g[5] == g[7]
    assert g[6] == dash_app._compute_placeholders("b300")["capex"][1]


def test_dash_boxes_carry_bounds_and_pair_with_labels(dash_app):
    def walk(c):
        yield c
        kids = getattr(c, "children", None)
        for k in (kids if isinstance(kids, (list, tuple)) else [kids] if kids is not None else []):
            if hasattr(k, "to_plotly_json"):
                yield from walk(k)

    nodes = list(walk(dash_app.app.layout))
    ctl = next(n for n in nodes if getattr(n, "id", None) == "econ-compute-controls")
    bounds = dash_app._COMPUTE_BOUNDS
    for key, *_ in dash_app._COMPUTE_BOXES:
        box = next(n for n in walk(ctl) if getattr(n, "id", None) == f"econ-compute-{key}")
        assert (box.min, box.max) == tuple(bounds[key]), key
        pair = next(n for n in ctl.children
                    if any(getattr(x, "id", None) == f"econ-compute-{key}-label" for x in walk(n))
                    and getattr(n, "id", None) != "econ-compute-controls")
        assert pair.style.get("whiteSpace") == "nowrap"
        assert any(getattr(x, "id", None) == f"econ-compute-{key}" for x in walk(pair))


def test_ladder_points_are_keyed_by_their_label():
    """Plotly.react skips a label whose text was "" in the previous figure;
    a point id naming the label forces a fresh point (browser-reproduced)."""
    seen = {}
    for g in ("h100", "mi355x", "h100"):
        r = T.compute(g)
        fig = json.loads(build_spread_ladder(r, T.ladder_rows(r)).to_json())
        for t in fig["data"]:
            assert len(t["ids"]) == 1 and t["text"][0] in t["ids"][0]
        seen.setdefault(g, [t["ids"] for t in fig["data"]])
        assert seen[g] == [t["ids"] for t in fig["data"]]      # deterministic
