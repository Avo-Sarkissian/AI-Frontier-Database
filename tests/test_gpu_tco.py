"""data/gpu_tco.py: owned cost per GPU-hour, financing presets, the rental
ladder rows and the weekly GetDeploying slice (Economics › Compute)."""
from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import pytest

import build_static
from data import gpu_tco as T
from data.econ_common import STATUS_VALUES

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "econ"
SOFR_FIXED = {"pct": 3.93, "date": "2026-10-26", "source_url": T.SOFR_URL}


@pytest.fixture
def sofr(monkeypatch):
    """Pin SOFR so worked figures do not move with the daily fixing."""
    T.reset_cache()
    monkeypatch.setattr(T, "latest_sofr", lambda: dict(SOFR_FIXED))
    yield
    T.reset_cache()


@pytest.fixture
def no_sofr(monkeypatch):
    T.reset_cache()
    monkeypatch.setattr(T, "latest_sofr", lambda: None)
    yield
    T.reset_cache()


# ── arithmetic ────────────────────────────────────────────────────────────────

def test_crf_matches_closed_form():
    assert T.crf(0.09, 5) == pytest.approx(0.2570925, abs=1e-7)
    assert T.crf(0, 5) == 0.2
    with pytest.raises(ValueError):
        T.crf(0.1, 0)
    with pytest.raises(ValueError):
        T.crf(-0.01, 5)


def test_report_h100_figure_reproduces():
    """The report's worked H100 example: 9%, 5 yr, PUE 1.3, $0.087/kWh,
    $200/kW-month gives $1.41 at full use (4 yr $1.59, 6 yr $1.29)."""
    f = lambda n: T.owned_breakdown(31250, 1.275, 1.3, 0.087, 200, 0.09, n, 1.0)["total_100"]
    assert f(5) == pytest.approx(1.4107, abs=1e-3)
    assert f(4) == pytest.approx(1.5946, abs=1e-3)
    assert f(6) == pytest.approx(1.2888, abs=1e-3)
    at70 = T.owned_breakdown(31250, 1.275, 1.3, 0.087, 200, 0.09, 5, 0.70)["total_at_util"]
    assert at70 == pytest.approx(2.0153, abs=1e-3)


def test_default_worked_example(sofr):
    """H100, CoreWeave 6 yr at SOFR 3.93% + 5.50%, 60% use, curated inputs."""
    r = T.compute()
    assert (r["gpu"], r["financing"]["preset"], r["util_pct"]) == ("h100", "coreweave", 60)
    bd = r["selected"]["breakdown"]
    expect_capital = 31250 * T.crf(0.0943, 6) / 8760
    assert bd["capital"] == pytest.approx(expect_capital, rel=1e-12)
    assert bd["power"] == pytest.approx(1.275 * 1.3 * 0.0909, rel=1e-12)
    assert bd["colo"] == pytest.approx(195.94 * 1.275 * 12 / 8760, rel=1e-12)
    assert bd["total_100"] == pytest.approx(1.2983, abs=1e-4)
    assert bd["total_at_util"] == pytest.approx(2.1639, abs=1e-4)
    assert r["financing"]["rate"]["value"] == pytest.approx(0.0943)


def test_utilisation_divides_owned_only(sofr):
    lo, hi = T.compute(util_pct=50), T.compute(util_pct=100)
    for a, b in zip(lo["rows"], hi["rows"]):
        if a["breakdown"]:
            assert a["breakdown"]["total_at_util"] == pytest.approx(
                a["breakdown"]["total_100"] / 0.5)
            assert b["breakdown"]["total_at_util"] == pytest.approx(b["breakdown"]["total_100"])
    la, lb = T.ladder_rows(lo), T.ladder_rows(hi)
    rent_a = la[la["rung"] != "owned"].reset_index(drop=True)
    rent_b = lb[lb["rung"] != "owned"].reset_index(drop=True)
    pd.testing.assert_frame_equal(rent_a, rent_b)
    o = lambda d: float(d[d["rung"] == "owned"]["usd_per_gpu_hr"].iloc[0])
    assert o(la) == pytest.approx(2 * o(lb))


def test_shares_sum_to_one_and_units(sofr):
    r = T.compute()
    fac = r["facility"]
    priced = [x for x in r["rows"] if x["breakdown"]]
    assert len(priced) >= 5
    for x in priced:
        bd, kw = x["breakdown"], x["inputs"]["it_kw"]["value"]
        assert abs(sum(bd["shares"].values()) - 1.0) < 1e-9
        assert bd["power"] == kw * fac["pue"]["value"] * fac["usd_per_kwh"]["value"]
        assert bd["colo"] == fac["colo_usd_kw_month"]["value"] * kw * 12.0 / 8760


def test_unpriced_gpus_are_not_priced(sofr):
    r = T.compute("mi300x")
    sel = r["selected"]
    assert sel["breakdown"] is None and sel["missing"] == ["CAPEX"]
    assert sel["inputs"]["capex_usd"]["value"] is None
    assert sel["inputs"]["capex_usd"]["status"] == "unavailable"
    for key in ("b300", "mi300x", "mi325x", "mi355x"):
        row = next(x for x in r["rows"] if x["spec"].key == key)
        assert row["breakdown"] is None, key
    typed = T.compute("mi300x", capex_usd=20000)
    assert typed["selected"]["breakdown"]["total_100"] > 0
    assert typed["selected"]["inputs"]["capex_usd"]["typed"] is True


def test_typed_capex_and_kw_change_only_the_selected_gpu(sofr):
    base = T.compute("h100")
    typed = T.compute("h100", capex_usd=40000, it_kw=2.0)
    for a, b in zip(base["rows"], typed["rows"]):
        if a["spec"].key == "h100":
            assert b["breakdown"]["total_100"] > a["breakdown"]["total_100"]
        elif a["breakdown"]:
            assert b["breakdown"]["total_100"] == a["breakdown"]["total_100"]


def test_typed_values_are_clamped_and_junk_ignored(sofr):
    r = T.compute(rate_pct=999, life_years=0, capex_usd="abc", pue="")
    assert r["financing"]["rate"]["value"] == 0.5
    assert r["financing"]["life"]["value"] == 1.0
    assert r["selected"]["inputs"]["capex_usd"]["typed"] is False
    assert r["facility"]["pue"]["typed"] is False


def test_bad_controls_fall_back_to_defaults(sofr):
    r = T.compute("no-such-gpu", "no-such-preset", "abc")
    assert (r["gpu"], r["financing"]["preset"], r["util_pct"]) == ("h100", "coreweave", 60)


def test_default_preset_uses_live_sofr(sofr):
    fin = T.resolve_financing("coreweave")
    assert fin["rate"]["value"] == pytest.approx(0.0943)
    assert "2026-10-26" in fin["rate"]["text"] and "3.93" in fin["rate"]["text"]
    assert fin["rate"]["status"] == "derived"


def test_without_sofr_spread_presets_are_unavailable_and_fixed_still_computes(no_sofr):
    cw = T.compute(preset="coreweave")
    assert cw["financing"]["rate"]["value"] is None
    assert cw["financing"]["rate"]["status"] == "unavailable"
    assert "SOFR unavailable" in cw["financing"]["rate"]["text"]
    assert not any(x["breakdown"] for x in cw["rows"])
    lam = T.compute(preset="lambda")
    assert lam["financing"]["rate"]["value"] == pytest.approx(0.0678)
    assert lam["selected"]["breakdown"] is not None
    from components.charts.econ_compute import build_tco_stack
    fig = build_tco_stack(cw)
    assert "rate unavailable" in fig.layout.annotations[0].text


def test_preset_inherit_is_labelled(sofr):
    neb = T.resolve_financing("nebius")
    assert neb["life"]["value"] == 4
    assert neb["rate"]["value"] == pytest.approx(0.0943)
    assert "discloses no borrowing rate" in neb["rate"]["note"]
    cru = T.resolve_financing("crusoe")
    assert cru["life"]["value"] == 6 and "discloses no useful life" in cru["life"]["note"]
    assert cru["rate"]["value"] == pytest.approx((3.93 + 4.875) / 100)
    for key, life in (("semianalysis", 4), ("burry", 3)):
        f = T.resolve_financing(key)
        assert f["life"]["value"] == life
        assert "discloses no borrowing rate" in f["rate"]["note"]
    burry = T.resolve_financing("burry")
    assert "argument" in burry["label"].lower() and "argument" in burry["life"]["note"].lower()
    lam = T.resolve_financing("lambda")
    assert lam["rate"]["text"] == "6.78% fixed" and lam["life"]["value"] == 5


def test_every_resolved_input_carries_provenance(sofr):
    r = T.compute()
    for x in r["rows"]:
        for p in x["inputs"].values():
            assert p["status"] in STATUS_VALUES
            if p["value"] is not None:
                assert p["source_url"].startswith("https://") and p["as_of"]
    for p in r["facility"].values():
        assert p["status"] in STATUS_VALUES and p["source_url"].startswith("https://")


def test_kw_from_epoch_tdp_ratio():
    T.reset_cache()
    specs = T.epoch_specs()
    if "AMD Instinct MI300X" not in specs:
        pytest.skip("Epoch mirror absent")
    kw = T.resolve_gpu_inputs("mi300x")["it_kw"]
    tdp = specs["AMD Instinct MI300X"]["tdp_w"]
    h100 = specs["NVIDIA H100 SXM5 80GB"]["tdp_w"]
    assert kw["value"] == pytest.approx(tdp * 1.275 / h100)
    assert kw["status"] == "derived" and "Epoch AI TDP" in kw["note"]


# ── weekly slice ─────────────────────────────────────────────────────────────

def _sample() -> pd.DataFrame:
    return pd.read_csv(FX / "gd_weekly_sample.csv", dtype=str, keep_default_na=False)


def test_slim_weekly_whitelists_and_never_merges():
    src = _sample()
    out = T.slim_weekly(src)
    assert set(out["series"]) <= set(T.SERIES)
    assert set(out["gpu_key"]) == {"h100", "mi300x"}          # nvidia-a100 is not priced
    assert not out.duplicated(["gpu_key", "series", "date"]).any()
    # every median is the source row's string, verbatim
    key = {("ON_DEMAND", ""): "on_demand", ("SPOT", ""): "spot",
           ("RESERVATION", "12"): "reserved_12m", ("RESERVATION", "36"): "reserved_36m"}
    slug = {"nvidia-h100": "h100", "amd-mi300x": "mi300x"}
    expect = {}
    for r in src.to_dict("records"):
        s = key.get((r["billing_type"], r["reservation_months"]))
        if s and r["gpu_slug"] in slug:
            expect[(slug[r["gpu_slug"]], s, r["date"])] = r["median_price"]
    got = {(r["gpu_key"], r["series"], r["date"]): r["median_usd"]
           for r in out.to_dict("records")}
    # exactly the whitelisted rows: the decoy billing types (ON_DEMAND_MINUTE,
    # RESERVATION_12_MONTH, CUSTOM, other terms) are dropped, never folded in
    assert got == expect
    assert {"ON_DEMAND_MINUTE", "RESERVATION_12_MONTH", "CUSTOM"} <= set(src["billing_type"])


def test_slim_weekly_keeps_a_missing_week_missing():
    out = T.slim_weekly(_sample())
    od = out[(out["gpu_key"] == "h100") & (out["series"] == "on_demand")]
    assert "2026-10-05" not in set(od["date"])
    assert len(od) == 5


def test_slim_weekly_raises_on_a_renamed_column():
    with pytest.raises(ValueError, match="median_price"):
        T.slim_weekly(_sample().rename(columns={"median_price": "median"}))


def test_slim_weekly_is_small_bounded_and_byte_stable():
    a, b = T.slim_weekly_csv(), T.slim_weekly_csv()
    if not a:
        pytest.skip("GetDeploying archive absent")
    assert a == b
    limit = build_static.DERIVED_MAX_BYTES[T.DERIVED_WEEKLY]
    assert len(a.encode()) < limit
    df = pd.read_csv(io.StringIO(a), dtype=str)
    dates = pd.to_datetime(df["date"])
    assert (dates.max() - dates.min()).days <= T.SERIES_WEEKS * 7
    assert list(df.columns) == T.WEEKLY_COLUMNS


def test_sofr_and_epoch_slices():
    s = T.sofr_latest_csv()
    if s:
        df = pd.read_csv(io.StringIO(s))
        assert len(df) == 1 and 0 <= float(df["percent_rate"].iloc[0]) <= 25
    e = T.epoch_specs_csv()
    if e:
        df = pd.read_csv(io.StringIO(e))
        assert set(df["epoch_name"]) == {g.epoch_name for g in T.GPUS}
        assert df["tdp_w"].notna().all()


# ── ladder ────────────────────────────────────────────────────────────────────

def test_ladder_rows_are_ordered_dated_and_sourced(sofr):
    lad = T.ladder_rows(T.compute("h100"))
    order = [k for k, _ in T.LADDER_RUNGS]
    ranks = [order.index(r) for r in lad["rung"]]
    assert ranks == sorted(ranks)
    assert lad["rung"].iloc[0] == "owned"
    for r in lad.to_dict("records"):
        assert r["source_url"].startswith("https://"), r
        assert r["as_of"] and r["status"] in STATUS_VALUES, r
        assert r["usd_per_gpu_hr"] > 0
    assert set(lad["rung"]) >= {"inferencex", "contract_index", "marketplace", "neocloud_list",
                                "hyperscaler"}
    idx = lad[lad["rung"] == "contract_index"]
    assert sorted(idx["usd_per_gpu_hr"]) == [1.70, 2.35]


def test_ladder_has_no_caption_only_rows():
    pts = pd.read_csv(ROOT / T.PRICE_POINTS_FILE, dtype=str, keep_default_na=False)
    assert set(pts["display"]) <= {"plot", "caption"}
    assert len(T.load_price_points()) == (pts["display"] == "plot").sum()


def test_marketplace_rung_is_the_latest_week_per_series(sofr):
    lad = T.ladder_rows(T.compute("h100"))
    mk = lad[lad["rung"] == "marketplace"]
    wk = T.weekly_for("h100")
    newest = wk["date"].max()
    assert set(mk["period"]) == {newest}
    for r in mk.to_dict("records"):
        s = next(k for k, v in T.SERIES_LABELS.items() if v == r["label"])
        w = wk[(wk["series"] == s) & (wk["date"] == newest)]
        assert float(w["median_usd"].iloc[0]) == r["usd_per_gpu_hr"]
    assert "all form factors" in mk["basis"].iloc[0]


def test_export_frame_has_every_gpu_and_the_ladder(sofr):
    ex = T.export_frame(T.compute("b200"))
    tco = ex[ex["chart"] == "tco"]
    assert list(tco["gpu_key"]) == list(T.GPU_KEYS)
    for col in ("capex_usd_status", "capex_usd_source_url", "rate_pct", "owned_usd_gpu_hr_at_use",
                "share_capital", "sofr_date"):
        assert col in ex.columns
    lad = ex[ex["chart"] == "ladder"]
    assert len(lad) and set(lad["gpu_key"]) == {"b200"}
