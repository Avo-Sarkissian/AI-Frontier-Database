"""Economics › Serve: the serving-cost model (data/serving_cost.py).

Fixture tests/fixtures/econ/ix_points_serve_sample.csv holds 12 real rows
copied on 2026-10-27 from data/carried/econ/inferencex_points.csv (A2's
schema; sources https://inferencex.semianalysis.com/api/v1/views/inference,
run links https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<id>), so
these tests never hit the network and do not move with the hourly data.
"""
import inspect
import math
from pathlib import Path

import pandas as pd
import pytest

from captions import CAPTIONS, CAPTION_TEXT
from data import serving_cost as sc
from data.local_models import GPU_BY_NAME, SLO_FLOORS_TPS, get_local_df

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "econ" / "ix_points_serve_sample.csv"


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(FIXTURE, dtype={"run_id": str})


@pytest.fixture(scope="module")
def local():
    return get_local_df(include_pending=False)


def _row(local, name):
    hit = local[local["name"] == name]
    if hit.empty:
        pytest.skip(f"{name} is not in the current open-weights catalogue")
    return hit.iloc[0]


# ── The formula ──────────────────────────────────────────────────────────────

def test_cost_formula_units():
    assert sc.cost_per_m(2.0, 1000, 1.0) == pytest.approx(0.5556, rel=1e-3)
    assert sc.cost_per_m(2.0, 1000, 0.5) == pytest.approx(2 * sc.cost_per_m(2.0, 1000, 1.0))
    assert sc.cost_per_m(0.0, 1000, 1.0) == 0.0
    for bad in (0, -5, float("nan"), None, "x"):
        assert math.isnan(sc.cost_per_m(2.0, bad, 1.0))
    assert math.isnan(sc.cost_per_m(2.0, 1000, 0))
    assert math.isnan(sc.cost_per_m(float("inf"), 1000, 1))


def test_formula_matches_inferencex_costhOutput():
    """InferenceX views/inference (https://inferencex.semianalysis.com/api/v1/
    views/inference), Llama-3.3-70B-Instruct-FP8, 8k/1k, B200 TRT-LLM FP8 TP1,
    conc 128, run 18924487526 (2025-10-29), fetched 2026-10-27:
    outputTputPerGpu 1043.4813203964407 at h = $1.73 -> costhOutput
    0.4605310571088923."""
    assert sc.cost_per_m(1.73, 1043.4813203964407, 1.0) == pytest.approx(
        0.4605310571088923, rel=1e-6)


def test_deepseek_anchor_in_caption():
    a = sc.DEEPSEEK_ANCHOR
    cost = sc.cost_per_m(a["usd_per_node_hr"], a["decode_tok_s_per_node"], 1.0)
    assert cost == pytest.approx(0.300, abs=5e-4)
    assert a["r1_output_price"] / cost == pytest.approx(7.29, abs=0.01)
    cap = CAPTION_TEXT["economics_serve"]
    for s in ("$0.30", "$2.19", "545%", "CC0"):
        assert s in cap
    assert a["source_url"].startswith("https://github.com/deepseek-ai/open-infra-index/")


def test_run_urls_are_built_and_guarded():
    assert sc.run_url(18924487526) == sc.IX_RUNS_PREFIX + "18924487526"
    assert sc.run_url("28916523710.0") == sc.IX_RUNS_PREFIX + "28916523710"
    assert sc.run_url("abc") == "" and sc.run_url(None) == ""


@pytest.mark.parametrize("url,ok", [
    ("https://github.com/SemiAnalysisAI/InferenceX/actions/runs/18924487526", True),
    ("javascript:alert(1)", False),
    ("http://github.com/SemiAnalysisAI/InferenceX/actions/runs/1", False),
    ("https://evil.example/SemiAnalysisAI/InferenceX/actions/runs/1", False),
    ("https://github.com/SemiAnalysisAI/InferenceX/actions/runs/1?x=<b>", False),
    ("", False), (None, False),
])
def test_safe_run_url(url, ok):
    assert sc.safe_run_url(url) == (url if ok else "")


# ── Service level and the modelled throughput ────────────────────────────────

def test_datacenter_slo_is_35_and_every_estimate_meets_it(local):
    assert SLO_FLOORS_TPS["datacenter"] == 35.0
    assert sc.MIN_INTERACTIVITY == 35.0
    names = ["Llama 3.3 Instruct 70B", "MiniMax-M3", "gpt-oss-120b (High)"]
    for name in names:
        row = local[local["name"] == name]
        if row.empty:
            continue
        for preset in sc.IX_HARDWARE_TO_PRESET.values():
            e = sc.serving_estimate(row.iloc[0], preset, 0.5)
            if e is None:
                continue
            assert e.per_session_tps >= 35.0
            assert e.n_gpus in sc.GPU_COUNTS
            assert e.tps_per_gpu > 0


def test_the_gpu_count_is_the_cost_optimal_one(local):
    """n is the count with the most tokens per GPU among those that serve the
    model at all; every other valid n does no better."""
    from data.local_models import optimal_concurrency, tflops_for_gpu, vram_breakdown
    row = _row(local, "Llama 3.3 Instruct 70B")
    preset = "NVIDIA H100 SXM"
    k = 0.6
    best = sc.serving_estimate(row, preset, k)
    assert best is not None
    g = GPU_BY_NAME[preset]
    for n in sc.GPU_COUNTS:
        vb = vram_breakdown(row["params_b"], "Q8", sc.SERVING_CTX, name=row["name"],
                            active_b=row["active_b"], is_moe=bool(row["moe"]),
                            max_context_tokens=int(row["context_k"]) * 1000, gpu_count=n)
        c = optimal_concurrency(
            params_b=row["params_b"], active_b=row["active_b"], quant="Q8",
            vram_gb=g["vram_gb"] * n, bandwidth_gbps=g["bandwidth_gbps"] * n * k,
            hw_type=g["hw_type"], ctx_tokens=sc.SERVING_CTX, kv_bytes_tok=vb.kv_bytes_tok,
            weights_gib=vb.weights_gib, fp16_tflops=tflops_for_gpu(preset) * n,
            slo="datacenter", gpu_count=n, moe=bool(row["moe"]))
        if c is None or c.per_session_tps < 35:
            continue
        assert c.total_tps / n <= best.tps_per_gpu * (1 + 1e-9)
    # 70B at FP8 (~70 GiB of weights) cannot hold an 8k sequence on ONE 80 GB H100
    assert best.n_gpus >= 2


def test_a_model_too_big_for_eight_gpus_is_not_served():
    huge = {"name": "synthetic 2.4T", "params_b": 2400.0, "active_b": 100.0,
            "context_k": 128, "moe": True}
    assert sc.serving_estimate(huge, "NVIDIA H100 SXM", 1.0) is None
    short = {"name": "synthetic 4k", "params_b": 7.0, "active_b": 7.0,
             "context_k": 4, "moe": False}
    assert sc.serving_estimate(short, "NVIDIA H100 SXM", 1.0) is None, \
        "a 4k-context model cannot run the 8k/1k workload at all"


def test_k_is_monotone_and_util_is_inverse(local):
    row = _row(local, "MiniMax-M3")
    prev = 0.0
    for k in (0.2, 0.4, 0.8, 1.6):
        e = sc.serving_estimate(row, "NVIDIA B200 SXM", k)
        assert e is not None
        assert e.tps_per_gpu >= prev * (1 - 1e-9)
        prev = e.tps_per_gpu
    products = {round(sc.cost_per_m(1.73, 900.0, u / 100) * u, 9) for u in (10, 35, 60, 100)}
    assert len(products) == 1


# ── Calibration ──────────────────────────────────────────────────────────────

def test_calibration_recovers_a_planted_k(local):
    names = [n for n in ("Llama 3.3 Instruct 70B", "MiniMax-M3", "Qwen3.5 397B A17B (Reasoning)")
             if (local["name"] == n).any()]
    if len(names) < 2:
        pytest.skip("not enough calibration models in the catalogue")
    preset = "NVIDIA B200 SXM"
    rows = {n: local[local["name"] == n].iloc[0] for n in names}

    def est(name, k):
        return sc.serving_estimate(rows[name], preset, k)
    pairs = [(n, est(n, 0.5).tps_per_gpu) for n in names]
    c = sc._fit_k(preset, pairs, estimate=est)
    step = (sc.K_MAX / sc.K_MIN) ** (1 / (sc.K_GRID_N - 1))
    assert 0.5 / step <= c.k <= 0.5 * step
    assert c.gsd < 1.05
    assert c.n_models == len(names) and not c.fallback


def test_calibration_ignores_non_like_for_like_rows(sample):
    """fp4, MTP, disaggregated and >8-GPU rows never enter the fit, and an
    agentic-traces or sub-35 tok/s row never counts at all; any of the former
    may still be the measured point."""
    lfl = sc.measured_best(sample, "MiniMax-M3", "b200", calibration=True)
    assert lfl["precision"] == "fp8" and lfl["spec_method"] == "none"
    assert lfl["deployment"] == "single-node" and lfl["gpu_count"] <= 8
    assert lfl["tps"] == pytest.approx(579.25)
    anyc = sc.measured_best(sample, "MiniMax-M3", "b200")
    assert anyc["tps"] == pytest.approx(1099.85)
    assert anyc["deployment"] == "disagg" and anyc["gpu_count"] == 16
    assert anyc["run_url"].startswith(sc.IX_RUNS_PREFIX)
    # Llama on b200: the 1047 tok/s fp4 row is at 34.0 tok/s/user, below 35
    best = sc.measured_best(sample, "Llama-3.3-70B-Instruct-FP8", "b200")
    assert best["tps"] == pytest.approx(802.522)
    assert sc.measured_best(sample, "Llama-3.3-70B-Instruct-FP8", "mi355x") is None


def test_the_derived_slice_keeps_both_maxima(tmp_path, sample):
    full = tmp_path / "points.csv"
    sample.to_csv(full, index=False)
    import io
    slim = pd.read_csv(io.StringIO(sc.derived_points_csv(full)), dtype={"run_id": str})
    assert set(slim.columns) == set(sc._POINT_COLUMNS)
    for model, hw in (("MiniMax-M3", "b200"), ("Llama-3.3-70B-Instruct-FP8", "b200"),
                      ("Llama-3.3-70B-Instruct-FP8", "h100")):
        for cal in (False, True):
            a = sc.measured_best(sample, model, hw, calibration=cal)
            b = sc.measured_best(slim, model, hw, calibration=cal)
            assert (a is None) == (b is None)
            if a:
                assert a["tps"] == b["tps"] and a["point_id"] == b["point_id"]
    assert len(slim) < len(sample)


def test_unfitted_presets_get_flagged_vendor_median(monkeypatch):
    """With only NVIDIA points, every AMD preset falls back to k = 1 (no AMD
    fit at all) and an NVIDIA preset with no points takes the NVIDIA median."""
    fake = {"NVIDIA H100 SXM": sc.Calib(0.4, 3, 1.2, False, False, "a"),
            "NVIDIA B200 SXM": sc.Calib(0.6, 3, 1.3, False, False, "b")}
    monkeypatch.setattr(sc, "_fit_k", lambda preset, pairs, estimate=None: fake.get(preset))
    monkeypatch.setattr(sc, "_calibration_pairs", lambda points, hw: [])
    import io
    df = pd.read_csv(io.StringIO(sc.calibration_csv())).set_index("preset")
    h200 = df.loc["NVIDIA H200 SXM"]
    assert bool(h200["fallback"]) and h200["k"] == pytest.approx(0.5)
    mi = df.loc["AMD Instinct MI300X"]
    assert bool(mi["fallback"]) and mi["k"] == 1.0
    assert not bool(df.loc["NVIDIA H100 SXM"]["fallback"])


def test_the_shipped_calibration_is_fitted_on_real_runs():
    cal = sc.fit_calibration()
    assert set(cal) == set(sc.IX_HARDWARE_TO_PRESET.values())
    fitted = [c for c in cal.values() if not c.fallback]
    if not fitted:
        pytest.skip("no InferenceX archive in this checkout")
    for c in fitted:
        assert sc.K_MIN <= c.k <= sc.K_MAX
        assert c.n_models >= 1 and c.gsd >= 1.0
        # the residual band the caption promises (read modelled points as ±2x)
        assert c.gsd < 2.5


# ── Mapping, rates, sources ───────────────────────────────────────────────────

def test_mapping_keys_point_at_real_presets_and_db_keys():
    for preset in sc.IX_HARDWARE_TO_PRESET.values():
        assert preset in GPU_BY_NAME
    for base, m in sc.IX_MODEL_MAP.items():
        assert m["identity"] in ("same", "sibling") and m["ix_name"] and m["db_keys"]
    archive = ROOT / sc.IX_POINTS_FILE
    if not archive.exists():
        pytest.skip("no InferenceX archive in this checkout")
    names = set(pd.read_csv(archive, usecols=["model"])["model"])
    for base, m in sc.IX_MODEL_MAP.items():
        assert m["ix_name"] in names or base in sc.IX_KNOWN_ABSENT, base


def test_fallback_rates_are_sourced():
    src = inspect.getsource(sc)
    assert sc.IX_OPTIONS_URL in src and "2026-10-27" in src
    assert set(sc._IX_RATES_FALLBACK) == set(sc.IX_HARDWARE_TO_PRESET)
    for own, rent in sc._IX_RATES_FALLBACK.values():
        assert own > 0 and rent > 0
    rates = sc.gpu_rates()
    if (ROOT / sc.IX_RATES_FILE).exists():
        for r in rates.values():
            assert r["source"].endswith("(archived mark)")
        archived = pd.read_csv(ROOT / sc.IX_RATES_FILE).set_index("hw_key")
        assert rates["NVIDIA H100 SXM"]["own"] == float(
            archived.loc["h100", "cost_own_usd_per_chip_hr"])


def test_the_default_utilisation_is_inferencex_and_cited():
    assert sc.DEFAULT_UTILISATION_PCT == 60
    assert sc.IX_OPENAPI_URL in inspect.getsource(sc)
    assert sc.coerce_util(None) == 60 and sc.coerce_util(5) == sc.UTIL_MIN_PCT
    assert sc.coerce_util(500) == sc.UTIL_MAX_PCT and sc.coerce_util("x") == 60


# ── The frame: like units, no fake zeros ─────────────────────────────────────

def _hosted():
    from data.ingest import get_models
    return get_models()


def test_markups_use_like_units():
    hosted = _hosted()
    frame = sc.serving_frame(hosted=hosted)
    if frame.empty:
        pytest.skip("no open hosted models")
    assert (frame["markup_out"] == frame["price_out"] / frame["usd_per_m_out"]).all()
    mix = (8 * frame["price_in"] + frame["price_out"]) / frame["usd_per_m_out"]
    assert frame["markup_mix"].values == pytest.approx(mix.values, rel=1e-12)
    # the blended `price` column is never read
    blanked = hosted.copy()
    blanked["price"] = float("nan")
    pd.testing.assert_frame_equal(sc.serving_frame(hosted=blanked), frame)


def test_missing_price_is_excluded_not_zero():
    hosted = _hosted().copy()
    opened = hosted[hosted["open_weights"].eq(True)].index[:3]
    hosted.loc[opened[0], "price_out"] = float("nan")
    hosted.loc[opened[1], "price_out"] = 0.0
    frame = sc.serving_frame(hosted=hosted)
    gone = set(hosted.loc[opened[:2], "model"])
    assert not gone & set(frame["model"])
    assert (frame["usd_per_m_out"] > 0).all()
    assert (frame["price_out"] > 0).all()


def test_only_open_models_with_a_spec_sheet_are_priced():
    hosted = _hosted()
    frame = sc.serving_frame(hosted=hosted)
    by_model = hosted.set_index("model")
    for m in frame["model"]:
        assert bool(by_model.loc[m, "open_weights"]) if not isinstance(
            by_model.loc[m, "open_weights"], pd.Series) else by_model.loc[m, "open_weights"].any()
    assert frame["base_model"].is_unique


def test_a_specific_gpu_prices_on_that_gpu_and_cheapest_is_the_minimum():
    hosted = _hosted()
    cheapest = sc.serving_frame("cheapest", "own", 60, hosted).set_index("base_model")
    for preset in sc.IX_HARDWARE_TO_PRESET.values():
        one = sc.serving_frame(preset, "own", 60, hosted)
        assert (one["gpu"] == preset).all()
        for _, r in one.iterrows():
            assert cheapest.loc[r["base_model"], "usd_per_m_out"] <= r["usd_per_m_out"] + 1e-12


def test_measured_points_carry_a_run_link_and_siblings_say_so():
    frame = sc.serving_frame(hosted=_hosted())
    meas = frame[frame["evidence"] == "measured"]
    if meas.empty:
        pytest.skip("no measured point in this checkout")
    for _, r in meas.iterrows():
        assert r["run_url"] == "" or r["run_url"].startswith(sc.IX_RUNS_PREFIX)
        assert r["ix_model"] and r["identity"] in ("same", "sibling")
        assert r["interactivity_tok_s_user"] >= 35
    mod = frame[frame["evidence"] == "modelled"]
    assert (mod["run_url"] == "").all()


def test_rent_and_lower_use_cost_more():
    hosted = _hosted()
    own = sc.serving_frame("NVIDIA H100 SXM", "own", 60, hosted).set_index("base_model")
    rent = sc.serving_frame("NVIDIA H100 SXM", "rent", 60, hosted).set_index("base_model")
    half = sc.serving_frame("NVIDIA H100 SXM", "own", 30, hosted).set_index("base_model")
    common = own.index.intersection(rent.index)
    assert len(common)
    assert (rent.loc[common, "usd_per_m_out"] > own.loc[common, "usd_per_m_out"]).all()
    assert half.loc[common, "usd_per_m_out"].values == pytest.approx(
        2 * own.loc[common, "usd_per_m_out"].values)


def test_axis_range_frames_prices_and_the_default_view():
    """Every API price and every default-view cost is inside; other control
    settings are fitted by econ_serve.fit_range, which only grows it."""
    from components.charts.econ_serve import fit_range
    lo, hi = sc.axis_range(_hosted())
    assert lo < hi
    d, _m = sc.serving_data(None, None, None, hosted=_hosted())
    vals = pd.concat([d["usd_per_m_out"], d["price_out"]]).map(math.log10)
    assert (vals >= lo).all() and (vals <= hi).all()
    for preset in sc.IX_HARDWARE_TO_PRESET.values():
        for basis in sc.BASES:
            for u in (sc.UTIL_MIN_PCT, 60, sc.UTIL_MAX_PCT):
                f = sc.serving_frame(preset, basis, u, _hosted())
                flo, fhi = fit_range((lo, hi), f)
                assert flo <= lo and fhi >= hi
                vals = pd.concat([f["usd_per_m_out"], f["price_out"]])
                assert (vals.map(math.log10) >= flo).all() and (vals.map(math.log10) <= fhi).all()


def test_local_readout_prices_the_tab_throughput():
    assert sc.local_usd_per_m(2.0, 1, 1000) == pytest.approx(sc.cost_per_m(2.0, 1000))
    assert sc.local_usd_per_m(2.0, 4, 1000) == pytest.approx(4 * sc.cost_per_m(2.0, 1000))
    for bad in (None, float("nan"), 0, -1):
        assert math.isnan(sc.local_usd_per_m(bad, 1, 1000))
    assert math.isnan(sc.local_usd_per_m(2.0, 1, 0))


def test_read_derived_prefers_the_shipped_slice(tmp_path, monkeypatch):
    from data import econ_common
    monkeypatch.setattr(econ_common, "ROOT", tmp_path)
    (tmp_path / "data" / "derived").mkdir(parents=True)
    (tmp_path / "data" / "derived" / "x.csv").write_text("a,b\n1,2\n")
    df = econ_common.read_derived("data/derived/x.csv", lambda: "a,b\n9,9\n")
    assert df.to_dict("records") == [{"a": 1, "b": 2}]
    df = econ_common.read_derived("data/derived/y.csv", lambda: "a,b\n9,9\n")
    assert df.to_dict("records") == [{"a": 9, "b": 9}]
    assert econ_common.read_derived("data/derived/z.csv", lambda: "").empty

    def boom():
        raise OSError("no archive")
    assert econ_common.read_derived("data/derived/z.csv", boom).empty
    assert set(econ_common.STATUS_VALUES) == {"published", "reported", "derived",
                                              "estimated", "unavailable"}
