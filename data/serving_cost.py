"""Economics › Serve: what it costs to serve an open-weights model on datacenter
GPUs, and how that compares with its API price. Pyodide-safe (stdlib + pandas;
no network, nothing from data/econ_archive.py, which is bot-only).

THE FORMULA (units checked against InferenceX's own cost metric below)

    $ / 1M output tokens = R ($/GPU-hr) × 1e6 / (T (output tok/s per GPU) × 3,600 × u)

  R  SemiAnalysis InferenceX's published owning (h) or renting (r) rate for the
     GPU, $/chip-hr (data/carried/econ/inferencex_rates.csv, archived by the
     hourly bot; latest row per hw_key is the current mark). InferenceX says
     these reflect "large-hyperscaler-volume ownership and rental pricing"
     (IX_ABOUT_URL): the cheapest an operator pays, not the market rate.
  T  output tokens per second per GPU at InferenceX's 8k-in / 1k-out workload,
     at a 35 tok/s per-user floor (InferenceX's calculator target). MEASURED
     where InferenceX ran the model on that GPU (best configuration of any
     kind: FP8/FP4, MTP, disaggregated); there the floor is on InferenceX's
     MEDIAN per-user interactivity (its fixed-sequence views publish the
     median), so up to half the users of a measured point decode slower.
     MODELLED everywhere else, by this dashboard's Run Local roofline in a
     tensor-parallel FP8-class mode, with a per-GPU bandwidth correction k
     fitted to the like-for-like InferenceX runs; the roofline gives every
     session the same rate, so there EVERY session is at >= 35 tok/s.
  u  utilisation: the share of paid GPU-hours spent serving at that throughput.
     Default 60%, InferenceX's own profit-estimator default.

T is OUTPUT throughput per GPU of the whole deployment, so the prefill GPU-time
that 8k input tokens cost is charged to the output tokens (as InferenceX's
costhOutput does). The output-only markup, price_out / cost, is therefore a
LOWER bound on the margin; the like-for-like figure is the same-mix markup,
(8 × price_in + price_out) / cost: the API bill for 1M output tokens that
arrive with 8M uncached input tokens, the workload the GPU cost assumes.

Verified on 2026-10-27 against InferenceX views/inference
(https://inferencex.semianalysis.com/api/v1/views/inference, Llama-3.3-70B-
Instruct-FP8, 8k/1k, B200 TRT-LLM FP8 TP1, conc 128, run 18924487526):
outputTputPerGpu 1043.4813203964407 at h = $1.73 gives costhOutput
0.4605310571088923 = 1.73 × 1e6 / (1043.48 × 3600). Pinned in the tests.
"""
from __future__ import annotations

import functools
import html as _html
import math
import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from data.econ_common import ROOT, read_derived
from data.local_models import (
    GPU_BY_NAME, SLO_FLOORS_TPS, get_local_df, optimal_concurrency, tflops_for_gpu,
    vram_breakdown,
)

# ── Workload and service level ───────────────────────────────────────────────
SERVING_SLO = "datacenter"            # SLO_FLOORS_TPS["datacenter"] = 35 tok/s/user
SERVING_ISL = 8192                    # InferenceX fixed-sequence 8k/1k
SERVING_OSL = 1024
SERVING_CTX = SERVING_ISL + SERVING_OSL   # KV sized for a whole sequence: 9,216
SERVING_SEQUENCE = "8k/1k"
# FP8-class weights (Q8, 8.5 bpw). The KV cache stays FP16, vLLM's "auto" KV
# dtype for an FP8 checkpoint (data/local_models.DEFAULT_KV_QUANT).
SERVING_QUANT = "Q8"
GPU_COUNTS = (1, 2, 4, 8)             # one node; InferenceX's like-for-like set
MIN_INTERACTIVITY = SLO_FLOORS_TPS[SERVING_SLO]
# The modelled scan's session cap per GPU count: the largest concurrency
# InferenceX itself swept at 8k/1k on one node with that many GPUs, so the
# roofline never extrapolates past the measured evidence. Run Local's display
# bound (data/local_models.MAX_REPORTED_CONCURRENCY = 256) is NOT a serving
# limit and used to set the cost of 33 of 58 modelled points. Read from
# data/carried/econ/inferencex_points.csv (sequence 8k/1k, single-node rows:
# max concurrency 1,024 on 1, 2 and 4 GPUs, 4,096 on 8), archive as of
# 2026-10-27; source https://inferencex.semianalysis.com/api/v1/views/inference
# (status: published). A test re-derives it from the archive.
SERVING_MAX_SESSIONS: dict[int, int] = {1: 1024, 2: 1024, 4: 1024, 8: 4096}

# ── Controls ──────────────────────────────────────────────────────────────────
# InferenceX profit estimator: "utilization ... default 60",
# https://inferencex.semianalysis.com/api/openapi.json, fetched 2026-10-23.
DEFAULT_UTILISATION_PCT = 60
UTIL_MIN_PCT, UTIL_MAX_PCT, UTIL_STEP = 10, 100, 5
DEFAULT_SERVE_GPU = "cheapest"
DEFAULT_SERVE_BASIS = "own"
DEFAULT_LOCAL_RATE_BASIS = "rent"
BASES = ("own", "rent")
BASIS_LABELS = {"own": "Own", "rent": "Rent"}
# InferenceX's about page: its cost marks reflect "large-hyperscaler-volume
# ownership and rental pricing" (https://inferencex.semianalysis.com/about,
# read 2026-10-27). Most providers pay more than either.
BASIS_TITLES = {
    "own": "InferenceX's large-hyperscaler-volume OWNING $/GPU-hr: the cheapest "
           "cost any operator has",
    "rent": "InferenceX's large-hyperscaler-volume RENTING $/GPU-hr",
}
BASIS_TITLE = ("SemiAnalysis InferenceX's $/GPU-hr marks, which reflect "
               "large-hyperscaler-volume ownership and rental pricing "
               "(inferencex.semianalysis.com/about). Most providers pay more.")

IX_OPTIONS_URL = "https://inferencex.semianalysis.com/api/v1/views/options?format=json"
IX_OPENAPI_URL = "https://inferencex.semianalysis.com/api/openapi.json"
IX_SITE_URL = "https://inferencex.semianalysis.com/"
IX_ABOUT_URL = "https://inferencex.semianalysis.com/about"
IX_RUNS_PREFIX = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/"
AA_URL = "https://artificialanalysis.ai/leaderboards/models"

IX_RATES_FILE = "data/carried/econ/inferencex_rates.csv"
IX_POINTS_FILE = "data/carried/econ/inferencex_points.csv"
DERIVED_POINTS = "data/derived/ix_serve_points.csv"
DERIVED_CALIBRATION = "data/derived/ix_serve_calibration.csv"

# InferenceX hardware key -> the Run Local preset with the same part. TDPs in
# inferencex_rates.csv (700/700/1000/1200/750/1000/1400 W) match these SXM/OAM
# parts. GB200/GB300/VR200 NVL72 racks, RTX PRO 6000, TPU7x and Jalapeño have
# no Run Local preset and are not used.
IX_HARDWARE_TO_PRESET: dict[str, str] = {
    "h100":   "NVIDIA H100 SXM",
    "h200":   "NVIDIA H200 SXM",
    "b200":   "NVIDIA B200 SXM",
    "b300":   "NVIDIA B300 SXM",
    "mi300x": "AMD Instinct MI300X",
    "mi325x": "AMD Instinct MI325X",
    "mi355x": "AMD Instinct MI355X",
}
PRESET_TO_IX_HARDWARE = {v: k for k, v in IX_HARDWARE_TO_PRESET.items()}

# Used ONLY when the archived rates file is absent (a checkout before the bot's
# first InferenceX fetch). InferenceX options endpoint, hardware[].costPerHour
# {h, r}, $/chip-hr, IX_OPTIONS_URL, fetched 2026-10-27 (status: published).
_IX_RATES_FALLBACK: dict[str, tuple[float, float]] = {
    "h100": (1.17, 2.00), "h200": (1.22, 2.90), "b200": (1.73, 3.70),
    "b300": (2.26, 4.25), "mi300x": (0.95, 1.30), "mi325x": (1.10, 1.60),
    "mi355x": (1.50, 2.90),
}
_IX_RATES_FALLBACK_DATE = "2026-10-27"

# Artificial Analysis base model name -> the InferenceX model it was measured
# as. Hand-kept from the InferenceX options endpoint (models[].dbKeys and
# display names, IX_OPTIONS_URL, fetched 2026-10-27). `ix_name` is the display
# name in the points archive's `model` column.
#   same     the same checkpoint
#   sibling  a sibling checkpoint of the same architecture, labelled as such on
#            every surface (never used to fit the calibration)
IX_MODEL_MAP: dict[str, dict] = {
    "Llama 3.3 Instruct 70B": {"ix_name": "Llama-3.3-70B-Instruct-FP8",
                               "db_keys": ["llama70b"], "identity": "same"},
    "gpt-oss-120b":           {"ix_name": "gpt-oss-120b",
                               "db_keys": ["gptoss120b"], "identity": "same"},
    "Qwen3.5 397B A17B":      {"ix_name": "Qwen-3.5-397B-A17B",
                               "db_keys": ["qwen3.5"], "identity": "same"},
    "MiniMax-M3":             {"ix_name": "MiniMax-M3",
                               "db_keys": ["minimaxm3"], "identity": "same"},
    "GLM-5.3":                {"ix_name": "GLM-5.3",
                               "db_keys": ["glm5.3"], "identity": "same"},
    # InferenceX's DeepSeek-V4-Pro is the 2026-04-24 release; AA lists the
    # 0813 checkpoint of the same architecture.
    "DeepSeek V4 Pro 0813":   {"ix_name": "DeepSeek-V4-Pro",
                               "db_keys": ["dsv4"], "identity": "sibling"},
    "Kimi K2.7 Code":         {"ix_name": "Kimi-K2.5",
                               "db_keys": ["kimik2.5", "kimik2.6"], "identity": "sibling"},
}
# Mapped models the archive holds no 8k/1k rows for (2026-10-27): InferenceX
# lists GLM-5.3 at 8k/1k on MI355X only, and the bot's views call has not
# returned it. They stay "modelled" until it does.
IX_KNOWN_ABSENT = frozenset({"GLM-5.3"})

# DeepSeek's own disclosure, quoted in the caption as a sanity anchor. CC0-1.0,
# https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/
# day_6_one_more_thing_deepseekV3R1_inference_system_overview.md, read
# 2026-10-27 (status: published). Window 2025-02-27 12:00 to 02-28 12:00 UTC+8:
# $2 per H800-hour assumed, ~14.8k output tok/s per 8xH800 node in decode,
# total daily cost $87,072 (prefill and decode nodes, average 226.75 nodes in
# use), 168B output tokens that day, R1 output $2.19/M, 545% theoretical
# cost-profit margin. Two figures follow from it (deepseek_anchor()):
#   decode only  $16/node-hr / (14.8k x 3,600) ~ $0.30/M: decode GPUs alone,
#                before prefill, at full use; NOT the chart's basis.
#   whole day    $87,072 / 168,000 M ~ $0.52/M: every node, prefill charged to
#                output tokens, at the day's real utilisation, which is the
#                chart's basis (y = all GPU time / output tokens).
DEEPSEEK_ANCHOR = {
    "source_url": ("https://github.com/deepseek-ai/open-infra-index/blob/main/"
                   "202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_"
                   "inference_system_overview.md"),
    "as_of": "2025-02-28", "read_on": "2026-10-27", "status": "published",
    "usd_per_node_hr": 16.0, "decode_tok_s_per_node": 14800.0,
    "total_daily_cost_usd": 87072.0, "output_tokens_day": 168e9,
    "r1_output_price": 2.19, "cost_profit_margin_pct": 545,
}


def deepseek_anchor() -> dict:
    """The two $/1M-output figures DEEPSEEK_ANCHOR implies (status: derived):
    decode-only and whole-day, each with its ratio to R1's output price."""
    a = DEEPSEEK_ANCHOR
    decode = a["usd_per_node_hr"] * 1e6 / (a["decode_tok_s_per_node"] * 3600.0)
    day = a["total_daily_cost_usd"] / (a["output_tokens_day"] / 1e6)
    return {"decode_usd_per_m": decode, "decode_ratio": a["r1_output_price"] / decode,
            "day_usd_per_m": day, "day_ratio": a["r1_output_price"] / day}


# ── Small pure helpers ────────────────────────────────────────────────────────

def cost_per_m(rate, tps_per_gpu, util_frac=1.0) -> float:
    """$ per 1M output tokens. NaN (never 0 or inf) when it cannot be priced."""
    try:
        rate, tps, u = float(rate), float(tps_per_gpu), float(util_frac)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(rate) and math.isfinite(tps) and math.isfinite(u)):
        return float("nan")
    if rate < 0 or tps <= 0 or u <= 0:
        return float("nan")
    return rate * 1e6 / (tps * 3600.0 * u)


_RUN_URL_RE = re.compile(r"^https://github\.com/SemiAnalysisAI/InferenceX/actions/runs/\d+/?$")


def run_url(run_id) -> str:
    """The GitHub Actions run behind an InferenceX point, or "" (digits only)."""
    s = str(run_id).strip() if run_id is not None else ""
    if s.endswith(".0"):
        s = s[:-2]
    return f"{IX_RUNS_PREFIX}{s}" if s.isdigit() else ""


def safe_run_url(url) -> str:
    """Only an https InferenceX run link survives; anything else is ""."""
    s = str(url or "").strip()
    return s if _RUN_URL_RE.match(s) else ""


def _base_name(name: str) -> str:
    from components.charts.constants import base_model_name
    return base_model_name(str(name))


def coerce_basis(basis) -> str:
    return basis if basis in BASES else DEFAULT_SERVE_BASIS


def coerce_util(util_pct) -> float:
    from static_helpers import coerce_number
    u = coerce_number(util_pct, default=DEFAULT_UTILISATION_PCT, minimum=UTIL_MIN_PCT)
    return min(u, float(UTIL_MAX_PCT))


def coerce_gpu(gpu) -> str:
    return gpu if gpu in PRESET_TO_IX_HARDWARE else DEFAULT_SERVE_GPU


# ── GPU rates (the single registry: F extends it for presets IX does not price) ─

@functools.lru_cache(maxsize=1)
def _rates_table() -> tuple:
    path = ROOT / IX_RATES_FILE
    rows = {}
    if path.exists():
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df = df.rename(columns={"valid_from": "observed_on"})
            for _, r in df.sort_values("observed_on", kind="mergesort").iterrows():
                try:
                    own = float(r["cost_own_usd_per_chip_hr"])
                    rent = float(r["cost_rent_usd_per_chip_hr"])
                except (TypeError, ValueError):
                    continue
                rows[r["hw_key"]] = (own, rent, r["observed_on"], r.get("label", ""))
        except (ValueError, OSError, KeyError, pd.errors.EmptyDataError):
            rows = {}
    return tuple(sorted(rows.items()))


def gpu_rates() -> dict[str, dict]:
    """{Run Local preset name: {own, rent, source, date, hw_key, label}}.

    Latest archived InferenceX mark per hardware key; the sourced fallback
    constant only when the archive is absent. Presets InferenceX does not price
    are absent (F adds owned-TCO / rental medians for them here)."""
    table = dict(_rates_table())
    out = {}
    for hw, preset in IX_HARDWARE_TO_PRESET.items():
        if hw in table:
            own, rent, date, label = table[hw]
            src = "SemiAnalysis InferenceX (archived mark)"
        elif hw in _IX_RATES_FALLBACK:
            own, rent = _IX_RATES_FALLBACK[hw]
            date, label = _IX_RATES_FALLBACK_DATE, hw.upper()
            src = "SemiAnalysis InferenceX (fallback constant)"
        else:
            continue
        out[preset] = {"own": own, "rent": rent, "source": src, "date": date,
                       "hw_key": hw, "label": label or hw.upper(),
                       "source_url": IX_OPTIONS_URL}
    return out


def preset_label(preset: str) -> str:
    return str(preset).replace("NVIDIA ", "").replace("AMD Instinct ", "")


def priced_cards_phrase() -> str:
    """Which cards a $/GPU-hr preset fills, for every tooltip that says so:
    only the parts InferenceX prices, never "any datacenter card"."""
    names = [preset_label(p) for p in IX_HARDWARE_TO_PRESET.values()]
    return f"one of the {len(names)} datacenter cards InferenceX prices ({', '.join(names)})"


# ── InferenceX measured points ────────────────────────────────────────────────

_POINT_COLUMNS = ["point_id", "model", "hw_key", "framework", "spec_method", "deployment",
                  "precision", "concurrency", "gpu_count", "tensor_parallel",
                  "expert_parallel", "interactivity_tok_s_user",
                  "output_tput_tok_s_chip", "run_date", "run_id"]


def _like_for_like(df: pd.DataFrame) -> pd.Series:
    """FP8, no speculative decoding, not disaggregated, one node of <= 8 GPUs:
    the configurations the tensor-parallel FP8 roofline describes."""
    gpus = pd.to_numeric(df["gpu_count"], errors="coerce")
    return ((df["precision"].astype(str) == "fp8")
            & (df["spec_method"].astype(str) == "none")
            & (~df["deployment"].astype(str).isin(["disagg", "multi-node"]))
            & (gpus <= 8))


def derived_points_csv(path: Path | None = None) -> str:
    """The slim slice of A2's points archive the browser needs, as CSV text.

    8k/1k rows whose MEDIAN per-user interactivity (InferenceX's fixed-sequence
    views publish the median) is >= 35 tok/s, on the hardware Serve prices
    (IX_HARDWARE_TO_PRESET) and for the models it maps (IX_MODEL_MAP); per
    (model, hardware) only the best row overall and the best like-for-like row
    are kept, because measured_best only ever takes those maxima. ~50 rows
    instead of 4,300, well inside the derived-file budget."""
    path = Path(path) if path is not None else ROOT / IX_POINTS_FILE
    if not path.exists():
        return ""
    df = pd.read_csv(path, dtype={"run_id": str})
    need = set(_POINT_COLUMNS) | {"sequence", "x_field"}
    if not need <= set(df.columns):
        return ""
    df = df[(df["sequence"] == SERVING_SEQUENCE) & (df["x_field"] == "median_intvty")]
    df = df[pd.to_numeric(df["interactivity_tok_s_user"], errors="coerce")
            >= MIN_INTERACTIVITY]
    df = df[pd.to_numeric(df["output_tput_tok_s_chip"], errors="coerce") > 0]
    # Only what serving_cost ever reads: other chips and models were half the
    # slice, and the archive it comes from is append-only.
    df = df[df["hw_key"].isin(IX_HARDWARE_TO_PRESET)
            & df["model"].isin({m["ix_name"] for m in IX_MODEL_MAP.values()})]
    if df.empty:
        return ""
    df = df[_POINT_COLUMNS].copy()
    df["output_tput_tok_s_chip"] = pd.to_numeric(df["output_tput_tok_s_chip"])
    keep = []
    for _, g in df.groupby(["model", "hw_key"], sort=True):
        g = g.sort_values(["output_tput_tok_s_chip", "point_id"], ascending=[False, True],
                          kind="mergesort")
        keep.append(g.head(1))
        lfl = g[_like_for_like(g)]
        if not lfl.empty:
            keep.append(lfl.head(1))
    out = pd.concat(keep).drop_duplicates("point_id").sort_values(["model", "hw_key",
                                                                   "point_id"])
    return out.to_csv(index=False)


@functools.lru_cache(maxsize=1)
def _points_cached() -> pd.DataFrame:
    df = read_derived(DERIVED_POINTS, derived_points_csv)
    if df.empty:
        return pd.DataFrame(columns=_POINT_COLUMNS)
    return df


def load_ix_points() -> pd.DataFrame:
    """The measured points (A2's schema, slimmed). Empty when the archive is absent."""
    return _points_cached().copy()


def measured_best(points: pd.DataFrame, ix_name: str, hw_key: str,
                  min_intvty: float = MIN_INTERACTIVITY,
                  calibration: bool = False) -> dict | None:
    """The highest output tok/s per GPU InferenceX measured for this model on
    this hardware with its MEDIAN user at >= min_intvty tok/s, or None.

    calibration=False: any configuration counts (FP4, MTP, disaggregated,
    multi-node), because a measurement of what the hardware does is the point.
    calibration=True: like-for-like rows only (_like_for_like)."""
    if points is None or points.empty:
        return None
    p = points[(points["model"] == ix_name) & (points["hw_key"] == hw_key)]
    p = p[pd.to_numeric(p["interactivity_tok_s_user"], errors="coerce") >= min_intvty]
    if calibration:
        p = p[_like_for_like(p)]
    p = p[pd.to_numeric(p["output_tput_tok_s_chip"], errors="coerce") > 0]
    if p.empty:
        return None
    r = p.sort_values(["output_tput_tok_s_chip", "point_id"], ascending=[False, True],
                      kind="mergesort").iloc[0]

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    return {
        "tps": float(r["output_tput_tok_s_chip"]),
        "precision": str(r["precision"]), "framework": str(r["framework"]),
        "spec_method": str(r["spec_method"]), "deployment": str(r["deployment"]),
        "gpu_count": _int(r["gpu_count"]), "tp": _int(r["tensor_parallel"]),
        "ep": _int(r["expert_parallel"]), "concurrency": _int(r["concurrency"]),
        "interactivity": float(r["interactivity_tok_s_user"]),
        "run_date": str(r["run_date"]), "point_id": _int(r["point_id"]),
        "run_url": safe_run_url(run_url(r["run_id"])),
    }


# ── Modelled throughput: Run Local physics in a tensor-parallel mode ─────────

class ServingEstimate(NamedTuple):
    n_gpus: int
    tps_per_gpu: float
    sessions: int
    per_session_tps: float
    binding: str
    kv_source: str


def serving_estimate(model_row, preset: str, k: float = 1.0) -> ServingEstimate | None:
    """Best output tok/s per GPU for `model_row` on n x `preset`, n in GPU_COUNTS.

    Tensor parallel: pooled VRAM = n x vram, bandwidth = n x bw x k, FLOPs =
    n x FP16 TFLOPS, through the same optimal_concurrency -> _step_seconds ->
    decode_roofline chain as Run Local, at the 8k/1k sequence length and a
    35 tok/s/user floor. Run Local's own layer-split mode is untouched. The
    session scan stops at SERVING_MAX_SESSIONS[n], InferenceX's own sweep,
    not Run Local's 256 display bound; binding "session_cap" when it binds.

    n is the count with the most tokens per GPU among those that fit the FP8
    weights plus at least one 9,216-token sequence and keep every session at
    >= 35 tok/s: the cost-optimal deployment, which is what InferenceX's best
    measured configuration is too. None when no n <= 8 serves it."""
    g = GPU_BY_NAME.get(preset)
    if g is None:
        return None
    get = (model_row.get if hasattr(model_row, "get") else model_row.__getitem__)
    try:
        params_b, active_b = float(get("params_b")), float(get("active_b"))
        ctx_k = int(float(get("context_k")))
    except (TypeError, ValueError):
        return None
    if not (params_b > 0 and active_b > 0):
        return None
    name, moe = str(get("name")), bool(get("moe"))
    tf = tflops_for_gpu(preset)
    best = None
    for n in GPU_COUNTS:
        vb = vram_breakdown(params_b, SERVING_QUANT, SERVING_CTX, name=name,
                            active_b=active_b, is_moe=moe,
                            max_context_tokens=ctx_k * 1000, gpu_count=n)
        if vb.ctx_used < SERVING_CTX:
            return None                    # cannot hold an 8k/1k sequence at all
        c = optimal_concurrency(
            params_b=params_b, active_b=active_b, quant=SERVING_QUANT,
            vram_gb=g["vram_gb"] * n, bandwidth_gbps=g["bandwidth_gbps"] * n * k,
            hw_type=g["hw_type"], ctx_tokens=SERVING_CTX, kv_bytes_tok=vb.kv_bytes_tok,
            weights_gib=vb.weights_gib, fp16_tflops=(tf * n if tf else None),
            slo=SERVING_SLO, gpu_count=n, moe=moe,
            max_sessions=SERVING_MAX_SESSIONS[n])
        if c is None or c.per_session_tps < MIN_INTERACTIVITY:
            continue
        est = ServingEstimate(n, c.total_tps / n, c.sessions, c.per_session_tps,
                              c.binding, vb.kv_source)
        if best is None or est.tps_per_gpu > best.tps_per_gpu * (1 + 1e-9):
            best = est
    return best


@functools.lru_cache(maxsize=1)
def _local_catalogue() -> pd.DataFrame:
    df = get_local_df(include_pending=False)
    return df if not df.empty else pd.DataFrame(columns=["name"])


def _local_row(name: str):
    df = _local_catalogue()
    hit = df[df["name"] == name]
    return None if hit.empty else hit.iloc[0]


@functools.lru_cache(maxsize=4096)
def _estimate_for(name: str, preset: str, k: float) -> ServingEstimate | None:
    row = _local_row(name)
    return None if row is None else serving_estimate(row, preset, k)


# ── Calibration: one bandwidth-efficiency k per GPU preset ───────────────────

K_MIN, K_MAX, K_GRID_N = 0.05, 2.0, 41
K_GRID = tuple(K_MIN * (K_MAX / K_MIN) ** (i / (K_GRID_N - 1)) for i in range(K_GRID_N))


class Calib(NamedTuple):
    k: float
    n_models: int
    gsd: float           # geometric SD of measured/modelled at k (exp RMS log residual)
    at_bound: bool
    fallback: bool
    models: str          # "; "-joined AA base names the fit used


def _calibration_pairs(points: pd.DataFrame, hw_key: str) -> list[tuple[str, float]]:
    """(local model name, like-for-like measured tok/s/GPU) for one hardware."""
    pairs = []
    for base, m in IX_MODEL_MAP.items():
        if m["identity"] != "same":
            continue
        best = measured_best(points, m["ix_name"], hw_key, calibration=True)
        if not best:
            continue
        cat = _local_catalogue()
        hits = cat[cat["name"].map(_base_name) == base] if not cat.empty else cat
        if hits.empty:
            continue
        name = sorted(hits["name"].astype(str))[0]
        pairs.append((name, best["tps"]))
    return pairs


def _fit_k(preset: str, pairs: list[tuple[str, float]],
           estimate=None) -> Calib | None:
    """k minimising the mean squared log residual over the pairs that can be
    served on `preset` at all (at K_MAX). A k at which any of them cannot be
    served is not a candidate. 41-point log grid, then ternary search within
    one grid step either side of the best grid point."""
    estimate = estimate or (lambda name, k: _estimate_for(name, preset, k))
    usable = [(n, t) for n, t in pairs if estimate(n, K_MAX) is not None]
    if not usable:
        return None

    def loss(k):
        s = 0.0
        for name, t_meas in usable:
            e = estimate(name, k)
            if e is None or e.tps_per_gpu <= 0:
                return float("inf")
            s += (math.log(t_meas) - math.log(e.tps_per_gpu)) ** 2
        return s / len(usable)

    losses = [loss(k) for k in K_GRID]
    i = min(range(len(K_GRID)), key=lambda j: (losses[j], j))
    if not math.isfinite(losses[i]):
        return None
    best_k, best_l = K_GRID[i], losses[i]
    lo = math.log(K_GRID[max(i - 1, 0)])
    hi = math.log(K_GRID[min(i + 1, len(K_GRID) - 1)])
    for _ in range(24):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        l1, l2 = loss(math.exp(m1)), loss(math.exp(m2))
        if l1 <= l2:
            hi = m2
        else:
            lo = m1
    k_t = math.exp((lo + hi) / 2)
    l_t = loss(k_t)
    if l_t < best_l:
        best_k, best_l = k_t, l_t
    at_bound = best_k <= K_GRID[0] * 1.0001 or best_k >= K_GRID[-1] * 0.9999
    names = "; ".join(sorted(_base_name(n) for n, _ in usable))
    return Calib(best_k, len(usable), math.exp(math.sqrt(best_l)), at_bound, False, names)


def calibration_csv() -> str:
    """Per-preset calibration as CSV text (DERIVED_DATA; read via read_derived).
    Presets with no usable pair take the median k of fitted presets of the
    same vendor (hw_type), flagged fallback; with none at all, k = 1."""
    points = _points_cached()
    fitted: dict[str, Calib] = {}
    for hw, preset in IX_HARDWARE_TO_PRESET.items():
        c = _fit_k(preset, _calibration_pairs(points, hw))
        if c is not None:
            fitted[preset] = c
    rows = []
    for hw, preset in IX_HARDWARE_TO_PRESET.items():
        c = fitted.get(preset)
        if c is None:
            vendor = GPU_BY_NAME[preset]["hw_type"]
            ks = sorted(v.k for p, v in fitted.items()
                        if GPU_BY_NAME[p]["hw_type"] == vendor)
            if ks:
                mid = len(ks) // 2
                k = ks[mid] if len(ks) % 2 else (ks[mid - 1] + ks[mid]) / 2
            else:
                k = 1.0
            c = Calib(k, 0, float("nan"), False, True, "")
        rows.append({"preset": preset, "hw_key": hw, "k": c.k, "n_models": c.n_models,
                     "gsd": c.gsd, "at_bound": int(c.at_bound),
                     "fallback": int(c.fallback), "models": c.models})
    return pd.DataFrame(rows).to_csv(index=False)


@functools.lru_cache(maxsize=1)
def fit_calibration() -> dict[str, Calib]:
    df = read_derived(DERIVED_CALIBRATION, calibration_csv)
    out = {}
    for _, r in df.iterrows():
        gsd = float(r["gsd"]) if pd.notna(r["gsd"]) else float("nan")
        models = "" if pd.isna(r.get("models")) else str(r["models"])
        out[str(r["preset"])] = Calib(float(r["k"]), int(r["n_models"]), gsd,
                                      bool(int(r["at_bound"])), bool(int(r["fallback"])),
                                      models)
    return out


def reset_cache() -> None:
    """Forget every cached frame (tests that swap the data files)."""
    for f in (_rates_table, _points_cached, _local_catalogue, _estimate_for,
              fit_calibration, _throughput_table):
        f.cache_clear()


# ── The open models that are both hosted (priced) and in the local catalogue ──

def open_hosted_models(hosted: pd.DataFrame) -> pd.DataFrame:
    """One row per open-weights model FAMILY with an AA output price and a spec
    sheet: the hosted catalogue's open rows (data/open_weights flag) joined by
    exact name to the open-weights catalogue, collapsed by base name to the
    best-scoring variant (variants share weights, so they share a serving cost).
    """
    cols = ["base_model", "model", "provider", "price_in", "price_out", "quality",
            "local_name"]
    if hosted is None or hosted.empty:
        return pd.DataFrame(columns=cols)
    h = hosted
    if "open_weights" not in h.columns:
        from data.open_weights import annotate_open_weights
        h = annotate_open_weights(h)
    po = pd.to_numeric(h["price_out"], errors="coerce")
    pi = pd.to_numeric(h["price_in"], errors="coerce")
    h = h[h["open_weights"].eq(True) & (po > 0) & pi.notna() & (pi >= 0)]
    local = set(_local_catalogue()["name"].astype(str))
    h = h[h["model"].astype(str).isin(local)].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    h["base_model"] = h["model"].map(_base_name)
    h = h.sort_values(["base_model", "quality", "model"], ascending=[True, False, True],
                      kind="mergesort").drop_duplicates("base_model")
    h["local_name"] = h["model"].astype(str)
    h["price_in"] = pd.to_numeric(h["price_in"], errors="coerce")
    h["price_out"] = pd.to_numeric(h["price_out"], errors="coerce")
    return h[cols].reset_index(drop=True)


# ── Per (model, GPU) throughput: measured where InferenceX ran it ────────────

@functools.lru_cache(maxsize=8)
def _throughput_table(names: tuple[str, ...]) -> pd.DataFrame:
    """For each local model name x preset: T and where it came from. Cached on
    the tuple of names, so the browser does the physics once per session."""
    points = _points_cached()
    calib = fit_calibration()
    rows = []
    for name in names:
        base = _base_name(name)
        mapping = IX_MODEL_MAP.get(base)
        for hw, preset in IX_HARDWARE_TO_PRESET.items():
            c = calib.get(preset)
            k = c.k if c else 1.0
            meas = measured_best(points, mapping["ix_name"], hw) if mapping else None
            est = _estimate_for(name, preset, k)
            row = {"local_name": name, "preset": preset, "hw_key": hw,
                   "calib_k": k, "calib_models": c.n_models if c else 0,
                   "calib_gsd": c.gsd if c else float("nan"),
                   "calib_fallback": bool(c.fallback) if c else True,
                   "ix_model": "", "identity": "", "precision": "", "framework": "",
                   "spec_method": "", "deployment": "", "interactivity_tok_s_user": float("nan"),
                   "run_date": "", "run_url": "", "sessions": float("nan"),
                   "per_session_tps": float("nan"), "binding": ""}
            if meas:
                row.update({"evidence": "measured", "tok_s_per_gpu": meas["tps"],
                            "gpu_count": meas["gpu_count"], "ix_model": mapping["ix_name"],
                            "identity": mapping["identity"],
                            "precision": meas["precision"], "framework": meas["framework"],
                            "spec_method": meas["spec_method"],
                            "deployment": meas["deployment"],
                            "interactivity_tok_s_user": meas["interactivity"],
                            "run_date": meas["run_date"], "run_url": meas["run_url"],
                            "tp": meas["tp"], "ep": meas["ep"]})
            elif est is not None:
                row.update({"evidence": "modelled", "tok_s_per_gpu": est.tps_per_gpu,
                            "gpu_count": est.n_gpus, "sessions": est.sessions,
                            "per_session_tps": est.per_session_tps,
                            "binding": est.binding, "precision": "fp8 (modelled)",
                            "tp": est.n_gpus, "ep": None})
            else:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def _table_for(models: pd.DataFrame) -> pd.DataFrame:
    names = tuple(sorted(models["local_name"].astype(str))) if not models.empty else ()
    return _throughput_table(names)


EXPORT_COLUMNS = [
    "model", "base_model", "provider", "price_in", "price_out", "gpu", "gpu_count",
    "basis", "usd_per_gpu_hr", "rate_source", "rate_observed_on", "utilisation_pct",
    "tok_s_per_gpu", "usd_per_m_out", "markup_out", "markup_mix", "evidence",
    "identity", "ix_model", "precision", "framework", "spec_method", "deployment",
    "interactivity_tok_s_user", "run_date", "run_url", "calib_k", "calib_models",
    "calib_gsd", "calib_fallback", "sessions", "per_session_tps", "binding",
]


def serving_data(gpu=None, basis=None, util_pct=None,
                 hosted: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """(frame, meta) for the Serve chart and its ↓CSV.

    One row per open family that can be priced on `gpu` ("cheapest" = the
    preset with the lowest cost for that family, measured where it exists).
    Markups use like units only: output price over output cost, and the 8:1
    same-mix bill over the same cost. The site's blended `price` is never read.
    """
    from data.workload_mix import effective_price, resolve_mix

    gpu, basis, util = coerce_gpu(gpu), coerce_basis(basis), coerce_util(util_pct)
    if hosted is None:
        from data.ingest import get_models
        hosted = get_models()
    models = open_hosted_models(hosted)
    rates = gpu_rates()
    meta = {"gpu": gpu, "basis": basis, "util_pct": util, "n_families": len(models),
            "rates": rates, "calibration": fit_calibration()}
    table = _table_for(models)
    if models.empty or table.empty:
        meta["n_unserved"] = len(models)
        return pd.DataFrame(columns=EXPORT_COLUMNS), meta
    t = table.copy()
    t["usd_per_gpu_hr"] = t["preset"].map(lambda p: rates.get(p, {}).get(basis, float("nan")))
    t["rate_source"] = t["preset"].map(lambda p: rates.get(p, {}).get("source", ""))
    t["rate_observed_on"] = t["preset"].map(lambda p: rates.get(p, {}).get("date", ""))
    t["usd_per_m_out"] = [cost_per_m(r, tps, util / 100.0)
                          for r, tps in zip(t["usd_per_gpu_hr"], t["tok_s_per_gpu"])]
    t = t[t["usd_per_m_out"].notna()]
    if gpu != DEFAULT_SERVE_GPU:
        t = t[t["preset"] == gpu]
    else:
        t = (t.sort_values(["local_name", "usd_per_m_out", "preset"], kind="mergesort")
              .drop_duplicates("local_name"))
    df = models.merge(t, on="local_name", how="inner")
    meta["n_unserved"] = len(models) - len(df)
    if df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), meta
    df["gpu"] = df["preset"]
    df["basis"] = basis
    df["utilisation_pct"] = util
    df["markup_out"] = df["price_out"] / df["usd_per_m_out"]
    same_mix = resolve_mix(["custom", 8, 0])          # 8 in : 1 out, no cache
    eff, _ = effective_price(df["price_in"], df["price_out"], None, same_mix)
    # eff is per blended token; the bill for 8M in + 1M out is 9 x that.
    df["markup_mix"] = (eff.values * (same_mix.w_in + same_mix.w_out)) / df["usd_per_m_out"]
    df = df.sort_values(["price_out", "base_model"], kind="mergesort").reset_index(drop=True)
    for c in EXPORT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[EXPORT_COLUMNS + [c for c in ("tp", "ep") if c in df.columns]], meta


def serving_frame(gpu=None, basis=None, util_pct=None,
                  hosted: pd.DataFrame | None = None) -> pd.DataFrame:
    """The export frame (serving_data without the meta)."""
    return serving_data(gpu, basis, util_pct, hosted)[0][EXPORT_COLUMNS]


def axis_range(hosted: pd.DataFrame | None = None) -> tuple[float, float]:
    """log10 (lo, hi) shared by both axes, from the data alone, never the
    controls: the cheapest cost any preset reaches (own rate, 100% use), the
    dearest (rent rate, UTIL_MIN use) and the price range, padded."""
    if hosted is None:
        from data.ingest import get_models
        hosted = get_models()
    models = open_hosted_models(hosted)
    vals = [v for v in pd.to_numeric(models.get("price_out", pd.Series(dtype=float)),
                                     errors="coerce") if v and v > 0]
    table = _table_for(models)
    rates = gpu_rates()
    for _, r in table.iterrows():
        rr = rates.get(r["preset"])
        if not rr:
            continue
        for v in (cost_per_m(rr["own"], r["tok_s_per_gpu"], 1.0),
                  cost_per_m(rr["rent"], r["tok_s_per_gpu"], UTIL_MIN_PCT / 100.0)):
            if math.isfinite(v) and v > 0:
                vals.append(v)
    if not vals:
        return (-2.0, 1.0)
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    return (math.floor((lo - 0.1) * 10) / 10, math.ceil((hi + 0.1) * 10) / 10)


def serve_options() -> dict:
    """Everything the Serve controls need, so no shell declares a default."""
    rates = gpu_rates()
    gpus = [{"value": DEFAULT_SERVE_GPU, "label": "Cheapest per model",
             "title": "For each model, the GPU with the lowest cost at these settings"}]
    for hw, preset in IX_HARDWARE_TO_PRESET.items():
        r = rates.get(preset)
        if not r:
            continue
        gpus.append({
            "value": preset,
            "label": f"{preset_label(preset)} · own ${r['own']:.2f} · rent ${r['rent']:.2f}",
            "title": (f"{r['source']}, observed {r['date']}: ${r['own']:.2f}/GPU-hr "
                      f"owned, ${r['rent']:.2f}/GPU-hr rented, at large-hyperscaler "
                      f"volume ({IX_ABOUT_URL})"),
        })
    return {
        "gpus": gpus, "default_gpu": DEFAULT_SERVE_GPU,
        "bases": [{"value": b, "label": BASIS_LABELS[b], "title": BASIS_TITLES[b]}
                  for b in BASES],
        "basis_title": BASIS_TITLE,
        "default_basis": DEFAULT_SERVE_BASIS,
        "default_util_pct": DEFAULT_UTILISATION_PCT,
        "util_min": UTIL_MIN_PCT, "util_max": UTIL_MAX_PCT, "util_step": UTIL_STEP,
        "util_title": ("Share of paid GPU-hours spent serving at the plotted throughput, "
                       "owned or rented. Default 60%: InferenceX's profit-estimator "
                       f"default ({IX_OPENAPI_URL}, 2026-10-23)."),
        "rate_source": "SemiAnalysis InferenceX", "rate_source_url": IX_OPTIONS_URL,
        "default_local_basis": DEFAULT_LOCAL_RATE_BASIS,
    }


def local_usd_per_m(rate, gpu_count, total_tps) -> float:
    """Run Local / Agent Stack readout: the tab's own layer-split aggregate
    throughput at its 10 tok/s floor, at full use. NaN when unpriceable."""
    try:
        n = max(int(gpu_count), 1)
    except (TypeError, ValueError):
        n = 1
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(r) or r <= 0:
        return float("nan")
    return cost_per_m(r * n, total_tps, 1.0)


def esc(v) -> str:
    return _html.escape(str(v), quote=True)
