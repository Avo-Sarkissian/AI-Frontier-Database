"""Economics › Compute: what a GPU-hour costs to own, and what it rents for.
Pyodide-safe (stdlib + pandas; no network, nothing from data/econ_archive.py,
which is bot-only).

OWNED COST PER GPU-HOUR (all figures per GPU)

    capital = capex × CRF(r, n) / 8,760
    power   = IT kW × PUE × $/kWh                 (kW × $/kWh = $/h)
    colo    = colo $/kW-month × IT kW × 12 / 8,760
    owned   = capital + power + colo              per GPU-hour owned, busy or idle
    owned_u = owned / u                           per BUSY GPU-hour (the stack)

    CRF(r, n) = r (1 + r)^n / ((1 + r)^n − 1), and 1/n when r = 0

  r  cost of capital per year: the latest NY Fed SOFR (archived by the bot)
     plus a filing-sourced spread, or a disclosed fixed rate (PRESETS_FILE)
  n  useful life in years, from the same filings
  u  utilisation, the Economics tab's shared USE slider (serving_cost's
     default, InferenceX's profit-estimator default): the share of PAID
     GPU-hours that do work. The rule is Serve's, owned or rented alike: cost
     per busy hour = cost per paid hour ÷ u. The stack shows owned ÷ u. The
     ladder and the weekly chart stay per paid GPU-hour, busy or idle, the
     basis every rental, list price and InferenceX mark is published on, so the
     owned mark sits there at full use (owned, not owned_u); dividing every
     mark by u would move them all alike and leave every ratio unchanged.
  8,760 = 365 × 24 hours, the year convention (no leap day).

The three components over their sum are shown as shares on the chart: they
must sum to 1, and they are the units check (SemiAnalysis: colocation and
electricity "typically make up less than 20%" of TCO).

Every default is a row of a hand-curated CSV in data/curated/ carrying its own
source_url, as_of and status (econ_common.STATUS_VALUES); nothing numeric is
declared here except unit conventions and input bounds. A GPU with no public
all-in system price (B300, MI300X, MI325X, MI355X) is never priced from an
invented number: it reads "enter CAPEX" until someone types one. Capex
defaults come from sources on different price bases (CAPEX_BASES, the
`basis` column of GPU_FILE), and every bar names its basis: they are not
like-for-like across GPUs.

THE RENTAL LADDER for the selected GPU, cheapest to dearest, each rung its own
series and never merged with another: owned (this page) · InferenceX's own and
rent marks · SemiAnalysis's 1-year contract index (two quoted points) ·
GetDeploying weekly marketplace medians (latest week) · neocloud list prices ·
AWS capacity blocks · hyperscaler on-demand list (Azure, AWS; fetched weekly
by the bot). The weekly series chart shows GetDeploying's four billing series
for the GPU, each its own line.
"""
from __future__ import annotations

import csv
import functools
import io
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.econ_common import ROOT, read_derived

# ── Files ─────────────────────────────────────────────────────────────────────
GPU_FILE = "data/curated/gpu_tco.csv"
ASSUMPTIONS_FILE = "data/curated/tco_assumptions.csv"
PRESETS_FILE = "data/curated/financing_presets.csv"
PRICE_POINTS_FILE = "data/curated/gpu_price_points.csv"
CURATED_FILES = (GPU_FILE, ASSUMPTIONS_FILE, PRESETS_FILE, PRICE_POINTS_FILE)
HYPERSCALER_FILE = "data/raw/econ/hyperscaler_gpu_prices.csv"
# Bot archives (A2), read only at build time to make the derived slices.
GD_ARCHIVE = "data/carried/econ/gpu_prices_weekly.csv"
SOFR_ARCHIVE = "data/carried/econ/sofr.csv"
EPOCH_HARDWARE = "data/raw/epoch/ml_hardware.csv"
IX_RATES_FILE = "data/carried/econ/inferencex_rates.csv"
# Build-time slices (build_static.DERIVED_DATA, read through read_derived).
DERIVED_WEEKLY = "data/derived/gpu_rental_weekly.csv"
DERIVED_SOFR = "data/derived/sofr_latest.csv"
DERIVED_EPOCH = "data/derived/epoch_gpu_specs.csv"

# ── Source links (credits on every surface) ──────────────────────────────────
GD_DATASET_URL = "https://getdeploying.com/dataset/gpu-prices"
GD_CREDIT = "GetDeploying (CC BY 4.0)"
EPOCH_URL = "https://epoch.ai/data/machine-learning-hardware"
EPOCH_CREDIT = "Epoch AI, 'Data on Machine Learning Hardware' (CC BY 4.0)"
SOFR_URL = "https://www.newyorkfed.org/markets/reference-rates/sofr"
IX_CHIP_URL = "https://inferencex.semianalysis.com/chips/{slug}"
# The owned rung is this page's own arithmetic; its provenance is the inputs.
CURATED_URL = ("https://github.com/Avo-Sarkissian/AI-Frontier-Database/blob/main/"
               "data/curated/gpu_tco.csv")

HOURS_PER_YEAR = 365 * 24

# ── GPUs ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuSpec:
    key: str             # gpu_key in the curated files
    label: str
    form: str            # hgx8 | nvl72 | oam8
    gd_slug: str         # GetDeploying gpu_slug
    epoch_name: str      # Epoch AI "Hardware name", exactly
    ix_key: str          # InferenceX hardware key (inferencex_rates.hw_key)
    ix_chip: str         # InferenceX chip page slug


# Slugs and names checked against the archives on 2026-10-27: every gd_slug
# has weekly rows in gpu_prices_weekly.csv, every epoch_name is a row of the
# Epoch mirror, every ix_key a row of inferencex_rates.csv, and every chip page
# answers 200 (inferencex.semianalysis.com/chips/<slug>).
GPUS: tuple[GpuSpec, ...] = (
    GpuSpec("h100", "H100 SXM", "hgx8", "nvidia-h100", "NVIDIA H100 SXM5 80GB", "h100", "h100"),
    GpuSpec("h200", "H200 SXM", "hgx8", "nvidia-h200", "NVIDIA H200 SXM", "h200", "h200"),
    GpuSpec("b200", "B200", "hgx8", "nvidia-b200", "NVIDIA B200", "b200", "b200"),
    GpuSpec("b300", "B300", "hgx8", "nvidia-b300", "NVIDIA B300 (Blackwell Ultra)", "b300", "b300"),
    GpuSpec("gb200", "GB200 NVL72", "nvl72", "nvidia-gb200", "NVIDIA GB200", "gb200",
            "gb200-nvl72"),
    GpuSpec("gb300", "GB300 NVL72", "nvl72", "nvidia-gb300", "NVIDIA GB300 (Blackwell Ultra)",
            "gb300", "gb300-nvl72"),
    GpuSpec("mi300x", "MI300X", "oam8", "amd-mi300x", "AMD Instinct MI300X", "mi300x", "mi300x"),
    GpuSpec("mi325x", "MI325X", "oam8", "amd-mi325x", "AMD Instinct MI325X", "mi325x", "mi325x"),
    GpuSpec("mi355x", "MI355X", "oam8", "amd-mi355x", "AMD Instinct MI355X", "mi355x", "mi355x"),
)
GPU_BY_KEY = {g.key: g for g in GPUS}
GPU_KEYS = tuple(GPU_BY_KEY)


def gd_label(spec: GpuSpec) -> str:
    """GetDeploying's 'nvidia-h100' slug pools SXM, PCIe and NVL listings, so
    its series is never called 'H100 SXM'."""
    if spec.key == "h100":
        return "H100 (all form factors, per GetDeploying)"
    return f"{spec.label.split(' ')[0]} (GetDeploying '{spec.gd_slug}')"


DEFAULT_GPU = "h100"
DEFAULT_PRESET = "coreweave"
REFERENCE_KW_GPU = "h100"       # whose all-in kW / Epoch TDP ratio fills AMD/B300

# Input bounds: what the boxes accept (a typed value outside is clamped, and
# the chart subtitle shows the value used). Not defaults.
LIFE_RANGE = (1.0, 15.0)                   # years
RATE_PCT_RANGE = (0.0, 50.0)               # % per year
CAPEX_RANGE = (1_000.0, 5_000_000.0)       # USD per GPU
KW_RANGE = (0.05, 20.0)                    # kW per GPU
PUE_RANGE = (1.0, 3.0)
KWH_RANGE = (0.0, 1.0)                     # USD per kWh
COLO_RANGE = (0.0, 2_000.0)                # USD per kW-month

# Weekly series: GetDeploying billing type (+ reservation months) -> key.
# Every other billing type (CUSTOM, ON_DEMAND_MINUTE, RESERVATION_12_MONTH,
# other terms) is dropped, never folded in.
SERIES = ("on_demand", "spot", "reserved_12m", "reserved_36m")
SERIES_LABELS = {"on_demand": "on-demand", "spot": "spot",
                 "reserved_12m": "12-month reserved", "reserved_36m": "36-month reserved"}
# The spread ladder's point labels: the long forms ran two reserved terms into
# one 46-character label ("36-month reserved $2.97 · 12-month reserved $3.27")
# that read as one price. The legend and hovers keep SERIES_LABELS.
SERIES_SHORT = {"on_demand": "on-demand", "spot": "spot",
                "reserved_12m": "12-mo reserved", "reserved_36m": "36-mo reserved"}
_SERIES_OF = {("ON_DEMAND", ""): "on_demand", ("SPOT", ""): "spot",
              ("RESERVATION", "12"): "reserved_12m", ("RESERVATION", "36"): "reserved_36m"}
# median_usd is GetDeploying's median_price, a median over OFFERINGS (an
# offering is one provider's listing; a provider can list several): with two
# offerings it is their midpoint, and it differs from provider_median_price
# whenever offering_count != provider_count. Labels say "median of N offerings
# from M providers".
WEEKLY_COLUMNS = ["gpu_key", "date", "series", "median_usd", "provider_count",
                  "offering_count"]
# The shipped slice holds the last two years; the archive keeps every week.
SERIES_WEEKS = 104

# What a capex default's price includes (GPU_FILE `basis`). Not numbers:
# labels for the source each curated row already cites.
CAPEX_BASES: dict[str, tuple[str, str]] = {
    "hyperscaler_allin": ("hyperscaler volume, all-in",
                          "SemiAnalysis's all-in estimate for a typical hyperscaler (server plus "
                          "storage, networking and other items), 2025"),
    "street_scaled": ("street price ×250/190",
                      "a 2026 street or appraisal price for the bare 8-GPU server, scaled by "
                      "SemiAnalysis's H100 all-in over bare-server ratio (250/190); not "
                      "hyperscaler volume, so it reads dearer than the hyperscaler-volume bars"),
    "rack_excl_core": ("rack, excl. storage",
                       "a 2026 rack purchase-plus-deployment price that excludes cluster core "
                       "switching and storage, so it reads cheaper than an all-in figure"),
    "typed": ("typed", "typed on this page"),
}


LADDER_RUNGS: tuple[tuple[str, str], ...] = (
    ("owned", "Owned (this page)"),
    ("inferencex", "InferenceX TCO marks"),
    ("contract_index", "1-yr contract index"),
    ("marketplace", "Marketplace medians"),
    ("neocloud_list", "Neocloud list"),
    ("capacity_block", "AWS capacity block"),
    ("hyperscaler", "Hyperscaler on-demand"),
)
LADDER_COLUMNS = ["rung", "rung_label", "series_name", "label", "usd_per_gpu_hr", "period",
                  "as_of", "basis", "status", "source_url", "note"]


# ── Pure arithmetic ───────────────────────────────────────────────────────────

def crf(rate: float, years: float) -> float:
    """Capital-recovery factor per year. rate is a fraction (0.09 = 9%)."""
    if years is None or not years > 0:
        raise ValueError(f"useful life must be > 0 years, got {years!r}")
    if rate is None or not math.isfinite(rate) or rate < 0:
        raise ValueError(f"rate must be >= 0, got {rate!r}")
    if rate == 0:
        return 1.0 / years
    g = (1.0 + rate) ** years
    return rate * g / (g - 1.0)


def owned_breakdown(capex_usd: float, it_kw: float, pue: float, usd_per_kwh: float,
                    colo_usd_kw_month: float, rate: float, years: float,
                    util: float = 1.0) -> dict:
    """$/GPU-hr components at full use, the total at `util` (a fraction), and
    each component's share of the total."""
    if not 0 < util <= 1:
        raise ValueError(f"utilisation must be in (0, 1], got {util!r}")
    c = crf(rate, years)
    capital = capex_usd * c / HOURS_PER_YEAR
    power = it_kw * pue * usd_per_kwh
    colo = colo_usd_kw_month * it_kw * 12.0 / HOURS_PER_YEAR
    total = capital + power + colo
    shares = ({"capital": capital / total, "power": power / total, "colo": colo / total}
              if total > 0 else {"capital": float("nan"), "power": float("nan"),
                                 "colo": float("nan")})
    return {"crf": c, "capital": capital, "power": power, "colo": colo,
            "total_100": total, "total_at_util": total / util, "util": util,
            "shares": shares}


# ── Loading ───────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, ValueError, OSError):
        return pd.DataFrame()


def reset_cache() -> None:
    for f in (_read_csv, load_weekly, latest_sofr, epoch_specs, _ix_marks):
        getattr(f, "cache_clear", lambda: None)()     # a test may have patched one


def _num(v) -> float | None:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _prov(row: dict, value=None, typed: bool = False) -> dict:
    """One input with its provenance, the shape every surface reads."""
    return {"value": value, "unit": row.get("unit", ""), "status": row.get("status", ""),
            "source_url": row.get("source_url", ""), "as_of": row.get("as_of", ""),
            "note": row.get("note", ""), "typed": typed}


def _typed_prov(value, unit: str, raw=None) -> dict:
    """A typed box's value. `clamped` when the typed number was outside the
    box's bounds and `value` is the bound used; `raw` is what was typed."""
    clamped = raw is not None and raw != value
    return {"value": value, "unit": unit, "status": "typed", "source_url": "",
            "as_of": "", "typed": True, "clamped": clamped, "raw": raw,
            "note": ("typed on this page" if not clamped
                     else f"typed {raw:g} on this page, clamped to {value:g}")}


def load_gpu_defaults() -> dict[str, dict[str, dict]]:
    """{gpu_key: {param: row}} from GPU_FILE."""
    df = _read_csv(GPU_FILE)
    out: dict[str, dict[str, dict]] = {}
    for r in df.to_dict("records"):
        out.setdefault(r["gpu_key"], {})[r["param"]] = r
    return out


def load_assumptions() -> dict[str, dict]:
    df = _read_csv(ASSUMPTIONS_FILE)
    return {r["param"]: r for r in df.to_dict("records")}


def load_presets() -> list[dict]:
    """[{key, label, params: {param: row}}] in file order."""
    df = _read_csv(PRESETS_FILE)
    out: dict[str, dict] = {}
    for r in df.to_dict("records"):
        p = out.setdefault(r["preset_key"], {"key": r["preset_key"], "label": r["label"],
                                             "params": {}})
        p["params"][r["param"]] = r
    return list(out.values())


def load_price_points() -> pd.DataFrame:
    df = _read_csv(PRICE_POINTS_FILE)
    if df.empty:
        return df
    return df[df["display"] == "plot"].reset_index(drop=True)


def load_hyperscaler() -> pd.DataFrame:
    return _read_csv(HYPERSCALER_FILE)


# ── Derived slices (build time -> pydata.zip; computed locally) ───────────────

def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def slim_weekly(archive: pd.DataFrame) -> pd.DataFrame:
    """The four whitelisted series for the GPUs this view prices, last
    SERIES_WEEKS weeks, cells verbatim (no averaging, no merging). Raises on a
    renamed upstream column so a build says so loudly."""
    need = {"gpu_slug", "date", "billing_type", "reservation_months", "median_price",
            "provider_count", "offering_count"}
    missing = need - set(archive.columns)
    if missing:
        raise ValueError(f"GetDeploying archive lacks columns {sorted(missing)}")
    slug_key = {g.gd_slug: g.key for g in GPUS}
    a = archive[archive["gpu_slug"].isin(slug_key)].copy()
    rm = a["reservation_months"].astype(str).str.strip()
    rm = rm.where(rm == "", rm.map(lambda s: str(int(float(s))) if _num(s) is not None else s))
    a["series"] = [_SERIES_OF.get((b, m)) for b, m in zip(a["billing_type"], rm)]
    a = a[a["series"].notna()]
    if a.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    newest = max(date.fromisoformat(d[:10]) for d in a["date"])
    cutoff = (newest - timedelta(weeks=SERIES_WEEKS)).isoformat()
    a = a[a["date"] > cutoff]
    out = pd.DataFrame({
        "gpu_key": a["gpu_slug"].map(slug_key), "date": a["date"], "series": a["series"],
        "median_usd": a["median_price"], "provider_count": a["provider_count"],
        "offering_count": a["offering_count"],
    })
    order = {k: i for i, k in enumerate(GPU_KEYS)}
    sorder = {k: i for i, k in enumerate(SERIES)}
    out = out.assign(_g=out["gpu_key"].map(order), _s=out["series"].map(sorder))
    out = out.sort_values(["_g", "_s", "date"], kind="mergesort").drop(columns=["_g", "_s"])
    return out.reset_index(drop=True)


def slim_weekly_csv(path: Path | None = None) -> str:
    path = Path(path) if path is not None else ROOT / GD_ARCHIVE
    if not path.exists():
        return ""
    archive = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = slim_weekly(archive)
    return _csv_text(WEEKLY_COLUMNS, df[WEEKLY_COLUMNS].values.tolist())


def sofr_latest_csv(path: Path | None = None) -> str:
    """One row: the newest SOFR fixing in the bot's archive."""
    path = Path(path) if path is not None else ROOT / SOFR_ARCHIVE
    if not path.exists():
        return ""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["percent_rate"].map(_num).notna()]
    if df.empty:
        return ""
    last = df.sort_values("effective_date", kind="mergesort").iloc[-1]
    return _csv_text(["effective_date", "percent_rate"],
                     [[last["effective_date"], last["percent_rate"]]])


def epoch_specs_csv(path: Path | None = None) -> str:
    """The nine Epoch AI rows this view names: TDP and chip release price."""
    path = Path(path) if path is not None else ROOT / EPOCH_HARDWARE
    if not path.exists():
        return ""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = [g.epoch_name for g in GPUS]
    sub = df[df["Hardware name"].isin(names)]
    rows = []
    for n in names:
        hit = sub[sub["Hardware name"] == n]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append([n, r.get("TDP (W)", ""), r.get("Release price (USD)", ""),
                     str(r.get("Last modified", ""))[:10]])
    return _csv_text(["epoch_name", "tdp_w", "release_price_usd", "last_modified"], rows)


@functools.lru_cache(maxsize=1)
def load_weekly() -> pd.DataFrame:
    df = read_derived(DERIVED_WEEKLY, slim_weekly_csv)
    if df.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    df["date"] = df["date"].astype(str)
    df["median_usd"] = pd.to_numeric(df["median_usd"], errors="coerce")
    for col in ("provider_count", "offering_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else float("nan")
    return df


@functools.lru_cache(maxsize=1)
def latest_sofr() -> dict | None:
    """{"pct", "date", "source_url"} or None when the archive is absent."""
    df = read_derived(DERIVED_SOFR, sofr_latest_csv)
    if df.empty:
        return None
    pct = _num(df.iloc[0]["percent_rate"])
    if pct is None:
        return None
    return {"pct": pct, "date": str(df.iloc[0]["effective_date"]), "source_url": SOFR_URL}


@functools.lru_cache(maxsize=1)
def epoch_specs() -> dict[str, dict]:
    df = read_derived(DERIVED_EPOCH, epoch_specs_csv)
    out = {}
    for r in df.to_dict("records") if not df.empty else []:
        out[str(r["epoch_name"])] = {"tdp_w": _num(r.get("tdp_w")),
                                     "release_price_usd": _num(r.get("release_price_usd")),
                                     "last_modified": str(r.get("last_modified", ""))}
    return out


@functools.lru_cache(maxsize=1)
def _ix_marks() -> tuple:
    """Latest InferenceX (own, rent, observed_on) per hardware key."""
    df = _read_csv(IX_RATES_FILE)
    if df.empty:
        return ()
    df = df.rename(columns={"valid_from": "observed_on"})
    marks = {}
    for r in df.sort_values("observed_on", kind="mergesort").to_dict("records"):
        own, rent = _num(r.get("cost_own_usd_per_chip_hr")), _num(r.get("cost_rent_usd_per_chip_hr"))
        if own is not None and rent is not None:
            marks[r["hw_key"]] = (own, rent, r["observed_on"])
    return tuple(sorted(marks.items()))


# ── Controls ──────────────────────────────────────────────────────────────────

def coerce_gpu(gpu) -> str:
    return gpu if gpu in GPU_BY_KEY else DEFAULT_GPU


def coerce_preset(preset) -> str:
    keys = [p["key"] for p in load_presets()]
    return preset if preset in keys else DEFAULT_PRESET


def _clamp_typed(v, bounds: tuple[float, float]) -> float | None:
    """A typed box: None when empty or not a number, else clamped to bounds."""
    t = _typed(v, bounds)
    return None if t is None else t[0]


def _typed(v, bounds: tuple[float, float]) -> tuple[float, float] | None:
    """(clamped value, raw typed number), or None for an empty or non-numeric box."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    x = _num(v)
    if x is None:
        return None
    return min(max(x, bounds[0]), bounds[1]), x


def _util(util_pct) -> float:
    from data.serving_cost import coerce_util
    return coerce_util(util_pct)


# ── Resolution: every input with its provenance ──────────────────────────────

def _preset(key: str) -> dict:
    for p in load_presets():
        if p["key"] == key:
            return p
    return {"key": key, "label": key, "params": {}}


def resolve_financing(preset=None, life_years=None, rate_pct=None) -> dict:
    """Life and rate for a preset, typed values winning. A preset that discloses
    no life (Crusoe) or no rate (Nebius, SemiAnalysis, Burry) takes the default
    preset's, and the note says so. A SOFR-spread rate with no SOFR archive is
    unavailable (rate None), never guessed."""
    key = coerce_preset(preset)
    p, base = _preset(key), _preset(DEFAULT_PRESET)
    out = {"preset": key, "label": p["label"], "sofr": latest_sofr()}

    life_t = _typed(life_years, LIFE_RANGE)
    if life_t is not None:
        out["life"] = _typed_prov(life_t[0], "years", life_t[1])
    elif "life_years" in p["params"]:
        row = p["params"]["life_years"]
        out["life"] = _prov(row, _num(row["value"]))
    else:
        row = base["params"]["life_years"]
        out["life"] = {**_prov(row, _num(row["value"])),
                       "note": f"{p['label']} discloses no useful life; "
                               f"{base['label'].split(' ')[0]}'s is used. " + row["note"]}

    rate_t = _typed(rate_pct, RATE_PCT_RANGE)
    sofr = out["sofr"]
    if rate_t is not None:
        used, raw = rate_t
        clamped = used != raw
        out["rate"] = {**_typed_prov(used / 100.0, "per year"), "clamped": clamped,
                       "raw": raw / 100.0,
                       "note": ("typed on this page" if not clamped
                                else f"typed {raw:g}% on this page, clamped to {used:g}%"),
                       "text": f"{used:.2f}% typed" + (" (clamped)" if clamped else "")}
        return out
    params = p["params"] if ("fixed_pct" in p["params"] or "spread_pct" in p["params"]) \
        else base["params"]
    inherited = params is base["params"] and key != DEFAULT_PRESET
    if "fixed_pct" in params:
        row = params["fixed_pct"]
        r = _num(row["value"]) / 100.0
        rate = {**_prov(row, r), "text": f"{r * 100:.2f}% fixed"}
    else:
        row = params["spread_pct"]
        spread = _num(row["value"])
        if sofr is None:
            rate = {**_prov(row, None), "status": "unavailable",
                    "text": f"SOFR unavailable + {spread:.2f}%",
                    "note": "No SOFR in the archive: pick a fixed-rate preset or type a RATE. "
                            + row["note"]}
        else:
            r = (sofr["pct"] + spread) / 100.0
            rate = {**_prov(row, r), "status": "derived",
                    "text": (f"SOFR {sofr['pct']:.2f}% ({sofr['date']}) + {spread:.2f}% "
                             f"= {r * 100:.2f}%"),
                    "spread_pct": spread,
                    "note": (f"NY Fed SOFR {sofr['pct']:.2f}% on {sofr['date']} "
                             f"({SOFR_URL}) plus the spread. " + row["note"])}
    if inherited:
        rate["note"] = (f"{p['label']} discloses no borrowing rate; "
                        f"{base['label'].split(' ')[0]}'s is used. " + rate["note"])
    out["rate"] = rate
    return out


def resolve_facility(pue=None, usd_per_kwh=None, colo_usd_kw_month=None) -> dict:
    a = load_assumptions()
    out = {}
    for name, typed, bounds in (("pue", pue, PUE_RANGE), ("usd_per_kwh", usd_per_kwh, KWH_RANGE),
                                ("colo_usd_kw_month", colo_usd_kw_month, COLO_RANGE)):
        t = _typed(typed, bounds)
        row = a.get(name, {})
        out[name] = (_typed_prov(t[0], row.get("unit", ""), t[1]) if t is not None
                     else _prov(row, _num(row.get("value"))))
    return out


def _derived_kw(spec: GpuSpec, row: dict, defaults: dict) -> dict:
    """IT kW from Epoch TDP × the reference GPU's all-in kW / its Epoch TDP."""
    specs = epoch_specs()
    ref = GPU_BY_KEY[REFERENCE_KW_GPU]
    ref_kw = _num(defaults.get(ref.key, {}).get("it_kw", {}).get("value"))
    ref_tdp = (specs.get(ref.epoch_name) or {}).get("tdp_w")
    tdp = (specs.get(spec.epoch_name) or {}).get("tdp_w")
    if not (ref_kw and ref_tdp and tdp):
        return {**_prov(row, None), "status": "unavailable",
                "note": "Epoch AI TDP not available in this build. " + row.get("note", "")}
    kw = tdp * ref_kw / ref_tdp
    return {**_prov(row, kw),
            "note": (f"Epoch AI TDP {tdp:,.0f} W × ({ref_kw:g} kW / {ref_tdp:,.0f} W, the "
                     f"{ref.label} all-in-to-TDP ratio) = {kw:.3f} kW. " + row.get("note", ""))}


def resolve_gpu_inputs(gpu_key: str, capex_usd=None, it_kw=None) -> dict:
    spec = GPU_BY_KEY[gpu_key]
    defaults = load_gpu_defaults()
    rows = defaults.get(gpu_key, {})
    out = {}
    cap_t = _typed(capex_usd, CAPEX_RANGE)
    crow = rows.get("capex_usd", {})
    out["capex_usd"] = (_typed_prov(cap_t[0], "USD per GPU", cap_t[1]) if cap_t is not None
                        else _prov(crow, _num(crow.get("value"))))
    basis = "typed" if cap_t is not None else (crow.get("basis") or "")
    out["capex_usd"]["basis"] = basis if basis in CAPEX_BASES else ""
    kw_t = _typed(it_kw, KW_RANGE)
    krow = rows.get("it_kw", {})
    if kw_t is not None:
        out["it_kw"] = _typed_prov(kw_t[0], "kW per GPU", kw_t[1])
    elif krow.get("method") == "epoch_tdp_ratio":
        out["it_kw"] = _derived_kw(spec, krow, defaults)
    else:
        out["it_kw"] = _prov(krow, _num(krow.get("value")))
    return out


def compute(gpu_key=None, preset=None, util_pct=None, life_years=None, rate_pct=None,
            capex_usd=None, it_kw=None, pue=None, usd_per_kwh=None,
            colo_usd_kw_month=None) -> dict:
    """Everything the three charts and the export need, for one control state.

    Life, rate, PUE, $/kWh and colo apply to every GPU (one financing and one
    facility for all); a typed CAPEX or kW applies to the selected GPU only.
    The capex defaults are NOT on one price basis (CAPEX_BASES): each row's
    inputs carry capex_usd["basis"] and every surface names it."""
    gpu = coerce_gpu(gpu_key)
    util = _util(util_pct)
    fin = resolve_financing(preset, life_years, rate_pct)
    fac = resolve_facility(pue, usd_per_kwh, colo_usd_kw_month)
    rows = []
    for spec in GPUS:
        sel = spec.key == gpu
        inp = resolve_gpu_inputs(spec.key, capex_usd if sel else None, it_kw if sel else None)
        vals = (inp["capex_usd"]["value"], inp["it_kw"]["value"], fac["pue"]["value"],
                fac["usd_per_kwh"]["value"], fac["colo_usd_kw_month"]["value"],
                fin["rate"]["value"], fin["life"]["value"])
        bd = owned_breakdown(*vals, util=util / 100.0) if all(v is not None for v in vals) \
            else None
        missing = [n for n, v in zip(("CAPEX", "kW", "PUE", "$/kWh", "colo", "rate", "life"),
                                     vals) if v is None]
        rows.append({"spec": spec, "selected": sel, "inputs": inp, "breakdown": bd,
                     "missing": missing})
    return {"gpu": gpu, "util_pct": util, "financing": fin, "facility": fac, "rows": rows,
            "selected": next(r for r in rows if r["selected"])}


# ── Ladder ────────────────────────────────────────────────────────────────────

def _fmt_usd(v) -> str:
    return "—" if v is None or not math.isfinite(v) else f"${v:,.2f}"


def ladder_rows(result: dict) -> pd.DataFrame:
    """Tidy rows for the selected GPU's spread ladder, one per mark."""
    spec: GpuSpec = result["selected"]["spec"]
    bd = result["selected"]["breakdown"]
    rows: list[dict] = []
    if bd is not None:
        sofr = result["financing"].get("sofr") or {}
        rows.append({
            "rung": "owned", "series_name": "Owned cost (this page)",
            "label": "owned",
            "usd_per_gpu_hr": bd["total_100"], "period": sofr.get("date", ""),
            "as_of": sofr.get("date", ""),
            "basis": (f"capital + power + colo per GPU-hour owned, busy or idle, like every "
                      f"rental here ({_fmt_usd(bd['total_at_util'])} per busy hour at "
                      f"{result['util_pct']:g}% use)"),
            "status": "derived", "source_url": CURATED_URL,
            "note": "this page's arithmetic on the sourced inputs shown in the stack's hover"})
    marks = dict(_ix_marks()).get(spec.ix_key)
    if marks:
        own, rent, seen = marks
        url = IX_CHIP_URL.format(slug=spec.ix_chip)
        for lab, v, basis in (("IX own", own, "owning at large-hyperscaler volume"),
                              ("IX rent", rent, "renting at large-hyperscaler volume")):
            rows.append({"rung": "inferencex", "series_name": f"InferenceX {lab[3:]} mark",
                         "label": lab, "usd_per_gpu_hr": v, "period": seen, "as_of": seen,
                         "basis": f"SemiAnalysis AI Cloud TCO Model mark, {basis}",
                         "status": "published", "source_url": url,
                         "note": "current SemiAnalysis mark, observed on this date"})
    pts = load_price_points()
    if not pts.empty:
        for r in pts[pts["gpu_key"] == spec.key].to_dict("records"):
            per = r["period"]
            lab = r["provider"]
            if r["rung"] == "contract_index":
                y, m = per.split("-")[:2]
                lab = f"{date(int(y), int(m), 1):%b %Y}"
            rows.append({"rung": r["rung"], "series_name": r["series_name"], "label": lab,
                         "usd_per_gpu_hr": _num(r["value"]), "period": per,
                         "as_of": r["as_of"], "basis": r["basis"], "status": r["status"],
                         "source_url": r["source_url"], "note": r["note"]})
    wk = load_weekly()
    if not wk.empty:
        mine = wk[wk["gpu_key"] == spec.key]
        if not mine.empty:
            newest = mine["date"].max()
            for s in SERIES:
                hit = mine[(mine["series"] == s) & (mine["date"] == newest)]
                if hit.empty or pd.isna(hit.iloc[0]["median_usd"]):
                    continue
                h = hit.iloc[0]
                rows.append({"rung": "marketplace",
                             "series_name": f"GetDeploying {SERIES_LABELS[s]} median",
                             "label": SERIES_SHORT[s], "usd_per_gpu_hr": float(h["median_usd"]),
                             "period": newest, "as_of": newest,
                             "basis": (f"{offerings_phrase(h.get('offering_count'), h['provider_count'])}, "
                                       f"week of {newest} (newest week is provisional), "
                                       f"{gd_label(spec)}"),
                             "status": "published", "source_url": GD_DATASET_URL,
                             "note": "weekly medians archived by this site's bot; "
                                     + GD_CREDIT})
    hyp = load_hyperscaler()
    if not hyp.empty:
        for r in hyp[hyp["gpu_key"] == spec.key].to_dict("records"):
            cloud = "Azure" if r["cloud"] == "azure" else "AWS"
            sku = r["sku"].replace("Standard_", "")
            rows.append({"rung": "hyperscaler", "series_name": f"{cloud} on-demand list",
                         "label": cloud, "usd_per_gpu_hr": _num(r["price_per_gpu_hr"]),
                         "period": r["observed_on"], "as_of": r["observed_on"],
                         "basis": (f"{sku}, {r['region']}, ${_num(r['price_per_instance_hr']):,.2f}"
                                   f"/hr ÷ {r['gpus_per_instance']} GPUs, Linux on-demand list "
                                   f"(effective {r['effective_date']})"),
                         "status": "published", "source_url": r["source_url"],
                         "note": "fetched weekly by this site's bot; observed_on = first day "
                                 "seen at this price"})
    df = pd.DataFrame(rows, columns=[c for c in LADDER_COLUMNS if c != "rung_label"])
    labels = dict(LADDER_RUNGS)
    df.insert(1, "rung_label", df["rung"].map(labels))
    order = {k: i for i, (k, _) in enumerate(LADDER_RUNGS)}
    df = df[df["usd_per_gpu_hr"].notna()]
    df = df.assign(_o=df["rung"].map(order)).sort_values(["_o", "usd_per_gpu_hr", "label"],
                                                          kind="mergesort")
    return df.drop(columns="_o").reset_index(drop=True)


def offerings_phrase(offerings, providers) -> str:
    """'median of 109 offerings from 39 providers': GetDeploying's median_price
    is taken over offerings, not providers (see WEEKLY_COLUMNS)."""
    def n(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return int(x) if math.isfinite(x) else None
    o, p = n(offerings), n(providers)
    plural = lambda k, w: f"{k} {w}{'s' if k != 1 else ''}"
    if o is None:
        return "median of offerings" + (f" from {plural(p, 'provider')}" if p is not None else "")
    return f"median of {plural(o, 'offering')}" + (
        f" from {plural(p, 'provider')}" if p is not None else "")


def weekly_for(gpu_key: str) -> pd.DataFrame:
    wk = load_weekly()
    if wk.empty:
        return wk
    return wk[wk["gpu_key"] == gpu_key].reset_index(drop=True)


# ── Options (manifest) and export ────────────────────────────────────────────

def _title(p: dict, fmt: str) -> str:
    v = p.get("value")
    val = "—" if v is None else fmt.format(v)
    bits = [val, p.get("status", "")]
    if p.get("as_of"):
        bits.append(p["as_of"])
    src = f" · {p['source_url']}" if p.get("source_url") else ""
    note = f". {p['note']}" if p.get("note") else ""
    return " · ".join(b for b in bits if b) + src + note


def compute_options() -> dict:
    """Everything the Compute controls need, so neither shell declares a
    default: option lists, the default GPU and preset, and per GPU / preset the
    resolved default for each box (its placeholder) with its provenance (its
    tooltip). Capex placeholders are in $K, the box's unit."""
    fac = resolve_facility()
    sofr = latest_sofr()
    gpus, defaults = [], {}
    for spec in GPUS:
        inp = resolve_gpu_inputs(spec.key)
        cap, kw = inp["capex_usd"], inp["it_kw"]
        gpus.append({"value": spec.key, "label": spec.label,
                     "has_capex": cap["value"] is not None,
                     "title": (f"{spec.label}: all-in capex "
                               + (f"${cap['value'] / 1000:,.1f}K per GPU" if cap["value"]
                                  else "not public, type one") + f", {gd_label(spec)}")})
        defaults[spec.key] = {
            "capex_k": None if cap["value"] is None else round(cap["value"] / 1000.0, 3),
            "it_kw": None if kw["value"] is None else round(kw["value"], 4),
            "capex_title": ("CAPEX $K per GPU: " + _title(cap, "${:,.0f}")
                            + (f". Basis: {CAPEX_BASES[cap['basis']][1]}"
                               if cap.get("basis") else "")),
            "kw_title": "IT kW per GPU, all-in: " + _title(kw, "{:.3f} kW"),
        }
    presets, preset_defaults = [], {}
    for p in load_presets():
        fin = resolve_financing(p["key"])
        r = fin["rate"]
        presets.append({"value": p["key"], "label": p["label"],
                        "title": f"Life: {_title(fin['life'], '{:g} yr')} | Rate: {r['text']}"})
        preset_defaults[p["key"]] = {
            "life_years": fin["life"]["value"],
            "rate_pct": None if r["value"] is None else round(r["value"] * 100.0, 3),
            "life_title": "LIFE, years: " + _title(fin["life"], "{:g} yr"),
            "rate_title": "RATE, % per year: " + r["text"] + ". " + r.get("note", ""),
        }
    facility = {}
    for name, label, fmt in (("pue", "PUE", "{:g}"), ("usd_per_kwh", "$/kWh", "${:.4f}"),
                             ("colo_usd_kw_month", "COLO $/kW-month", "${:,.2f}")):
        facility[name] = {"value": fac[name]["value"],
                          "title": f"{label}: " + _title(fac[name], fmt)}
    return {"gpus": gpus, "default_gpu": DEFAULT_GPU, "presets": presets,
            "default_preset": DEFAULT_PRESET, "defaults": defaults,
            "preset_defaults": preset_defaults, "facility": facility,
            "sofr": sofr,
            "bounds": {"life": LIFE_RANGE, "rate_pct": RATE_PCT_RANGE,
                       "capex_k": (CAPEX_RANGE[0] / 1000, CAPEX_RANGE[1] / 1000),
                       "it_kw": KW_RANGE, "pue": PUE_RANGE, "usd_per_kwh": KWH_RANGE,
                       "colo": COLO_RANGE}}


EXPORT_INPUTS = (("capex_usd", "inputs"), ("it_kw", "inputs"), ("pue", "facility"),
                 ("usd_per_kwh", "facility"), ("colo_usd_kw_month", "facility"),
                 ("life_years", "life"), ("rate", "rate"))


def tco_export_frame(result: dict) -> pd.DataFrame:
    """One row per GPU: every input with its source, as-of and status, the
    three components, and the totals at full use and at the chosen use."""
    fin, fac = result["financing"], result["facility"]
    out = []
    for row in result["rows"]:
        spec, bd = row["spec"], row["breakdown"]
        rec = {"gpu_key": spec.key, "gpu": spec.label, "selected": row["selected"],
               "priced": bd is not None, "missing_inputs": ", ".join(row["missing"]),
               "preset": fin["preset"], "util_pct": result["util_pct"]}
        for name, where in EXPORT_INPUTS:
            p = (row["inputs"][name] if where == "inputs" else fac[name] if where == "facility"
                 else fin["life"] if where == "life" else fin["rate"])
            col = "rate_pct" if name == "rate" else name
            v = p.get("value")
            rec[col] = (v * 100.0 if name == "rate" and v is not None else v)
            rec[f"{col}_status"] = p.get("status", "")
            rec[f"{col}_source_url"] = p.get("source_url", "")
            rec[f"{col}_as_of"] = p.get("as_of", "")
        cb = row["inputs"]["capex_usd"].get("basis", "")
        rec["capex_basis"] = CAPEX_BASES[cb][1] if cb else ""
        sofr = fin.get("sofr") or {}
        rec["sofr_pct"], rec["sofr_date"] = sofr.get("pct"), sofr.get("date", "")
        for k in ("crf", "capital", "power", "colo", "total_100", "total_at_util"):
            rec[{"capital": "capital_usd_gpu_hr", "power": "power_usd_gpu_hr",
                 "colo": "colo_usd_gpu_hr", "total_100": "owned_usd_gpu_hr_full_use",
                 "total_at_util": "owned_usd_gpu_hr_at_use"}.get(k, k)] = (
                bd[k] if bd else None)
        for k in ("capital", "power", "colo"):
            rec[f"share_{k}"] = bd["shares"][k] if bd else None
        out.append(rec)
    return pd.DataFrame(out)


def export_frame(result: dict) -> pd.DataFrame:
    """The ↓CSV: the TCO table (chart=tco) and the selected GPU's ladder
    (chart=ladder), columns unioned, `chart` first."""
    tco = tco_export_frame(result)
    tco.insert(0, "chart", "tco")
    lad = ladder_rows(result)
    lad.insert(0, "chart", "ladder")
    lad.insert(1, "gpu_key", result["gpu"])
    return pd.concat([tco, lad], ignore_index=True, sort=False)
