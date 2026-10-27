"""Economics › Labs: what the labs earn, what they keep, what they have
committed to spend and how many tokens they serve. Pyodide-safe (stdlib +
pandas); nothing here fetches.

Two inputs:

* Epoch AI's "Data on AI companies" hub (CC BY 4.0; Josh You, John Croxton,
  Venkat Somala, Yafah Edelman, epoch.ai). The bot mirrors its CSVs into
  data/raw/epoch/ (data/econ_archive.py, its own cadence, range-checked there).
  Two slim slices are cut from the mirrors at BUILD time and written into
  pydata.zip (build_static.DERIVED_DATA): DERIVED_REVENUE (annualised revenue
  rows) and DERIVED_TOKENS (rows carrying Epoch's normalised "Daily tokens").
  The cut DROPS any row outside the unit ranges with a printed warning; it
  never raises, because a raise here would block the hourly AA publish.

* data/curated/lab_ledger.csv, hand-curated: every row carries its source URL,
  the source's date, the date the figure refers to (as_of), a company-claim
  BADGE (disclosed = filing or on-record company statement; reported = press
  citing people or documents; estimated = analyst model or Epoch
  interpolation), the filing's audit status, a provenance `status`
  (econ_common.STATUS_VALUES), `ledger_state` (ship | excluded) and, for a
  company Epoch tracks, `epoch_company` (Epoch's exact company name: the key
  that pairs a commitment with its run-rate and a token row with Epoch's
  rows; blank = Epoch has no rows for it, so nothing is paired). Excluded
  rows (a leaked draft figure, a projection mislabelled as commitments) stay
  in the ledger with their reason and are never charted.

The value-capture STACK (view "stack", one `layer` per row: chips, capex,
apps, required) puts each layer's yearly dollars side by side: Nvidia's
filed data-center quarter × 4, the four hyperscalers' 2026 capex guidance
(a range: `value` low, `value_high` high), the labs' latest Epoch run-rates
(computed here, not curated) and Cursor's reported ARR, against Sequoia's
"$600B question" (Nvidia run-rate × 2 × 2), rerun on the filed quarter.

Every margin is drawn with ONE definition per row (entity × definition), so
no line ever joins a gross margin to a derived "revenue less all compute" or
to an analyst's inference-only estimate.
"""
from __future__ import annotations

import csv
import functools
import io
import math
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.econ_common import ROOT, read_derived

EPOCH_REVENUE = "data/raw/epoch/ai_companies_revenue_reports.csv"
EPOCH_USAGE = "data/raw/epoch/ai_companies_usage_reports.csv"
LEDGER_FILE = "data/curated/lab_ledger.csv"
DERIVED_REVENUE = "data/derived/epoch_lab_revenue.csv"
DERIVED_TOKENS = "data/derived/epoch_lab_tokens.csv"

EPOCH_URL = "https://epoch.ai/data/ai-companies"
EPOCH_LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
EPOCH_CREDIT = "Epoch AI, 'Data on AI companies' (CC BY 4.0)"
EPOCH_CITATION = ("Josh You, John Croxton, Venkat Somala, Yafah Edelman, "
                  "'Data on AI companies', epoch.ai")

# Epoch's company names -> this dashboard's provider palette keys
# (components/charts/constants.PROVIDER_COLORS). Anything else is grey.
EPOCH_TO_PALETTE: dict[str, str] = {
    "OpenAI": "OpenAI", "Anthropic": "Anthropic", "Google": "Google",
    "Mistral AI": "Mistral", "Cohere": "Cohere", "xAI": "SpaceXAI",
    "DeepSeek": "DeepSeek", "Meta": "Meta", "Z.ai (Zhipu)": "Z AI",
    "MiniMax": "MiniMax", "Moonshot AI": "Kimi",
}

BADGES = ("disclosed", "reported", "estimated")
FILING_STATUSES = ("audited", "unaudited", "n/a")
LEDGER_STATES = ("ship", "excluded")
LEDGER_VIEWS = ("margin", "commitment", "token", "revenue", "footnote", "ledger", "stack")
DENOMINATOR_BASES = ("epoch_runrate", "quarterly_revenue_x4")

LEDGER_COLUMNS = (
    "row_id", "entity", "palette_name", "entity_type", "view", "label", "metric",
    "definition", "value", "value_qualifier", "unit", "per", "scope", "numerator",
    "denominator", "nd_unit", "period_label", "period_start", "period_end", "badge",
    "filing_status", "derived", "denominator_basis", "source_name", "source_url",
    "source_date", "retrieved", "ledger_state", "as_of", "status", "note",
    "epoch_company", "layer", "value_high",
)
REVENUE_COLUMNS = ("epoch_id", "company", "palette_name", "date", "annualized_usd",
                   "revenue_type", "scope", "confidence", "source_type", "badge",
                   "source_url")
TOKEN_COLUMNS = ("epoch_id", "company", "palette_name", "product", "scope", "date",
                 "tokens_per_day", "confidence", "source_type", "badge", "source_url",
                 "statement", "graph_excluded", "origin")

# Epoch's own conventions: a monthly figure ÷ 30 (not 30.44), weekly ÷ 7,
# per-minute × 1,440 ("Google 9.7T monthly" -> 3.23e11/day; "API 6B tk/min"
# -> 8.6e12/day).
# (multiplier, divisor) so a weekly figure is value / 7 exactly, not value × (1/7).
_PER_DAY: dict[str, tuple[int, int]] = {"day": (1, 1), "week": (1, 7), "month": (1, 30),
                                        "minute": (1440, 1)}
TOKENS_PER_DAY_FACTOR: dict[str, float] = {k: m / d for k, (m, d) in _PER_DAY.items()}

# Unit ranges. Anything outside is a unit slip upstream, not a fact.
REVENUE_RANGE = (1e6, 1e12)          # annualised USD
TOKENS_RANGE = (1e8, 1e16)           # tokens per day
EARLIEST_DATE = date(2015, 1, 1)
FUTURE_SLACK_DAYS = 31

COMMITMENT_MAX_GAP_DAYS = 92         # a run-rate older than this is no denominator
TOKEN_DEDUP_DAYS = 7                 # a curated row this close to an Epoch row is dropped

# Token-volume scope: what the disclosure counts. Drawn as marker symbols.
SCOPE_LABELS = {"all": "all products", "api": "API only",
                "product": "one product or model family"}
SCOPE_SYMBOLS = {"all": "circle", "api": "square", "product": "triangle-up"}


# ── Small parsers ────────────────────────────────────────────────────────────

def _num(v) -> float | None:
    try:
        f = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(v) -> str | None:
    s = str(v or "").strip()[:10]
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def _col(df: pd.DataFrame, name: str) -> str | None:
    """The frame's column called `name`, ignoring case and surrounding space:
    Epoch spells it 'Source type' in one file and 'Source Type' in another."""
    want = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == want:
            return c
    return None


def _need(df: pd.DataFrame, names: tuple[str, ...], what: str) -> dict[str, str]:
    cols = {n: _col(df, n) for n in names}
    missing = [n for n, c in cols.items() if c is None]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}")
    return cols


_URL = re.compile(r"https://[^\s,;\"'<>]+")


def first_https(cell) -> str:
    """The first https URL in a cell ('' if none): Epoch packs several sources
    into one cell, and only an https link is ever emitted."""
    m = _URL.search(str(cell or ""))
    return m.group(0) if m else ""


def source_domain(url: str) -> str:
    m = re.match(r"https://([^/?#]+)", str(url or ""))
    return (m.group(1).lower().removeprefix("www.")) if m else ""


def epoch_badge(source_type, revenue_type=None) -> str:
    """Company-claim badge for an Epoch row.

    'Period interpolation' is Epoch annualising disclosed PERIOD revenue, so
    the annualised figure is its estimate whatever the source; otherwise a
    company disclosure anywhere in the source list wins, then a media report;
    'Other' or nothing is an estimate."""
    rt = "" if revenue_type is None or (isinstance(revenue_type, float)
                                         and math.isnan(revenue_type)) else str(revenue_type)
    if rt.strip().lower() == "period interpolation":
        return "estimated"
    st = "" if source_type is None or (isinstance(source_type, float)
                                        and math.isnan(source_type)) else str(source_type)
    st = st.lower()
    if "company disclosure" in st:
        return "disclosed"
    if "media report" in st:
        return "reported"
    return "estimated"


def token_scope(product) -> str:
    """What a token disclosure counts: 'all' (whole company, every product),
    'api' (API traffic only) or 'product' (one product or model family)."""
    p = "" if product is None or (isinstance(product, float) and math.isnan(product)) \
        else str(product).strip().lower()
    if ("full company" in p or "all products" in p or "all ai products" in p
            or ("chat" in p and "api" in p)):
        return "all"
    if p == "api" or p.endswith(" api"):
        return "api"
    return "product"


def tokens_per_day(value, per) -> float | None:
    v = _num(value)
    md = _PER_DAY.get(str(per or "").strip().lower())
    return None if v is None or md is None else v * md[0] / md[1]


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


# ── Unit checks ──────────────────────────────────────────────────────────────

def out_of_range(frame: pd.DataFrame, kind: str, today: date | None = None) -> pd.Series:
    """Boolean mask of rows whose value or date is outside the unit ranges.
    kind: 'revenue' (annualized_usd) or 'tokens' (tokens_per_day)."""
    today = today or date.today()
    col, (lo, hi) = (("annualized_usd", REVENUE_RANGE) if kind == "revenue"
                     else ("tokens_per_day", TOKENS_RANGE))
    if frame.empty:
        return pd.Series([], dtype=bool)
    vals = pd.to_numeric(frame[col], errors="coerce")
    latest = today + timedelta(days=FUTURE_SLACK_DAYS)

    def bad_date(d):
        iso = _iso(d)
        return iso is None or not (EARLIEST_DATE <= date.fromisoformat(iso) <= latest)

    return (vals.isna() | (vals < lo) | (vals > hi)) | frame["date"].map(bad_date)


def validate_units(frame: pd.DataFrame, kind: str, today: date | None = None) -> None:
    """Raise ValueError naming the first row outside the unit ranges."""
    bad = out_of_range(frame, kind, today)
    if bad.any():
        r = frame[bad].iloc[0]
        raise ValueError(f"{kind}: row {r.get('epoch_id', '?')!r} out of range "
                         f"({r.to_dict()})")


def _drop_out_of_range(df: pd.DataFrame, kind: str, today: date | None) -> pd.DataFrame:
    bad = out_of_range(df, kind, today)
    if bad.any():
        for _, r in df[bad].iterrows():
            print(f"  WARNING: Epoch {kind} row {r['epoch_id']!r} dropped "
                  f"(value or date out of range)")
    return df[~bad] if len(bad) else df


# ── Epoch slices (build time) ────────────────────────────────────────────────

def slim_revenue(raw: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Epoch revenue_reports -> REVENUE_COLUMNS: rows with a date and an
    annualised value, in range, sorted (company, date, id)."""
    c = _need(raw, ("Id", "Company", "Date", "Annualized revenue (USD)",
                    "Annualized revenue type", "Scope", "Confidence", "Source type",
                    "Source 1"), "revenue_reports")
    df = pd.DataFrame({
        "epoch_id": raw[c["Id"]].astype(str).str.strip(),
        "company": raw[c["Company"]].astype(str).str.strip(),
        "date": raw[c["Date"]].map(_iso),
        "annualized_usd": raw[c["Annualized revenue (USD)"]].map(_num),
        "revenue_type": raw[c["Annualized revenue type"]].fillna("").astype(str).str.strip(),
        "scope": raw[c["Scope"]].fillna("").astype(str).str.strip(),
        "confidence": raw[c["Confidence"]].fillna("").astype(str).str.strip(),
        "source_type": raw[c["Source type"]].fillna("").astype(str).str.strip(),
        "source_url": raw[c["Source 1"]].map(first_https),
    })
    df = df[df["date"].notna() & df["annualized_usd"].notna()]
    df = _drop_out_of_range(df, "revenue", today)
    df = df.assign(palette_name=df["company"].map(EPOCH_TO_PALETTE).fillna(""),
                   badge=[epoch_badge(s, t) for s, t in zip(df["source_type"],
                                                            df["revenue_type"])])
    df = df.sort_values(["company", "date", "epoch_id"], kind="mergesort")
    return df[list(REVENUE_COLUMNS)].reset_index(drop=True)


def slim_tokens(raw: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Epoch usage_reports -> TOKEN_COLUMNS: rows with a 'Daily tokens' value
    (Epoch's normalisation, already per day), in range."""
    c = _need(raw, ("Id", "Company", "Date", "Daily tokens", "Product", "Confidence",
                    "Source type", "Source 1"), "usage_reports")
    excl = _col(raw, "Exclude from graph view")
    df = pd.DataFrame({
        "epoch_id": raw[c["Id"]].astype(str).str.strip(),
        "company": raw[c["Company"]].astype(str).str.strip(),
        "product": raw[c["Product"]].fillna("").astype(str).str.strip(),
        "date": raw[c["Date"]].map(_iso),
        "tokens_per_day": raw[c["Daily tokens"]].map(_num),
        "confidence": raw[c["Confidence"]].fillna("").astype(str).str.strip(),
        "source_type": raw[c["Source type"]].fillna("").astype(str).str.strip(),
        "source_url": raw[c["Source 1"]].map(first_https),
        "graph_excluded": (raw[excl].map(_truthy) if excl else False),
    })
    df = df[df["date"].notna() & df["tokens_per_day"].notna()]
    df = _drop_out_of_range(df, "tokens", today)
    df = df.assign(palette_name=df["company"].map(EPOCH_TO_PALETTE).fillna(""),
                   scope=df["product"].map(token_scope),
                   badge=df["source_type"].map(epoch_badge),
                   statement=df["epoch_id"], origin="epoch")
    df = df.sort_values(["company", "date", "epoch_id"], kind="mergesort")
    return df[list(TOKEN_COLUMNS)].reset_index(drop=True)


def _frame_csv(df: pd.DataFrame, cols) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, columns=list(cols), lineterminator="\n",
              float_format="%.6g", quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue()


def revenue_csv(path: Path | None = None) -> str:
    """DERIVED_REVENUE text from the Epoch mirror ('' when the mirror is absent)."""
    path = Path(path) if path is not None else ROOT / EPOCH_REVENUE
    if not path.exists():
        return ""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return _frame_csv(slim_revenue(raw), REVENUE_COLUMNS)


def tokens_csv(path: Path | None = None) -> str:
    """DERIVED_TOKENS text from the Epoch mirror ('' when the mirror is absent)."""
    path = Path(path) if path is not None else ROOT / EPOCH_USAGE
    if not path.exists():
        return ""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return _frame_csv(slim_tokens(raw), TOKEN_COLUMNS)


# ── Loaders (browser and local alike) ────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_epoch_revenue() -> pd.DataFrame:
    df = read_derived(DERIVED_REVENUE, revenue_csv)
    if df.empty:
        return pd.DataFrame(columns=list(REVENUE_COLUMNS))
    df["annualized_usd"] = pd.to_numeric(df["annualized_usd"], errors="coerce")
    for col in REVENUE_COLUMNS:
        if col != "annualized_usd":
            df[col] = df[col].fillna("").astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_epoch_tokens() -> pd.DataFrame:
    df = read_derived(DERIVED_TOKENS, tokens_csv)
    if df.empty:
        return pd.DataFrame(columns=list(TOKEN_COLUMNS))
    df["tokens_per_day"] = pd.to_numeric(df["tokens_per_day"], errors="coerce")
    df["graph_excluded"] = df["graph_excluded"].map(_truthy)
    for col in TOKEN_COLUMNS:
        if col not in ("tokens_per_day", "graph_excluded"):
            df[col] = df[col].fillna("").astype(str)
    return df


@functools.lru_cache(maxsize=1)
def _ledger_all() -> pd.DataFrame:
    path = ROOT / LEDGER_FILE
    if not path.exists():
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in ("value", "numerator", "denominator", "value_high"):
        df[col] = df[col].map(_num)
    df["order"] = range(len(df))
    return df


def load_ledger(include_excluded: bool = False) -> pd.DataFrame:
    df = _ledger_all().copy()
    if not include_excluded:
        df = df[df["ledger_state"] == "ship"]
    return df.reset_index(drop=True)


def reset_cache() -> None:
    for f in (load_epoch_revenue, load_epoch_tokens, _ledger_all):
        f.cache_clear()


# ── Chart frames ─────────────────────────────────────────────────────────────

def runrate_frame() -> pd.DataFrame:
    """Epoch's 'Full company' annualised revenue rows (product/division rows
    are not a company's run-rate)."""
    df = load_epoch_revenue()
    if df.empty:
        return df
    df = df[df["scope"].str.strip().str.lower().eq("full company")
            & df["annualized_usd"].notna()]
    return df.sort_values(["company", "date", "epoch_id"], kind="mergesort").reset_index(drop=True)


def margin_rows() -> pd.DataFrame:
    """Shipped margin rows, one chart row per `label` (= entity × definition),
    each sorted by the end of its period."""
    df = load_ledger()
    df = df[(df["view"] == "margin") & df["value"].notna()].copy()
    if df.empty:
        return df
    first = df.groupby("label", sort=False)["order"].transform("min")
    df["row_order"] = first.rank(method="dense").astype(int) - 1
    df["sort_date"] = df["period_end"].where(df["period_end"] != "", df["as_of"])
    return df.sort_values(["row_order", "sort_date", "order"], kind="mergesort").reset_index(drop=True)


def commitment_rows(max_gap_days: int = COMMITMENT_MAX_GAP_DAYS) -> pd.DataFrame:
    """Shipped commitment rows with their denominator and multiple (×).

    A lab's denominator is the latest Epoch 'Full company' run-rate of the
    company named in its `epoch_company` (never matched by colour), dated on
    or before the commitment's as_of and at most `max_gap_days` earlier; a
    row with no company or no such run-rate is kept, flagged and not
    charted. A row whose denominator_basis is 'quarterly_revenue_x4' uses its
    own filed quarterly revenue × 4 (CoreWeave: the sell side, a different
    definition); a missing or non-positive quarter is flagged likewise."""
    led = load_ledger()
    led = led[(led["view"] == "commitment") & led["value"].notna()]
    rr = runrate_frame()
    out = []
    for _, r in led.iterrows():
        rec = {k: r[k] for k in ("row_id", "entity", "palette_name", "label", "definition",
                                 "value", "value_qualifier", "as_of", "badge",
                                 "filing_status", "source_url", "source_name",
                                 "denominator_basis", "order")}
        rec.update(den_usd=None, den_date="", den_badge="", den_label="", den_source="",
                   den_epoch_id="", multiple=None, charted=False, flag="")
        basis = r["denominator_basis"]
        company = str(r.get("epoch_company", "") or "").strip()
        if basis == "quarterly_revenue_x4":
            q = _num(r["denominator"])
            if q is not None and q > 0:
                rec.update(den_usd=q * 4, den_date=r["as_of"],
                           den_badge=r["badge"], den_source=r["source_url"],
                           den_label=f"Q revenue {_usd(q)} × 4")
            else:
                rec["flag"] = "no quarterly revenue denominator; not charted"
        elif basis == "epoch_runrate" and not company:
            rec["flag"] = "no Epoch company named (epoch_company); not charted"
        elif basis == "epoch_runrate":
            asof = _iso(r["as_of"])
            cand = rr[rr["company"] == company] if asof else rr.iloc[0:0]
            if asof is not None and not cand.empty:
                d = pd.to_datetime(cand["date"])
                gap = (pd.Timestamp(asof) - d).dt.days
                cand = cand[(gap >= 0) & (gap <= max_gap_days)]
            if not cand.empty:
                hit = cand.sort_values(["date", "epoch_id"], kind="mergesort").iloc[-1]
                rec.update(den_usd=float(hit["annualized_usd"]), den_date=hit["date"],
                           den_badge=hit["badge"], den_source=hit["source_url"],
                           den_epoch_id=hit["epoch_id"],
                           den_label=f"run-rate {_usd(hit['annualized_usd'])} "
                                     f"(Epoch, {hit['date']}, {hit['badge']})")
            else:
                rec["flag"] = (f"no Epoch run-rate within {max_gap_days} days before "
                               f"{r['as_of']}; not charted")
        else:
            rec["flag"] = f"unknown denominator basis {basis!r}; not charted"
        den = _num(rec["den_usd"])
        if den is not None and den > 0:
            rec["multiple"] = r["value"] / den
            rec["charted"] = math.isfinite(rec["multiple"])
        out.append(rec)
    cols = ["row_id", "entity", "palette_name", "label", "definition", "value",
            "value_qualifier", "as_of", "badge", "filing_status", "source_url", "source_name",
            "denominator_basis", "order", "den_usd", "den_date", "den_badge", "den_label",
            "den_source", "den_epoch_id", "multiple", "charted", "flag"]
    return pd.DataFrame(out, columns=cols)


_SCALE = ((1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million"))


def human_tokens(v: float) -> str:
    for base, word in _SCALE:
        if abs(v) >= base:
            return f"{v / base:.4g} {word}"
    return f"{v:,.0f}"


def token_rows() -> pd.DataFrame:
    """Epoch's usage rows plus the curated token rows, all in tokens per day.

    A curated row is dropped when Epoch already has a row for the same company
    (`epoch_company`, never the colour) AND the same scope, within
    TOKEN_DEDUP_DAYS days, that Epoch itself draws (not 'Exclude from graph
    view'): Epoch's own normalisation then wins. An API-only or hidden Epoch
    row never removes an all-products disclosure.

    `statement` is the quoted figure for a curated row and Epoch's row name
    for an Epoch row (its usage file carries only the per-day normalisation,
    not the raw count and period), so the hover labels them differently."""
    ep = load_epoch_tokens()
    rows = []
    for _, r in ep.iterrows():
        rows.append({
            "entity": r["company"], "palette_name": r["palette_name"], "scope": r["scope"],
            "product": r["product"], "date": r["date"],
            "tokens_per_day": r["tokens_per_day"], "badge": r["badge"],
            "statement": r["statement"], "value": None, "per": "day",
            "source_url": r["source_url"], "source_name": "Epoch AI usage_reports",
            "origin": "epoch", "ref": r["epoch_id"],
            "graph_excluded": bool(r["graph_excluded"]), "epoch_company": r["company"],
        })
    led = load_ledger()
    led = led[(led["view"] == "token") & led["value"].notna()]
    for _, r in led.iterrows():
        tpd = tokens_per_day(r["value"], r["per"])
        if tpd is None:
            continue
        q = {">": "over ", "~": "about ", "=": ""}.get(r["value_qualifier"] or "=", "")
        rows.append({
            "entity": r["entity"], "palette_name": r["palette_name"],
            "scope": r["scope"] or "all", "product": r["label"], "date": r["as_of"],
            "tokens_per_day": tpd, "badge": r["badge"],
            "statement": f"{q}{human_tokens(r['value'])} tokens per {r['per']}",
            "value": r["value"], "per": r["per"], "source_url": r["source_url"],
            "source_name": r["source_name"], "origin": "curated", "ref": r["row_id"],
            "graph_excluded": False,
            "epoch_company": str(r.get("epoch_company", "") or "").strip(),
        })
    cols = ["entity", "palette_name", "scope", "product", "date", "tokens_per_day", "badge",
            "statement", "value", "per", "source_url", "source_name", "origin", "ref",
            "graph_excluded", "epoch_company"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    drawn = df[(df["origin"] == "epoch") & ~df["graph_excluded"].astype(bool)]
    keep = []
    for _, r in df.iterrows():
        if r["origin"] == "curated" and r["epoch_company"]:
            same = drawn[(drawn["epoch_company"] == r["epoch_company"])
                         & (drawn["scope"] == r["scope"])]
            gaps = (pd.to_datetime(same["date"]) - pd.Timestamp(r["date"])).abs().dt.days
            if (gaps <= TOKEN_DEDUP_DAYS).any():
                keep.append(False)
                continue
        keep.append(True)
    df = df[keep]
    return df.sort_values(["entity", "scope", "date"], kind="mergesort").reset_index(drop=True)


# ── The value-capture stack ─────────────────────────────────────────────────

STACK_LAYERS = ("apps", "labs", "capex", "chips")      # top to bottom on the chart
STACK_LAYER_LABELS = {
    "apps": "Apps · Cursor, one app (reported ARR)",
    "labs": "Labs · latest run-rates (Epoch AI)",
    "capex": "Clouds · 2026 capex guidance",
    "chips": "Chips · Nvidia data-center revenue",
}
USD_PER_YEAR = {"year": 1, "quarter": 4}               # a usd row's `per` -> × to a year
# Sequoia's method (David Cahn, 2024-06-20, sequoiacap.com/article/ais-600b-question):
# Nvidia's run-rate × 2 (GPUs are half a data centre's total cost) × 2 (a 50%
# gross margin for the GPU's end user).
SEQUOIA_URL = "https://www.sequoiacap.com/article/ais-600b-question/"
SEQUOIA_DC_MULT = 2
SEQUOIA_MARGIN_MULT = 2
STACK_LAB_WINDOW_DAYS = 365   # a lab's run-rate older than this (vs the newest) is left out
_BADGE_RANK = {b: i for i, b in enumerate(BADGES)}


def weakest_badge(badges) -> str:
    """The weakest evidence among a layer's parts (estimated < reported < disclosed)."""
    have = [b for b in badges if b in _BADGE_RANK]
    return max(have, key=_BADGE_RANK.get) if have else "estimated"


def usd_range(lo, hi) -> str:
    """'$725–745B' when both ends share a unit, else '$900M–$1.1B'."""
    a, b = _usd(lo), _usd(hi)
    if a[-1] == b[-1] and a[-1] in "TBMK":
        return f"{a[:-1]}–{b[1:]}"
    return f"{a}–{b}"


def stack_lab_runrates(window_days: int = STACK_LAB_WINDOW_DAYS) -> pd.DataFrame:
    """Each company's latest Epoch run-rate or ARR SNAPSHOT (never an
    annualised period, which looks backward), kept only if dated within
    `window_days` of the newest snapshot in the data (no clock is read)."""
    rr = runrate_frame()
    if rr.empty:
        return rr
    snap = rr[rr["revenue_type"].str.strip().str.lower() != "period interpolation"]
    if snap.empty:
        return snap
    last = (snap.sort_values(["date", "epoch_id"], kind="mergesort")
            .groupby("company", sort=False).tail(1))
    newest = pd.Timestamp(last["date"].max())
    keep = (newest - pd.to_datetime(last["date"])).dt.days <= window_days
    return last[keep].sort_values(["annualized_usd", "company"], ascending=[False, True],
                                  kind="mergesort").reset_index(drop=True)


def _part(name, lo, hi, badge, as_of, url, q="", ref="", note=""):
    return {"name": name, "lo": lo, "hi": hi, "badge": badge, "as_of": as_of,
            "source_url": url, "qualifier": q, "ref": ref, "note": note}


def stack_rows() -> pd.DataFrame:
    """One row per layer (STACK_LAYERS) plus the two 'required' references,
    every value in USD per year: lo, hi (equal unless a range), mid, the
    weakest badge of its parts, its parts (list of dicts) and a `basis`
    sentence. A layer with no parts is left out; the rerun needs the chips
    layer."""
    led = load_ledger()
    led = led[(led["view"] == "stack") & led["value"].notna()
              & led["per"].isin(list(USD_PER_YEAR))]
    out = []

    def add(layer, kind, parts, basis):
        if not parts:
            return
        lo = sum(p["lo"] for p in parts)
        hi = sum(p["hi"] for p in parts)
        out.append({"layer": layer, "kind": kind, "lo": lo, "hi": hi, "mid": (lo + hi) / 2,
                    "badge": weakest_badge(p["badge"] for p in parts),
                    "as_of": max(p["as_of"] for p in parts),
                    "qualifier": ">" if any(p["qualifier"] == ">" for p in parts) else "",
                    "n_parts": len(parts), "parts": parts, "basis": basis})

    def ledger_parts(layer):
        parts = []
        for _, r in led[led["layer"] == layer].iterrows():
            k = USD_PER_YEAR[r["per"]]
            hi = r["value_high"] if pd.notna(r["value_high"]) else r["value"]
            q = str(r["value_qualifier"] or "")
            parts.append(_part(r["entity"], float(r["value"]) * k, float(hi) * k, r["badge"],
                               r["as_of"], r["source_url"], q=q, ref=r["row_id"],
                               note=(f"{r['period_label']} {_usd(r['value'])} × {k}"
                                     if k != 1 else str(r["period_label"]))))
        return parts

    chips = ledger_parts("chips")
    add("chips", "layer", chips, "filed quarter × 4 (revenue)")
    capex = ledger_parts("capex")
    add("capex", "layer", capex,
        f"sum of {len(capex)} companies' guidance (spend, not revenue; definitions differ)")
    labs = [_part(r["company"], float(r["annualized_usd"]), float(r["annualized_usd"]),
                  r["badge"], r["date"], r["source_url"], ref=r["epoch_id"],
                  note=str(r["revenue_type"]))
            for _, r in stack_lab_runrates().iterrows()]
    add("labs", "layer", labs,
        f"sum of {len(labs)} labs' latest run-rate or ARR (revenue; gross and net mixed)")
    add("apps", "layer", ledger_parts("apps"), "one app's reported ARR: a floor")
    if chips:
        k = SEQUOIA_DC_MULT * SEQUOIA_MARGIN_MULT
        add("required_now", "reference",
            [_part(p["name"], p["lo"] * k, p["hi"] * k, "estimated", p["as_of"],
                   p["source_url"], ref=p["ref"],
                   note=f"{_usd(p['lo'])}/yr × {SEQUOIA_DC_MULT} × {SEQUOIA_MARGIN_MULT}")
             for p in chips],
            "Sequoia's × 2 × 2 rerun on the filed Nvidia quarter (our arithmetic)")
    add("required_2024", "reference", ledger_parts("required"),
        "Sequoia Capital's figure, June 2024")
    cols = ["layer", "kind", "lo", "hi", "mid", "badge", "as_of", "qualifier", "n_parts",
            "parts", "basis"]
    return pd.DataFrame(out, columns=cols)


def stack_export_frame() -> pd.DataFrame:
    """The stack's numbers, one row per part, with its layer total."""
    rows = []
    for _, s in stack_rows().iterrows():
        for p in s["parts"]:
            rows.append({"layer": s["layer"], "kind": s["kind"], "layer_lo_usd_yr": s["lo"],
                         "layer_hi_usd_yr": s["hi"], "layer_basis": s["basis"],
                         "part": p["name"], "part_lo_usd_yr": p["lo"],
                         "part_hi_usd_yr": p["hi"], "qualifier": p["qualifier"],
                         "badge": p["badge"], "as_of": p["as_of"],
                         "source_url": p["source_url"], "ref": p["ref"], "note": p["note"],
                         "credit": EPOCH_CREDIT if s["layer"] == "labs" else ""})
    return pd.DataFrame(rows)


# ── Formatting shared by the charts, the ledger and the export ──────────────

def _usd(v) -> str:
    """Up to four significant digits, trailing zeros dropped, so a filed
    $103.7B or ~$265.6B reads as filed ($9B stays $9B)."""
    v = _num(v)
    if v is None:
        return "—"
    for base, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= base:
            x = v / base
            return f"${x:.4g}{suf}" if abs(x) < 1000 else f"${x:,.0f}{suf}"
    return f"${v:,.0f}"


def fmt_value(row) -> str:
    """A ledger row's value in its own unit, with its qualifier."""
    v = _num(row.get("value"))
    if v is None:
        return "—"
    q = {">": ">", "~": "~", "=": ""}.get(str(row.get("value_qualifier") or ""), "")
    unit = row.get("unit")
    if unit == "pct":
        return f"{q}{v:.1f}%"
    if unit == "usd":
        hi = _num(row.get("value_high"))
        text = usd_range(v, hi) if hi is not None and hi != v else _usd(v)
        per = str(row.get("per") or "")
        return f"{q}{text}" + (f" / {per}" if per in USD_PER_YEAR and per != "year" else "")
    if unit == "tokens":
        per = row.get("per") or "day"
        tpd = tokens_per_day(v, per)
        base = f"{q}{human_tokens(v)} / {per}"
        return base if per == "day" or tpd is None else f"{base} (≈ {human_tokens(tpd)} / day)"
    return f"{q}{v:g}"


def ledger_export_frame() -> pd.DataFrame:
    """Every Labs number behind the charts and the ledger, one row each, with
    a `chart` column naming which (ledger | commitment_multiple | runrate |
    tokens | stack) and a `credit` column on every Epoch-derived row."""
    led = load_ledger(include_excluded=True).drop(columns=["order"], errors="ignore")
    led.insert(0, "chart", "ledger")
    com = commitment_rows().drop(columns=["order"], errors="ignore")
    com.insert(0, "chart", "commitment_multiple")
    com["credit"] = [EPOCH_CREDIT if b == "epoch_runrate" and c else ""
                     for b, c in zip(com["denominator_basis"], com["charted"])]
    rr = runrate_frame().copy()
    rr.insert(0, "chart", "runrate")
    rr["credit"] = EPOCH_CREDIT
    tk = token_rows().copy()
    tk.insert(0, "chart", "tokens")
    tk["credit"] = [EPOCH_CREDIT if o == "epoch" else "" for o in tk.get("origin", [])]
    st = stack_export_frame()
    if not st.empty:
        st.insert(0, "chart", "stack")
    frames = [f for f in (led, com, rr, tk, st) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["chart"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "credit" in out.columns:
        out["credit"] = out["credit"].fillna("")
    return out
