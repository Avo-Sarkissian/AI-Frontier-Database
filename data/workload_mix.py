"""Workload MIX: what a model costs for a given input:output ratio and cache-hit rate.

Every hosted price on this dashboard used to be ONE blend, (input + 3 x output) / 4.
That is a fine default for chat-like traffic and badly wrong for the workloads
people now pay for: an agent re-reads a long, mostly cached context hundreds of
times per output token, so a $5 / $25 model that lists at $20 costs about $1 per
million tokens there. The MIX control prices the catalogue for a stated workload.

The formula, written once (weights w_in : w_out, cache-hit share h of INPUT
tokens, all prices USD per 1M tokens):

    effective = [ w_in * (h * p_cache + (1 - h) * p_in) + w_out * p_out ] / (w_in + w_out)

With r = w_in / w_out that is [r(h*c + (1-h)*in) + out] / (r + 1). It is a blended
price per 1M TOTAL tokens (input + output), so a monthly volume times it is a bill.

Rules:
  * h == 0 never reads the cache price, so a model with none is never flagged and
    0 * NaN can never poison a row.
  * h > 0 with no published cache price bills the cache hits at the FULL input
    price (CACHE_FILL_RATIO = 1.0: no discount is invented), and
    `cache_price_filled` marks the row so every surface can say so.
  * A row with no input or output price stays NaN and drops out of the price
    views exactly as it did before (the unhosted Rankings rows).
  * The cache-WRITE premium (Anthropic bills a cache write above the input price)
    is not modelled; `Mix.long` says so for every h > 0.

The default mix is the dashboard's historical basis. `apply_mix` returns the
SAME frame object for it, so the pre-rendered figures and every number on an
untouched page are byte-identical to what they were before this module existed;
only label text changes.

Pure pandas, no dash/requests: it runs in Pyodide (build_static.build_pybundle
ships it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# ── Cache fill-in ─────────────────────────────────────────────────────────────
# Where Artificial Analysis publishes no cached-input price for a model, its
# cache hits are billed at the FULL input price: no discount is assumed.
#   Why not a typical discount: over the 120 models AA does price a cache hit
#   for, cacheHitPrice / price1mInputTokens has median 0.10 and range
#   0.008-1.0 (https://artificialanalysis.ai/leaderboards/models, measured
#   2026-10-27), and SemiAnalysis InferenceX's calculator defaults cached input
#   to 10% (https://inferencex.semianalysis.com/api/openapi.json, read
#   2026-10-27). But that median is over models whose host OFFERS caching; a
#   missing price is as likely to mean the cheapest host has no cache discount
#   at all. A 10% fill-in gave 42 unpriced rows a 90% discount on 90% of their
#   input and put 11 of them in the 15 cheapest at the agentic mix, the floor
#   model included. 1.0 is the top of the published range, so a filled row is
#   never cheaper than its published list prices allow: an upper bound, flagged.
#   tests/test_workload_mix.py re-checks that no published ratio exceeds it.
CACHE_FILL_RATIO = 1.0
CACHE_FILL_SOURCE_URL = "https://artificialanalysis.ai/leaderboards/models"
CACHE_FILL_AS_OF = "2026-10-27"

DEFAULT_MIX_KEY = "ours"
CUSTOM_KEY = "custom"
# Custom ratio bounds: input tokens per ONE output token. 0 = output-only.
CUSTOM_RATIO_MAX = 1000.0


@dataclass(frozen=True)
class MixPreset:
    """One named workload. Every figure carries its source and the date it was
    read (`as_of`); `status` uses the shared provenance vocabulary
    (published / reported / derived / estimated)."""
    key: str
    w_in: float
    w_out: float
    cache_hit: float          # share of INPUT tokens served from cache, 0..1
    label: str                # the <option> text
    short: str                # what every price label appends
    source_url: str
    as_of: str
    status: str
    note: str


MIX_PRESETS: tuple[MixPreset, ...] = (
    MixPreset(
        key="ours", w_in=1.0, w_out=3.0, cache_hit=0.0,
        label="Ours 1:3 (default)",
        short="ours 1:3 in:out (not AA's)",
        source_url="https://github.com/Avo-Sarkissian/AI-Frontier-Database/blob/main/data/scraper.py",
        as_of="2026-05-06",
        status="derived",
        note=("This dashboard's own blend since 2026-05-06: (input + 3 x output) / 4 of "
              "Artificial Analysis's per-token list prices, no cache. It is ours, not AA's: "
              "AA's published blend weights the other way."),
    ),
    MixPreset(
        key="chat", w_in=1.0, w_out=1.0, cache_hit=0.0,
        label="Chat 1:1",
        short="chat 1:1 in:out",
        source_url="https://inferencex.semianalysis.com/api/v1/views/options?format=json",
        as_of="2026-10-23",
        status="estimated",
        note=("Illustrative even split, an estimate rather than a measured workload. "
              "Precedent: InferenceX's fixed 1k-in / 1k-out benchmark sequence."),
    ),
    MixPreset(
        key="aa", w_in=3.0, w_out=1.0, cache_hit=0.0,
        label="AA 3:1",
        short="AA 3:1 in:out",
        source_url="https://artificialanalysis.ai/api-reference",
        as_of="2026-10-23",
        status="published",
        note=("Artificial Analysis's standard blended price, price_1m_blended_3_to_1 = "
              "(3 x input + output) / 4, no cache."),
    ),
    MixPreset(
        key="observed", w_in=15.0, w_out=1.0, cache_hit=0.0,
        label="Observed 15:1 (late 2025)",
        short="observed 15:1 in:out",
        source_url="https://openrouter.ai/state-of-ai",
        as_of="2026-10-23",
        status="reported",
        note=("a16z x OpenRouter 'State of AI' 100-trillion-token study (Dec 2025, data "
              "Nov 2024-Nov 2025): by late 2025 the average request had grown to a prompt "
              "of >6K tokens and a completion of ≈400, so input:output was AT LEAST 15:1. "
              "Those are end-of-window values, not averages over the window (prompts were "
              "≈1.5K and completions ≈150 in early 2024). Quoted, not scraped. Cache rate "
              "not published, so none is assumed."),
    ),
    MixPreset(
        key="agentic", w_in=300.0, w_out=1.0, cache_hit=0.90,
        label="Agentic 300:1 · 90% cache",
        short="agentic 300:1 in:out · 90% cached",
        source_url="https://newsletter.semianalysis.com/p/ai-value-capture-the-shift-to-model",
        as_of="2026-05-01",
        status="estimated",
        note=("SemiAnalysis analyst estimate ('AI Value Capture', 2026-05-01): agentic work "
              "runs about 300:1 input:output with 90%+ cache hits. An estimate, not a "
              "measurement. Cache-write premium not modelled."),
    ),
)
_PRESETS = {p.key: p for p in MIX_PRESETS}


def _ratio_text(w_in: float, w_out: float) -> str:
    """'1:3', '15:1', '300:1', '0:1' — input:output, the smaller side as 1."""
    if w_out <= 0:
        return "1:0"
    r = w_in / w_out
    if r == 0:
        return "0:1"
    if r >= 1:
        return f"{_num(r)}:1"
    return f"1:{_num(1 / r)}"


def _num(v: float) -> str:
    """Up to 3 significant digits, no trailing zeros: 15, 2.5, 0.333."""
    return f"{v:.3g}" if v < 1000 else f"{v:.0f}"


@dataclass(frozen=True)
class Mix:
    """A resolved workload: weights, cache share and the labels that name it."""
    w_in: float
    w_out: float
    cache_hit: float
    key: str = DEFAULT_MIX_KEY
    short: str = ""
    source_url: str = ""
    as_of: str = ""
    status: str = ""
    note: str = ""

    @property
    def is_default(self) -> bool:
        return self.key == DEFAULT_MIX_KEY

    @property
    def ratio(self) -> float:
        """Input tokens per one output token."""
        return self.w_in / self.w_out if self.w_out else math.inf

    @property
    def output_share_pct(self) -> float:
        total = self.w_in + self.w_out
        return 100.0 * self.w_out / total if total else 0.0

    @property
    def _share_text(self) -> str:
        share = self.output_share_pct
        return f"{share:.0f}%" if share >= 10 or share == 0 else f"{share:.2g}%"

    @property
    def assumption(self) -> str:
        """Compact form for a chart subtitle: '75% output tokens (ours 1:3 ...)'.
        The Budget chart's whole cost model is tokens x this price, so its title
        states it; the default must read '75% output' (pinned by a test)."""
        return f"{self._share_text} output tokens  ·  {self.short}"

    @property
    def long(self) -> str:
        """The assumption in words, for tooltips, the CSV's price_basis column
        and captions. The default says '75% output' and "not AA's"."""
        parts = [f"{self.short}: {self._share_text} output tokens"]
        if self.cache_hit > 0:
            parts.append(f"{self.cache_hit * 100:.0f}% of input served from cache "
                         f"(AA cache price; {FILL_TEXT}); "
                         "cache-write premium not modelled")
        else:
            parts.append("no cache")
        if self.is_default:
            parts.append("AA's own published blend weights 3 input to 1 output")
        return ", ".join(parts)

    @property
    def title(self) -> str:
        """Hover/tooltip text: what the mix is and where its numbers come from."""
        src = f" Source: {self.source_url} ({self.as_of})." if self.source_url else ""
        note = f" {self.note}" if self.note else ""
        return f"{self.long}.{note}{src}"

    def as_args(self) -> list:
        """The positional form the browser sends: [key, ratio|None, cache_pct|None]."""
        if self.key == CUSTOM_KEY:
            return [CUSTOM_KEY, self.ratio, self.cache_hit * 100.0]
        return [self.key, None, None]


def _from_preset(p: MixPreset) -> Mix:
    return Mix(w_in=p.w_in, w_out=p.w_out, cache_hit=p.cache_hit, key=p.key,
               short=p.short, source_url=p.source_url, as_of=p.as_of,
               status=p.status, note=p.note)


DEFAULT_MIX = _from_preset(_PRESETS[DEFAULT_MIX_KEY])


def custom_mix(ratio, cache_pct) -> Mix:
    """A user-typed mix. Blank, NaN or infinite inputs take the DEFAULT mix's
    values (ratio 1:3, no cache) rather than an invented number; the rest is
    clamped to [0, CUSTOM_RATIO_MAX] and [0, 100]."""
    from static_helpers import coerce_number

    r = coerce_number(ratio, default=DEFAULT_MIX.ratio, minimum=0.0)
    r = min(r, CUSTOM_RATIO_MAX)
    pct = coerce_number(cache_pct, default=DEFAULT_MIX.cache_hit * 100.0, minimum=0.0)
    pct = min(pct, 100.0)
    h = pct / 100.0
    short = f"custom {_ratio_text(r, 1.0)} in:out · {_num(pct)}% cached"
    return Mix(w_in=r, w_out=1.0, cache_hit=h, key=CUSTOM_KEY, short=short,
               status="estimated", note="Typed by the viewer.")


def resolve_mix(spec=None) -> Mix:
    """Turn whatever a caller sent into a Mix. Never raises.

    Accepts None/'' (default), a preset key, the browser's [key, ratio, cache_pct]
    list, a dict ({key|m, ratio|mr, cache|mh}) or a Mix. Unknown keys fall back to
    the default: a URL or a stale page is user input.
    """
    if isinstance(spec, Mix):
        return spec
    key, ratio, cache = None, None, None
    try:
        if isinstance(spec, dict):
            key = spec.get("key", spec.get("m"))
            ratio = spec.get("ratio", spec.get("mr"))
            cache = spec.get("cache", spec.get("mh"))
        elif isinstance(spec, (list, tuple)):
            vals = list(spec) + [None, None, None]
            key, ratio, cache = vals[0], vals[1], vals[2]
        else:
            key = spec
    except Exception:
        return DEFAULT_MIX
    key = key if isinstance(key, str) else None
    if key == CUSTOM_KEY:
        return custom_mix(ratio, cache)
    preset = _PRESETS.get(key or DEFAULT_MIX_KEY)
    return _from_preset(preset) if preset else DEFAULT_MIX


def effective_price(p_in, p_out, p_cache, mix: Mix):
    """(effective $/1M total tokens, cache_price_filled mask) — Series in, Series out."""
    p_in = pd.to_numeric(pd.Series(p_in), errors="coerce").astype(float)
    p_out = pd.to_numeric(pd.Series(p_out), errors="coerce").astype(float).set_axis(p_in.index)
    if p_cache is None:
        p_cache = pd.Series(float("nan"), index=p_in.index)
    p_cache = pd.to_numeric(pd.Series(p_cache), errors="coerce").astype(float).set_axis(p_in.index)
    h = float(mix.cache_hit)
    # A zero input weight never reads the cache either (output-only pricing).
    if h > 0 and mix.w_in > 0:
        filled = p_cache.isna() & p_in.notna()
        cache = p_cache.where(p_cache.notna(), CACHE_FILL_RATIO * p_in)
        inp = h * cache + (1.0 - h) * p_in
    else:
        filled = pd.Series(False, index=p_in.index)
        inp = p_in
    eff = (mix.w_in * inp + mix.w_out * p_out) / (mix.w_in + mix.w_out)
    return eff, filled.astype(bool)


def apply_mix(df: pd.DataFrame, mix: Mix | None = None) -> pd.DataFrame:
    """The catalogue re-priced for `mix`.

    The default returns `df` itself, untouched. Otherwise a copy where `price` is
    the effective price, `price_list_blend` keeps the dashboard's list blend, and
    `cache_price_filled` marks rows with no published cache price (cache hits
    billed at the full input price, CACHE_FILL_RATIO). Rows,
    index and order never change.
    """
    mix = resolve_mix(mix)
    if mix.is_default or df is None or "price" not in df.columns:
        return df
    out = df.copy()
    missing = pd.Series(float("nan"), index=df.index)
    eff, filled = effective_price(df.get("price_in", missing), df.get("price_out", missing),
                                  df.get("cache_hit_price", missing), mix)
    out["price_list_blend"] = df["price"]
    out["price"] = eff.values
    out["cache_price_filled"] = filled.values
    return out


def fill_note(df: pd.DataFrame, mix: Mix) -> str:
    """'  ·  * N of M: no AA cache price, billed at full input' when it applies,
    else ''. Builders put it FIRST in the subtitle: Plotly titles do not wrap,
    and the provenance note is the part a narrow card must not clip."""
    if mix.cache_hit <= 0 or df is None or "cache_price_filled" not in df.columns:
        return ""
    priced = df[df["price"].notna() & (df["price"] > 0)] if "price" in df.columns else df
    n = int(priced["cache_price_filled"].eq(True).sum())
    if not n:
        return ""
    return f"  ·  {FILL_MARK} {n} of {len(priced)}: {FILL_SHORT}"


def filled_flags(df: pd.DataFrame) -> pd.Series:
    """Per-row bool: is this row's cache price the fill-in? False where the
    column is absent (the default mix, or h == 0)."""
    if df is None:
        return pd.Series(dtype=bool)
    col = df.get("cache_price_filled")
    if col is None:
        return pd.Series(False, index=df.index)
    return col.eq(True)


def is_filled(v) -> bool:
    """A `cache_price_filled` cell as a plain bool: numpy bools, None and NaN
    (a frame at the default mix has no such column) all handled."""
    try:
        return v is not None and not pd.isna(v) and bool(v)
    except (TypeError, ValueError):
        return False


def mark_price(v, filled) -> str:
    """fmt_price, plus FILL_MARK where the cache price is the fill-in."""
    return fmt_price(v) + (FILL_MARK if is_filled(filled) else "")


def price_marks(df: pd.DataFrame) -> pd.Series:
    """mark_price for every row of `df` (its `price` column)."""
    flags = filled_flags(df)
    return pd.Series([mark_price(p, f) for p, f in zip(df["price"], flags)], index=df.index)


def fill_hover(df: pd.DataFrame) -> pd.Series:
    """A hover line per row: FILL_HOVER where the row is filled, else ''."""
    return filled_flags(df).map(lambda f: FILL_HOVER if f else "")


# One wording for the fill-in everywhere (subtitle, hover, floor stat, panel).
FILL_MARK = "*"
FILL_SHORT = "no AA cache price, billed at full input"
FILL_TEXT = "none published: billed at the full input price, no discount assumed"
FILL_HOVER = f"<br>{FILL_MARK} no AA cache price: cache hits billed at full input"


def fmt_price(v) -> str:
    """$x.xx at $1 and above, $x.xxx down to a cent, 4 significant digits below —
    so a $0.0041 agentic floor never prints as $0.00."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if v >= 1:
        return f"${v:,.2f}"
    if v >= 0.01:
        return f"${v:.3f}"
    if v == 0:
        return "$0.00"
    return f"${v:.4g}"


def cache_label(row) -> str:
    """The detail panel's "Cached in" value: the cached-input price this
    dashboard uses for a model, and whose it is. AA's per-model figure where it
    publishes one (it can differ from the vendor's own page), else the full
    input price, marked. One function for Dash and the static site."""
    c, p_in = row.get("cache_hit_price"), row.get("price_in")
    if c is not None and pd.notna(c):
        return f"{fmt_price(c)} / 1M  ·  AA"
    if p_in is not None and pd.notna(p_in):
        return (f"{fmt_price(CACHE_FILL_RATIO * float(p_in))} / 1M{FILL_MARK}  ·  "
                "none published: input rate")
    return "—"


def fmt_floor(v) -> str:
    """The header stat's format: '$0.080' as it always was, sub-cent via fmt_price."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"${v:.3f}" if v >= 0.01 else fmt_price(v)


def floor_label(mix: Mix | None = None) -> str:
    return f"Floor price / 1M · {resolve_mix(mix).short}"


def table_price_header(mix: Mix | None = None) -> str:
    return f"Price ($/M · {resolve_mix(mix).short})"


def mix_options() -> list[dict]:
    """The MIX <select>: presets then Custom, each with its source in `title`
    and its numbers, so the browser can pre-fill the custom boxes from whatever
    preset was showing instead of inventing a starting value."""
    out = []
    for p in MIX_PRESETS:
        m = _from_preset(p)
        out.append({"value": p.key, "label": p.label, "title": m.title,
                    "ratio": m.ratio, "cache_pct": p.cache_hit * 100.0,
                    "status": p.status, "source_url": p.source_url, "as_of": p.as_of})
    out.append({"value": CUSTOM_KEY, "label": "Custom…",
                "title": ("Your own input:output ratio (input tokens per output token, "
                          f"0-{CUSTOM_RATIO_MAX:g}) and cache-hit rate (0-100% of input). "
                          "Models with no AA cache price bill cache hits at the full input price."),
                "ratio": None, "cache_pct": None, "status": "estimated",
                "source_url": "", "as_of": ""})
    return out


def mix_summary(df: pd.DataFrame, mix: Mix | None = None) -> dict:
    """What the header stat and the table header need for one mix, computed over
    the UNFILTERED catalogue (the floor is a market-wide claim, as it always was)."""
    mix = resolve_mix(mix)
    base = apply_mix(df, mix)
    priced = base[base["price"].notna() & (base["price"] > 0)]
    if priced.empty:
        floor_v, floor_model, floor_filled = float("nan"), "", False
    else:
        i = priced["price"].idxmin()
        floor_v = float(priced.loc[i, "price"])
        floor_model = str(priced.loc[i, "model"])
        floor_filled = (bool(priced.loc[i, "cache_price_filled"])
                        if "cache_price_filled" in priced.columns else False)
    n_filled = (int(priced["cache_price_filled"].eq(True).sum())
                if "cache_price_filled" in priced.columns else 0)
    return {
        "key": mix.key,
        "args": mix.as_args(),
        "short": mix.short,
        "long": mix.long,
        "title": mix.title,
        "floor_price": fmt_floor(floor_v),
        "floor_model": floor_model,
        "floor_filled": floor_filled,
        "floor_label": floor_label(mix),
        "table_header": table_price_header(mix),
        "n_filled": n_filled,
        "n_priced": int(len(priced)),
        # The floor stat's visible key when floor_filled (a touch screen has
        # no hover, so the '*' cannot be explained by a title alone).
        "fill_key": f"{FILL_MARK} {FILL_SHORT}",
    }
