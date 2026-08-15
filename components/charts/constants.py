"""
Shared chart constants — imported by all chart modules.
Single source of truth for colors and styling.
"""
import math as _math
import re as _re
import pandas as _pd

# Ordered from most-specific to least-specific so longer matches win
_NAME_SUBS = [
    (_re.compile(r'\(\s*Adaptive\s+Reasoning\s*\)', _re.I), '(AR)'),
    (_re.compile(r'\(\s*Non-reasoning\s*\)',          _re.I), '(NR)'),
    (_re.compile(r'\(\s*Reasoning\s*\)',              _re.I), '(R)'),
    (_re.compile(r'\(\s*xhigh\s*\)',                 _re.I), '(XH)'),
    (_re.compile(r'\(\s*minimal\s*\)',               _re.I), '(min)'),
    (_re.compile(r'\(\s*high\s*\)',                  _re.I), '(H)'),
    (_re.compile(r'\(\s*medium\s*\)',                _re.I), '(M)'),
    (_re.compile(r'\(\s*low\s*\)',                   _re.I), '(L)'),
]


def clean_model_name(name: str, max_len: int = 32) -> str:
    """Abbreviate verbose model name suffixes for display."""
    for pattern, repl in _NAME_SUBS:
        name = pattern.sub(repl, name)
    if len(name) > max_len:
        name = name[:max_len - 1] + '…'
    return name


_BASE_MODEL_RE = _re.compile(r'\s*\([^)]*\)\s*$')


def base_model_name(name: str) -> str:
    """Strip a trailing parenthetical (reasoning effort, snapshot date, …) to
    get the underlying model family name, e.g. 'GPT-5.5 (xhigh)' -> 'GPT-5.5'."""
    stripped = _BASE_MODEL_RE.sub('', name).strip()
    return stripped or name


def dedupe_to_best_variant(df: _pd.DataFrame) -> _pd.DataFrame:
    """Collapse same-family variants (reasoning-effort tiers, dated snapshots,
    …) to a single row — the highest-quality variant per family.

    Providers publish wildly different numbers of variants per model (OpenAI
    benchmarks five reasoning-effort tiers per GPT-5 model, Anthropic
    benchmarks one or two, some models only have one row at all), so plotting
    every row lets a single family dominate a landscape chart with near-
    duplicate points at the same price. Keeping only the peak variant makes
    the "best this model can do" comparison fair across providers.
    """
    if df.empty:
        return df
    working = df.copy()
    working["_base_model"] = working["model"].apply(base_model_name)
    return (
        working.sort_values("quality", ascending=False)
        .drop_duplicates("_base_model", keep="first")
        .drop(columns="_base_model")
    )


# Providers occasionally get renamed upstream by Artificial Analysis. Map the
# retired spelling onto the current one so a rename never silently drops a
# provider into the grey "Other" bucket (which is what happened to Microsoft,
# whose models rendered grey while a colour sat unused under "Microsoft Azure").
PROVIDER_ALIASES: dict[str, str] = {
    "Microsoft Azure": "Microsoft",
    "xAI":             "SpaceXAI",
}


def canonical_provider(name: str) -> str:
    """Resolve a provider name to the spelling the palette is keyed by."""
    return PROVIDER_ALIASES.get(name, name)


# --- Spotlight series palette -------------------------------------------------
# These nine providers can appear as their own series in the Overview scatter,
# so every pair of them has to be separable. The set was solved for, not picked
# by eye — see the palette note below for the measured separations.
#
# The CVD figure sits in the 6–8 floor band, which is only legal alongside
# secondary encoding — that is what PROVIDER_SHAPES below is for. Do not add a
# tenth spotlight provider without re-running the validator: at ten the set is
# infeasible once brand hues are pinned.
#
#   node scripts/validate_palette.js "<hexes>" --mode dark --surface "#111111" --pairs all
SPOTLIGHT_PROVIDERS: tuple[str, ...] = (
    "Anthropic", "Meta", "OpenAI", "Alibaba", "Google",
    "NVIDIA", "Amazon", "Mistral", "DeepSeek",
)

PROVIDER_COLORS: dict[str, str] = {
    # -- spotlight nine. Anthropic is pinned to its EXACT published brand orange,
    #    #D97757 ("clay", read off anthropic.com's own --swatch--clay token; the
    #    darker --swatch--accent is #C6613F). That pin drives everything else here.
    #
    #    Clay is a muted, low-chroma coral, which makes it a far harder colour to
    #    seat than the saturated orange that used to stand in for it: against the
    #    old Alibaba orange it was ΔE 7.1 for normal vision (floor is 15) and
    #    against the old NVIDIA green ΔE 1.5 under deuteranopia. No arrangement
    #    within four colour changes clears the floors — the set had to be re-solved.
    #
    #    The binding constraint was subtle: OpenAI's old #0e8c6d is DARK, and clay
    #    is light, so the two converged at ΔE 3.7 under protanopia — a pair no
    #    other slot could separate. Lightening OpenAI's green fixed it (ΔE 17.5).
    #
    #    Result is better than the palette it replaces: worst normal-vision pair
    #    ΔE 15.1, worst CVD ΔE 8.2 (deutan) / 9.6 (tritan) — above the ΔE 8 target,
    #    so this set no longer leans on marker shape to be legal, though the shapes
    #    stay as secondary encoding. All ≥ 3:1 on the #111111 surface. Nine is
    #    still the ceiling; a tenth makes the set infeasible.
    #
    #    Brand fidelity is kept where the pin allows: Anthropic, Meta, Google and
    #    DeepSeek wear their own hues. Alibaba's only brand colour is the orange
    #    Anthropic now owns, so per the collision rule it takes a distinct hue;
    #    OpenAI, NVIDIA and Mistral likewise moved to seat clay.
    #
    #    Re-validate any change with the dataviz skill's validator:
    #      node scripts/validate_palette.js "<hexes>" --mode dark \
    #           --surface "#111111" --pairs all
    "Anthropic":              "#d97757",  # Anthropic clay — exact brand · 6.05:1
    "Meta":                   "#0566db",  # Meta blue (#0064E0)         · 3.53:1
    "OpenAI":                 "#52d678",  # teal-green, lightened off clay · 10.28:1
    "Alibaba":                "#b46eb6",  # orange is Anthropic's → orchid · 5.30:1
    "Google":                 "#fea92f",  # Google yellow (#FBBC05)     · 9.83:1
    "NVIDIA":                 "#777221",  # green, darkened off Google  · 3.55:1
    "Amazon":                 "#bd088c",  # AWS orange taken → magenta  · 3.22:1
    "Mistral":                "#dcb1f2",  # Mistral orange taken → lavender · 9.44:1
    "DeepSeek":               "#7d9bff",  # DeepSeek blue (#4D6BFE)     · 7.20:1
    "Kimi":                   "#b200c1",  # magenta
    # -- secondary providers: never share the Overview legend with each other,
    #    so they only need to read distinctly in the tabs that show them all --
    "SpaceXAI":               "#ffffff",  # white — requested; xAI/SpaceX brand is white on black
    "Microsoft":              "#818cf8",  # indigo
    "Cohere":                 "#f87171",  # red
    "Z AI":                   "#7dd3fc",  # light blue
    "MiniMax":                "#86efac",  # light green
    "InclusionAI":            "#fca5a5",  # light red
    "Xiaomi":                 "#6ee7b7",  # teal
    "Baidu":                  "#fde68a",  # amber
    "IBM":                    "#93c5fd",  # periwinkle
    "LG AI Research":         "#c4b5fd",  # violet
    "Nous Research":          "#f9a8d4",  # rose
    "Reka AI":                "#a78bfa",  # purple-blue
    "AI21 Labs":              "#2dd4bf",  # teal — was #34d399, colliding with OpenAI
    "Allen Institute for AI": "#67e8f9",  # light cyan
    "Inception":              "#fb7185",  # hot pink
    "Upstage":                "#fbbf24",  # amber
    "Perplexity":             "#a3a3a3",  # neutral
    "KwaiKAT":                "#f97316",  # amber-orange
    "Deep Cogito":            "#a8a29e",  # stone
    "Thinking Machines":      "#5eead4",  # aquamarine
    "Tencent":                "#4ade80",  # green
    "StepFun":                "#e879f9",  # fuchsia
    "Arcee AI":               "#fcd34d",  # gold
    "LongCat":                "#bef264",  # yellow-green
    "Sapiens AI":             "#f0abfc",  # orchid
    "Nex AGI":                "#94a3b8",  # slate
    "Multiverse Computing":   "#d6d3d1",  # warm grey
    "Celeris":                "#fdba74",  # peach
}

DEFAULT_COLOR = "#6b7280"

# Marker shapes per provider — encodes provider identity via shape in addition
# to colour, so the "shape = provider family" claim in the Overview subtitle is
# true and colourblind readers can still separate the labs.
DEFAULT_SHAPE = "circle"

# Explicit shapes for the labs that dominate the catalogue, so their marker
# identity stays stable as the catalogue churns. Plain "circle" is reserved for
# the "Other" bucket and is deliberately unused here.
_EXPLICIT_SHAPES: dict[str, str] = {
    "Anthropic": "square",
    "OpenAI":    "diamond",
    "Google":    "triangle-up",
    "Meta":      "cross",
    "DeepSeek":  "star",
    "Mistral":   "pentagon",
    "SpaceXAI":  "hexagon",
    "Alibaba":   "triangle-down",
    "Amazon":    "diamond-tall",
    # NVIDIA is the closest colour to three other spotlight labs (OpenAI 15.7,
    # Anthropic 15.9, Alibaba 16.4), so it leans hardest on the shape channel.
    # star-triangle-up was indistinguishable from Google's triangle-up at
    # bubble sizes; hourglass shares its outline with nothing else here.
    "NVIDIA":    "hourglass",
    "Kimi":      "hexagon2",
    "Microsoft": "square-cross",
    "Cohere":    "circle-cross",
}

# Remaining filled symbols, assigned deterministically to every other provider
# in the palette so a new entry can never silently collide with a major lab.
_SHAPE_POOL: list[str] = [
    "hexagram", "triangle-left", "triangle-right", "diamond-wide", "octagon",
    "bowtie", "hourglass", "star-square", "star-diamond", "hexagon2",
    "triangle-ne", "triangle-se", "triangle-sw", "triangle-nw", "square-x",
    "diamond-cross", "diamond-x", "circle-x", "star-triangle-down", "x",
    "arrow-up", "arrow-down", "arrow-left", "arrow-right",
    "square-open", "diamond-open", "hexagon-open", "pentagon-open",
]


def _build_provider_shapes() -> dict[str, str]:
    # Spotlight providers must be pinned explicitly: pool assignment shifts
    # whenever an explicit shape changes, which once silently handed Kimi a
    # third triangle alongside Google's and Alibaba's.
    missing = [p for p in SPOTLIGHT_PROVIDERS if p not in _EXPLICIT_SHAPES]
    if missing:
        raise RuntimeError(f"spotlight providers need explicit shapes: {missing}")
    shapes = dict(_EXPLICIT_SHAPES)
    pool = [s for s in _SHAPE_POOL if s not in shapes.values()]
    for name in sorted(set(PROVIDER_COLORS) - set(shapes)):
        shapes[name] = pool.pop(0) if pool else DEFAULT_SHAPE
    return shapes


PROVIDER_SHAPES: dict[str, str] = _build_provider_shapes()

# How many providers earn their own colour/shape/legend entry. Capped at the
# size of the validated spotlight palette: past this the colours on screen would
# no longer be a set the validator has cleared. The rest fold into "Other" — and
# a 25-entry vertical legend was taller than the plot area anyway, so Plotly
# clipped it and hid the major labs behind a scrollbar.
MAX_LEGEND_PROVIDERS = len(SPOTLIGHT_PROVIDERS)

# Bubble diameter ramp for speed. Fixed reference (not the filtered subset's
# max) so a model keeps the same size when filters change, and sqrt-scaled so
# bubble *area* tracks throughput instead of diameter exaggerating it.
BUBBLE_MIN_PX = 7.0
BUBBLE_MAX_PX = 26.0
BUBBLE_SPEED_REF = 900.0   # tok/s mapping to BUBBLE_MAX_PX; faster models clamp

# Shared chart theme tokens
BG    = "#111111"
GRID  = "rgba(255,255,255,0.04)"
TICK  = "#999999"
AXIS  = "#aaaaaa"
FONT  = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# Price at or above which a bubble is drawn at its minimum size on charts that
# encode affordability. Fixed, for the same reason BUBBLE_SPEED_REF is.
BUBBLE_PRICE_REF = 20.0
# Bottom of the price bubble's log domain. The cheapest real model is $0.08;
# 0.05 keeps it just inside the scale rather than pinned at the maximum.
BUBBLE_PRICE_FLOOR = 0.05

# Local models run an order of magnitude slower than hosted APIs, so they get
# their own throughput reference.
LOCAL_SPEED_REF = 400.0

# Ceiling for the AA Intelligence Index axis and colour ramps. Fixed so a
# model keeps the same vertical position and the same tile colour whatever the
# filter — scaling either to the filtered frame's own max made a provider's
# tile darken while its average quality rose.
QUALITY_INDEX_MAX = 70.0

# Radar axis ceilings. Fixed for the same reason: the Compare chart claims
# "normalized 0-100 across all models", so a model's profile must not change
# shape when a filter narrows the frame it is drawn from.
RADAR_SPEED_MAX    = 2500.0   # tok/s
RADAR_PRICE_MAX    = 50.0     # USD / 1M tokens
RADAR_LATENCY_MAX  = 30.0     # seconds TTFT
RADAR_CONTEXT_K_MAX = 2000.0  # thousands of tokens


# ── Shared chart helpers ──────────────────────────────────────────────────────
# Every chart that encodes a magnitude in marker size, prints a correlation, or
# shows a per-provider legend hit the same three bugs. These exist so the fix
# lives in one place rather than being re-derived per module.

def bubble_size(values, ref: float, invert: bool = False,
                log: bool = False, floor: float | None = None) -> _pd.Series:
    """Marker diameters for a magnitude, on a FIXED reference scale.

    Normalising against the plotted frame's own max means the render paths —
    which always receive a filtered frame — silently rescale every mark when the
    user narrows a filter, so a model appears to change speed or price. Values
    are clamped at ``ref`` and sqrt-scaled so bubble *area* tracks the
    magnitude instead of diameter exaggerating it.

    invert=True encodes "smaller value = bigger bubble" (affordability).

    ``log`` places the value on a log scale between ``floor`` and ``ref``, which
    is required whenever the variable spans decades. Price does: it runs
    $0.08-$40.00 with a $0.96 median, so on the linear scale 53 of 95 models sat
    within 0.43 px of each other — a 12.5x price difference rendered as no
    visible difference — while $20.00, $23.75 and $40.00 all drew at exactly the
    7.00 px floor. Speed against a 900 ref is well spread linearly (8.76-26.00
    px measured), so it stays linear and this is opt-in rather than the default.
    """
    span = BUBBLE_MAX_PX - BUBBLE_MIN_PX
    lo = float(floor) if floor else None
    if log and (lo is None or lo <= 0 or ref <= lo):
        log = False           # a degenerate domain would divide by zero

    def _one(v):
        if _pd.isna(v) or v <= 0:
            return BUBBLE_MIN_PX
        if log:
            clamped = min(max(float(v), lo), float(ref))
            frac = ((_math.log10(clamped) - _math.log10(lo))
                    / (_math.log10(float(ref)) - _math.log10(lo)))
        else:
            frac = min(float(v), ref) / ref
        if invert:
            frac = 1.0 - frac
        frac = min(1.0, max(0.0, frac))
        return BUBBLE_MIN_PX + (frac ** 0.5) * span

    return _pd.Series(values).apply(_one)


def safe_corr(x, y, min_points: int = 3) -> float | None:
    """Pearson r, or None when it would be meaningless.

    np.corrcoef returns NaN for fewer than two distinct points, and NaN passes
    an ``is not None`` guard — which is how "r = nan" ended up printed over a
    chart whenever a search narrowed to a single model.
    """
    import numpy as np

    xs, ys = _pd.Series(x).astype(float), _pd.Series(y).astype(float)
    ok = xs.notna() & ys.notna()
    xs, ys = xs[ok], ys[ok]
    if len(xs) < min_points or xs.nunique() < 2 or ys.nunique() < 2:
        return None
    try:
        r = np.corrcoef(xs, ys)[0, 1]
    except Exception:
        return None
    return float(r) if np.isfinite(r) else None


def marker_outline(width: float = 0.8) -> dict:
    """Surface-coloured rim so overlapping marks stay countable.

    Several charts specified ``color="rgba(0,0,0,0)"`` — a fully transparent
    stroke, i.e. dead config — leaving dense clusters an undifferentiated blob.
    """
    return dict(width=width, color=BG)


def legend_below(y: float = -0.16) -> dict:
    """Horizontal legend under the axis.

    The vertical-legend-in-a-fixed-gutter layout clipped entries behind a
    scrollbar once they outgrew the plot height, and cost the data most of the
    width on narrow viewports. Pair with CHART_MARGIN.
    """
    return dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0)",
        borderwidth=0,
        font=dict(color="#999999", size=11, family=FONT),
        itemsizing="constant",
        orientation="h",
        x=0, y=y, xanchor="left", yanchor="top",
        tracegroupgap=2,
    )


# Margin that pairs with legend_below(): the old dict(r=172) reserved a gutter
# for the vertical legend and left narrow viewports 39% of the width for data.
CHART_MARGIN = dict(l=56, r=28, t=52, b=104)


def empty_figure(message: str = "No models match these filters"):
    """A styled placeholder for a frame with nothing to plot.

    Every render path receives a user-filtered frame, so "zero rows" is a
    reachable state, not a theoretical one — and several charts either raised
    (IndexError / KeyError on an empty frame) or returned a bare go.Figure(),
    which renders as Plotly's default WHITE canvas inside a dark dashboard.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, color="#999999", size=12),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=CHART_MARGIN,
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=message, showarrow=False,
            font=dict(color="#666666", size=13, family=FONT),
        )],
    )
    return fig


def unique_labels(names) -> list[str]:
    """Display names guaranteed distinct.

    Charts truncate long model names for the axis, which can collapse two
    models onto one categorical value — Plotly then draws both bars in the same
    row, burying one of them where it can never be hovered. Collisions get a
    numeric suffix so every bar keeps its own row.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            suffix = f" ({seen[name]})"
            out.append(name[: max(0, len(name) - len(suffix))] + suffix)
        else:
            seen[name] = 1
            out.append(name)
    return out


def log_ticks(lo: float, hi: float, fmt=None) -> tuple[list[float], list[str]]:
    """1-2-5 ticks across the decades spanned, with readable labels.

    A log axis carrying no explicit tickvals makes Plotly fall back to "D2"
    minor ticks, which render as a bare-mantissa run like
    "2 5 0.1 2 5 1 2 5 10 2 5 100" — the leading label reads as a value in its
    own right and misstates the axis by orders of magnitude.
    """
    import math

    if not (lo > 0 and hi > 0 and hi >= lo):
        return [], []
    fmt = fmt or (lambda v: f"{v:g}")
    vals: list[float] = []
    decade = math.floor(math.log10(lo))
    while 10 ** decade <= hi * 10:
        for mantissa in (1, 2, 5):
            v = mantissa * (10 ** decade)
            if lo / 1.6 <= v <= hi * 1.6:
                vals.append(v)
        decade += 1
    return vals, [fmt(v) for v in vals]


def spotlight_split(df: _pd.DataFrame, provider_col: str = "provider"):
    """Split providers into their own series vs a shared "Other" bucket.

    Returns ``(df, ordered_names)`` where df gains a ``display_provider``
    column. Only SPOTLIGHT_PROVIDERS get their own colour, which is what keeps
    the colours on screen a subset the palette validator has cleared for
    all-pairs separation; names are ordered densest-first so the legend leads
    with the labs a reader is looking for, and "Other" always sorts last.
    """
    out = df.copy()
    out["_canon_provider"] = out[provider_col].apply(canonical_provider)
    counts = out["_canon_provider"].value_counts()
    # Total order, not just value_counts' order. value_counts does not define
    # how ties break, and after dedupe the top providers genuinely tie (OpenAI
    # 8 / Google 8 / Mistral 8) — so pandas 3.x in CI and pandas 2.2.3 in the
    # browser produced DIFFERENT legend orders from identical data. The
    # pre-rendered figure and the in-browser re-render disagreed, and since the
    # worker's `ready` message triggers a re-render with no user action, legend
    # entries visibly swapped a few seconds after every page load.
    candidates = sorted(counts.index, key=lambda p: (-int(counts[p]), str(p)))
    named = [p for p in candidates if p in SPOTLIGHT_PROVIDERS][:MAX_LEGEND_PROVIDERS]
    named_set = set(named)
    out["display_provider"] = out["_canon_provider"].apply(
        lambda p: p if p in named_set else "Other"
    )
    ordered = [p for p in named if (out["display_provider"] == p).any()]
    if (out["display_provider"] == "Other").any():
        ordered.append("Other")
    return out, ordered


# Widest right gutter any chart may reserve for its row annotations. Four charts
# used to hard-code 130–280px AND leave 30–55% axis headroom, so the bars were
# squeezed twice and took ~43% of the width on desktop, far less on a phone.
MAX_RIGHT_GUTTER_PX = 200
# Annotations sit against the paper edge now, so the axis needs only a hair of
# headroom rather than a gutter's worth.
ANNOTATED_AXIS_HEADROOM = 1.02


def right_gutter(texts, size_px: int = 10, pad_px: int = 20) -> int:
    """Pixels to reserve for the longest right-side annotation.

    Sized from the actual strings so a chart neither clips its labels nor keeps
    a gutter it stopped needing; capped at MAX_RIGHT_GUTTER_PX so one very long
    model name cannot take the plot back.
    """
    longest = max((len(str(t)) for t in texts), default=0)
    return int(min(MAX_RIGHT_GUTTER_PX, longest * size_px * 0.55 + pad_px))


def fit_text(text: str, gutter_px: int, size_px: int = 10, pad_px: int = 20) -> str:
    """Truncate to what actually fits the gutter, with an ellipsis.

    The cap means a long label would otherwise run off the canvas — clipped mid
    word by the edge, which looks like a rendering fault rather than a choice.
    """
    text = str(text)
    max_chars = max(8, int((gutter_px - pad_px) / (size_px * 0.55)))
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
