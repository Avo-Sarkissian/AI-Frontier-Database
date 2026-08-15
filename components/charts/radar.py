"""
Model comparison radar chart.
Normalizes quality, speed, affordability, and context window
to 0–1 scale and overlays up to 5 selected models.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR,
    BG as _BG, FONT as _FONT, empty_figure, QUALITY_INDEX_MAX, legend_below,
    RADAR_SPEED_MAX, RADAR_PRICE_MAX, RADAR_LATENCY_MAX, RADAR_CONTEXT_K_MAX,
)

_PALETTE = [
    "#00d4ff", "#c084fc", "#34d399", "#fb923c", "#f472b6",
]

DIMS = ["Intelligence", "Speed", "Affordability", "Context", "Latency"]


def _context_k(ctx_str: str) -> float:
    """Convert '200k', '1m', '128k' etc. to numeric thousands."""
    if not ctx_str or pd.isna(ctx_str):
        return 0.0
    s = str(ctx_str).lower().strip()
    try:
        if s.endswith("m"):
            return float(s[:-1]) * 1000
        elif s.endswith("k"):
            return float(s[:-1])
        else:
            return float(s)
    except ValueError:
        return 0.0


def _log_norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Position a value on a log scale between lo and hi, clamped to [0, 1].

    Speed, price, context and latency are all heavy-tailed — speed runs
    24-2277 tok/s with most of the mass under 200, latency 0.2-102s with most
    under 5. On a linear scale the tail owns the axis and everything else piles
    at the origin: 139 of 156 models drew a Speed spoke under 10% of the radius,
    so a 298 tok/s model and a 52 tok/s model both read as "no speed".

    A log scale makes a *decade* the unit, which is how these quantities are
    actually reasoned about: 10x the speed is one step, whatever the base.
    """
    if not np.isfinite(value) or value <= 0 or hi <= lo:
        return 0.0
    lo = max(lo, 1e-9)
    frac = (np.log10(max(value, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    frac = float(min(1.0, max(0.0, frac)))
    return 1.0 - frac if invert else frac


def radar_reference(full_df: pd.DataFrame) -> dict:
    """The five axis endpoints, derived from the population rather than guessed.

    The old ceilings were hand-picked round numbers that had never been checked
    against the data, and every one of them was wrong in a way a visitor could
    see:

        axis          ceiling   pop max   consequence
        Intelligence     70.0     63.05   the best model could never reach 100%
        Speed          2500.0   1559.90   132/148 rendered inside the inner 10%
        Context (k)    2000.0   1000.00   every 1M model pinned at exactly 50.0%
        Price            50.0     40.00   —
        Latency          30.0    101.87   10 models clamped to a flat 0%, so a
                                          32s TTFT and a 102s TTFT drew the
                                          identical zero-length spoke

    Derived from the FULL catalogue and passed in, never from the frame being
    drawn: the subtitle promises a fixed reference, and a ceiling taken from a
    filtered frame would change a model's shape as the user filters.
    """
    q = pd.to_numeric(full_df.get("quality"), errors="coerce")
    s = pd.to_numeric(full_df.get("speed"), errors="coerce")
    p = pd.to_numeric(full_df.get("price"), errors="coerce")
    lat = pd.to_numeric(full_df.get("latency"), errors="coerce")
    ctx = full_df["context"].apply(_context_k) if "context" in full_df else pd.Series(dtype=float)

    def _lo(series, fallback):
        vals = series[series.notna() & (series > 0)] if series is not None and len(series) else None
        return float(vals.min()) if vals is not None and len(vals) else fallback

    def _hi(series, fallback):
        vals = series[series.notna() & (series > 0)] if series is not None and len(series) else None
        return float(vals.max()) if vals is not None and len(vals) else fallback

    return {
        "quality_max": _hi(q, QUALITY_INDEX_MAX),
        "speed_lo":    _lo(s, 1.0),      "speed_hi":   _hi(s, RADAR_SPEED_MAX),
        "price_lo":    _lo(p, 0.01),     "price_hi":   _hi(p, RADAR_PRICE_MAX),
        "ctx_lo":      _lo(ctx, 1.0),    "ctx_hi":     _hi(ctx, RADAR_CONTEXT_K_MAX),
        "lat_lo":      _lo(lat, 0.1),    "lat_hi":     _hi(lat, RADAR_LATENCY_MAX),
    }


def build_radar(df: pd.DataFrame, selected_models: list[str] | None = None,
                full_df: pd.DataFrame | None = None) -> go.Figure:
    if not selected_models:
        # Default: top-5 by quality
        selected_models = (
            df[df["quality"] > 0]
            .sort_values("quality", ascending=False)
            .head(5)["model"]
            .tolist()
        )

    plot_df = df[df["model"].isin(selected_models)].copy()
    if plot_df.empty:
        # A bare go.Figure() renders with Plotly's default WHITE canvas, which
        # flashed a blank white card inside the dark dashboard whenever the
        # selection was cleared.
        return empty_figure("Select up to 5 models to compare")

    # Fixed reference per axis, derived from the FULL catalogue. Both render
    # paths hand this function an already-filtered frame, so taking endpoints
    # from `df` would rescale the axes with the provider/search filters — the
    # same model's profile would change shape, and the subtitle's "normalized
    # across all models" would be untrue. `full_df` is the whole catalogue;
    # falling back to `df` keeps a bare two-argument call working.
    ref = radar_reference(full_df if full_df is not None and not full_df.empty else df)

    fig = go.Figure()

    for idx, model_name in enumerate(selected_models):
        rows = plot_df[plot_df["model"] == model_name]
        if rows.empty:
            continue
        row = rows.iloc[0]

        # Intelligence is the one axis that is already roughly uniform, so it
        # stays linear against the population max — the best model reads 100%.
        q_norm = row["quality"] / ref["quality_max"] if ref["quality_max"] else 0
        s_norm = _log_norm(row["speed"], ref["speed_lo"], ref["speed_hi"])
        # Affordability: cheap is good, so the price axis is inverted.
        p_norm = _log_norm(row["price"], ref["price_lo"], ref["price_hi"], invert=True)
        ctx_k  = _context_k(row.get("context", ""))
        c_norm = _log_norm(ctx_k, ref["ctx_lo"], ref["ctx_hi"])
        # Latency: lower is better — inverted.
        lat = row.get("latency", 0)
        l_norm = _log_norm(float(lat) if pd.notna(lat) else 0.0,
                           ref["lat_lo"], ref["lat_hi"], invert=True)

        # Clamp to [0, 1]. _log_norm already clamps, but quality is linear and
        # a model above the reference max would otherwise exceed the ring — a
        # negative radius is drawn straight through the centre of the polar plot.
        values = [min(1.0, max(0.0, v)) for v in (q_norm, s_norm, p_norm, c_norm, l_norm)]
        values_pct = [round(v * 100) for v in values]

        # One trace per MODEL, so colour must vary per model. Keying it on the
        # provider drew two Anthropic models in the identical hue and fill,
        # making the one chart whose whole job is comparison unable to tell
        # them apart. _PALETTE is indexed by trace instead, which is also what
        # its per-index design was always for.
        color = _PALETTE[idx % len(_PALETTE)]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=DIMS + [DIMS[0]],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
            line=dict(color=color, width=1.5),
            name=model_name[:28] + ("…" if len(model_name) > 28 else ""),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                f"Intelligence: {values_pct[0]}%<br>"
                f"Speed: {values_pct[1]}%<br>"
                f"Affordability: {values_pct[2]}%<br>"
                f"Context: {values_pct[3]}%<br>"
                f"Latency: {values_pct[4]}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Model Comparison"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  0–100% against the full catalogue; speed, price, context "
                "and latency on a log scale</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(color="#777777", size=9, family=_FONT),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
                showline=True,
            ),
            angularaxis=dict(
                tickfont=dict(color="#999999", size=12, family=_FONT),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
        ),
        # Model names are long, so the vertical legend was wider than its own
        # 160px gutter and overlapped the polar plot on narrow viewports.
        legend=legend_below(y=-0.08),
        margin=dict(l=60, r=40, t=52, b=110),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
