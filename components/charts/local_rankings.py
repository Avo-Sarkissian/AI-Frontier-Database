"""
Local models — overall quality rankings bar chart.

Shows ALL open-weight models ranked by AA Intelligence Index score,
regardless of user hardware. Color = model family. Right-side
annotations show parameter count and license.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG as _BG, GRID as _GRID, TICK as _TICK, AXIS as _AXIS, FONT as _FONT
from data.local_models import FAMILY_COLORS, DEFAULT_FAMILY_COLOR


def build_local_rankings(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of all local models ranked by quality score.
    df is the full output of data.local_models.get_local_df() (unfiltered by fits).
    De-duplicated on name so multi-quant rows don't appear twice.
    """
    # One row per model name — quality/family/params don't change with quant
    ranked = (
        df.drop_duplicates(subset=["name"])
          .sort_values("quality", ascending=True)
          .reset_index(drop=True)
    )

    if ranked.empty:
        return _empty("No models found.")

    # Truncate long names
    ranked["short_name"] = ranked["name"].apply(
        lambda n: n[:36] + "…" if len(n) > 36 else n
    )

    colors = ranked["family"].map(FAMILY_COLORS).fillna(DEFAULT_FAMILY_COLOR).tolist()
    max_q  = ranked["quality"].max() or 1

    fig = go.Figure()

    # Ghost track for full-width alignment
    fig.add_trace(go.Bar(
        y=ranked["short_name"],
        x=[max_q * 1.0] * len(ranked),
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.02)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Quality bars
    fig.add_trace(go.Bar(
        y=ranked["short_name"],
        x=ranked["quality"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        customdata=ranked[["name", "family", "params_b", "license", "context_k", "tags_str"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Family: %{customdata[1]}<br>"
            "Intelligence: %{x:.0f}<br>"
            "Parameters: %{customdata[2]:.0f}B<br>"
            "License: %{customdata[3]}<br>"
            "Context: %{customdata[4]}k tokens<br>"
            "Tags: %{customdata[5]}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Right-side annotations: param count + license
    for _, row in ranked.iterrows():
        color     = FAMILY_COLORS.get(row["family"], DEFAULT_FAMILY_COLOR)
        params_b  = row["params_b"]
        params_str = f"{params_b:.0f}B" if params_b >= 1 else f"{params_b*1000:.0f}M"
        license_str = row["license"]
        if len(license_str) > 12:
            license_str = license_str[:11] + "…"

        fig.add_annotation(
            x=max_q + 0.5,
            y=row["short_name"],
            text=f"{params_str}  ·  {license_str}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, family=_FONT, color=color),
            xref="x", yref="y",
        )

    height = max(500, len(ranked) * 24 + 80)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Open-Weight Model Rankings"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                f"  ·  {len(ranked)} models  ·  hardware-independent  ·  ranked by intelligence"
                "</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        xaxis=dict(
            title=dict(text="AA Intelligence Index", font=dict(color=_AXIS, size=12), standoff=12),
            range=[0, max_q * 1.55],
            gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=1,
            tickfont=dict(color=_TICK, size=11, family=_FONT),
            showgrid=True, showline=False, ticks="",
        ),
        yaxis=dict(
            tickfont=dict(color="#999999", size=11, family=_FONT),
            showgrid=False, showline=False, ticks="",
            automargin=True,
        ),
        barmode="overlay",
        margin=dict(l=20, r=200, t=52, b=36),
        height=height,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#888888", size=12),
        annotations=[dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=msg, showarrow=False,
            font=dict(color="#666666", size=13, family=_FONT),
            align="center",
        )],
        margin=dict(l=40, r=40, t=60, b=40), height=400,
    )
    return fig
