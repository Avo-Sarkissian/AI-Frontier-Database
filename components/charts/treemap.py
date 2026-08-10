"""
Provider ecosystem treemap.
Tile area = number of models; tile color = average quality score.
"""
import pandas as pd
import plotly.graph_objects as go

from components.charts.constants import BG as _BG, FONT as _FONT, QUALITY_INDEX_MAX


def build_treemap(df: pd.DataFrame) -> go.Figure:
    # Aggregate by provider
    agg = (
        df.groupby("provider")
        .agg(
            model_count=("model", "count"),
            avg_quality=("quality", "mean"),
            avg_price=("price", "mean"),
            avg_speed=("speed", "mean"),
            best_model=("quality", lambda s: df.loc[s.idxmax(), "model"]),
        )
        .reset_index()
    )
    agg = agg[agg["model_count"] >= 1].sort_values("model_count", ascending=False)

    hover = (
        "<b>%{label}</b><br>"
        "Models: %{customdata[0]}<br>"
        "Avg Intelligence: %{customdata[1]:.1f}<br>"
        "Avg Price: $%{customdata[2]:.3f}/M<br>"
        "Best model: %{customdata[3]}<br>"
        "<extra></extra>"
    )

    fig = go.Figure(go.Treemap(
        labels=agg["provider"],
        parents=[""] * len(agg),
        values=agg["model_count"],
        customdata=agg[["model_count", "avg_quality", "avg_price", "best_model"]].values,
        hovertemplate=hover,
        marker=dict(
            colors=agg["avg_quality"],
            # Fixed domain. Without cmin/cmax Plotly autoscales the ramp to
            # whatever subset is on screen, so a provider changed colour with
            # the filter — and could go DARKER as its average quality rose,
            # inverting the encoding. A single-provider filter was worse still:
            # the degenerate range collapsed to v±0.5 and every provider painted
            # the same mid-ramp navy whatever its actual score.
            cmin=0.0,
            cmax=QUALITY_INDEX_MAX,
            colorscale=[
                # The old bottom stop (#1a1a2e) sat within 9 RGB units of the
                # #111111 page, so the weakest provider was painted as the
                # background.
                [0.0,  "#243056"],
                [0.3,  "#16213e"],
                [0.55, "#0f3460"],
                [0.75, "#1a5276"],
                [0.9,  "#00909e"],
                [1.0,  "#00d4ff"],
            ],
            showscale=True,
            colorbar=dict(
                thickness=10,
                len=0.6,
                tickfont=dict(color="#999999", size=10, family=_FONT),
                title=dict(
                    text="AvgScore",
                    font=dict(color="#999999", size=10, family=_FONT),
                    side="right",
                ),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.07)",
                borderwidth=1,
                outlinewidth=0,
            ),
        ),
        textfont=dict(family=_FONT, color="#f2f2f2", size=12),
        tiling=dict(packing="squarify", pad=2),
        pathbar=dict(visible=False),
    ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color="#999999", size=12),
        title=dict(
            text=(
                "Provider Landscape"
                "  <span style='font-size:12px;color:#777777;font-weight:400'>"
                "  ·  area = # models  ·  color = avg intelligence</span>"
            ),
            font=dict(size=15, color="#f2f2f2", family=_FONT, weight=600),
            x=0.0, xanchor="left",
            pad=dict(l=20, t=16),
        ),
        margin=dict(l=20, r=20, t=52, b=20),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#161616", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f2f2f2", size=12, family=_FONT), namelength=-1,
        ),
    )

    return fig
