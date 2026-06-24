"""Browser bridge: ports the Dash callbacks to JSON-returning functions.
Runs inside Pyodide. No dash/requests/flask imports.

Task 5 extends this module (local/image/video/agent-stack).
"""
import json
import pandas as pd

from data.ingest import get_models
from components.charts.constants import PROVIDER_COLORS, DEFAULT_COLOR
from components.charts.pareto import build_pareto_scatter
from components.charts.quadrant import build_quadrant
from components.charts.treemap import build_treemap
from components.charts.provider_leaderboard import build_provider_leaderboard
from components.charts.rankings import build_rankings
from components.charts.bump_chart import build_value_leaders
from components.charts.radar import build_radar
from components.charts.cost_calc import build_cost_calc
from static_helpers import (
    apply_filters,
    compute_diverse5,
    ctx_to_k,
    quality_label,
    model_options,
    provider_options,
)

_DF = get_models()


def _apply_filters(providers, min_quality, search=""):
    return apply_filters(_DF, providers, min_quality, search)


# ── HTML-string mirrors of app.py's dash-component builders ──────────────────

def _build_raw_table_html(df: pd.DataFrame, selected: list) -> str:
    """Mirror of app.py _build_raw_table (lines ~130–189) as an HTML string."""
    _FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    rows = df[df["model"].isin(selected)].copy()
    if rows.empty:
        return "<div></div>"

    header_style = (
        "padding:8px 14px;font-size:9px;letter-spacing:0.08em;color:#555;"
        f"font-family:{_FONT};font-weight:600;text-transform:uppercase;"
        "text-align:right;white-space:nowrap;"
    )
    header_left_style = header_style.replace("text-align:right", "text-align:left")
    cell_base = (
        f"padding:7px 14px;font-size:11px;font-family:{_FONT};"
        "border-bottom:1px solid rgba(255,255,255,0.04);text-align:right;white-space:nowrap;"
    )

    def _th(label, align="right"):
        style = header_left_style if align == "left" else header_style
        return f'<th style="{style}">{label}</th>'

    def _td(text, color="#888", extra_style=""):
        style = f"{cell_base}color:{color};{extra_style}"
        return f'<td style="{style}">{text}</td>'

    header = (
        "<tr>"
        + _th("Model", "left")
        + _th("Provider", "left")
        + _th("Intelligence")
        + _th("Price ($/M tok)")
        + _th("Speed (tok/s)")
        + _th("Latency (TTFT)")
        + _th("Context")
        + "</tr>"
    )

    table_rows = []
    for _, r in rows.sort_values("quality", ascending=False).iterrows():
        pcolor = PROVIDER_COLORS.get(r["provider"], DEFAULT_COLOR)
        price_str = f"${r['price']:.4f}" if pd.notna(r["price"]) and r["price"] > 0 else "—"
        speed_str = f"{int(r['speed']):,}" if pd.notna(r["speed"]) and r["speed"] > 0 else "—"
        lat_str = f"{r['latency']:.2f}s" if pd.notna(r["latency"]) and r["latency"] > 0 else "—"
        ctx_str = str(r["context"]) if pd.notna(r.get("context")) else "—"

        model_td_style = (
            f"{cell_base}text-align:left;color:#ccc;max-width:260px;"
            "overflow:hidden;text-overflow:ellipsis;"
        )
        row_html = (
            "<tr>"
            + f'<td style="{model_td_style}">{r["model"]}</td>'
            + f'<td style="{cell_base}text-align:left;color:{pcolor};">{r["provider"]}</td>'
            + _td(f"{r['quality']:.1f}", "#f2f2f2")
            + _td(price_str)
            + _td(speed_str)
            + _td(lat_str)
            + _td(ctx_str)
            + "</tr>"
        )
        table_rows.append(row_html)

    label_style = (
        "font-size:9px;letter-spacing:0.1em;color:#444;"
        f"font-family:{_FONT};padding:14px 14px 6px;font-weight:600;"
    )
    table_style = "width:100%;border-collapse:collapse;overflow-x:auto;"
    return (
        f'<div><div style="{label_style}">RAW VALUES</div>'
        f'<table style="{table_style}">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        f"</table></div>"
    )


def _detail_html(row: pd.Series, provider: str) -> str:
    """Mirror of app.py toggle_detail_panel body (lines ~1259–1318) as HTML string."""
    _FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    color = PROVIDER_COLORS.get(provider, DEFAULT_COLOR)
    quality = float(row["quality"])
    q_max = float(_DF["quality"].max()) or 1.0
    quality_pct = quality / q_max * 100
    qlabel = quality_label(quality_pct)

    speed_val = row["speed"]
    speed_str = f"{int(speed_val):,} tok/s" if pd.notna(speed_val) and speed_val > 0 else "N/A"
    latency_val = row.get("latency", float("nan"))
    latency_str = f"{latency_val:.2f}s" if pd.notna(latency_val) and latency_val > 0 else "N/A"

    n_total = len(_DF[_DF["quality"] > 0])
    n_below = int((_DF["quality"] < quality).sum())
    pct = round(n_below / n_total * 100) if n_total else 0

    def _metric(lbl, val, accent=False):
        val_class_extra = " accent" if accent else ""
        return (
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
            f'<span style="font-size:9px;letter-spacing:0.08em;color:#555;'
            f'font-family:{_FONT};text-transform:uppercase;">{lbl}</span>'
            f'<span style="font-size:11px;color:{"#00d4ff" if accent else "#888"};'
            f'font-family:{_FONT};">{val}</span>'
            f'</div>'
        )

    # Intelligence bar section
    bar_track = (
        'style="width:100%;height:3px;background:rgba(255,255,255,0.06);'
        'border-radius:2px;margin-bottom:4px;"'
    )
    bar_fill = (
        f'style="width:{quality_pct:.1f}%;height:3px;'
        'background:linear-gradient(90deg,#00d4ff,#4c9eff);'
        'border-radius:2px;transition:width 0.4s ease;"'
    )
    intel_row = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;margin-bottom:6px;">'
        f'<span style="font-size:9px;letter-spacing:0.08em;color:#555;'
        f'font-family:{_FONT};text-transform:uppercase;">INTELLIGENCE</span>'
        f'<span style="font-size:11px;color:#00d4ff;font-family:{_FONT};">'
        f'{quality:.0f}  ·  {qlabel}</span>'
        f'</div>'
    )
    percentile_div = (
        f'<div style="font-size:10px;color:var(--text-3);margin-bottom:16px;">'
        f'Top {100 - pct}% of all models</div>'
    )
    intel_section = (
        f'<div>{intel_row}'
        f'<div {bar_track}><div {bar_fill}></div></div>'
        f'{percentile_div}</div>'
    )

    divider = '<div style="height:1px;background:rgba(255,255,255,0.06);margin:8px 0;"></div>'

    ctx_str = (str(row.get("context", "N/A")) or "N/A") if pd.notna(row.get("context")) else "N/A"
    price_str = f"${row['price']:.4f} / 1M tokens"

    return (
        f'<div style="color:{color};font-size:10px;font-family:{_FONT};">{provider}</div>'
        f'<div style="font-size:16px;font-family:{_FONT};color:#f2f2f2;margin-bottom:12px;">'
        f'{row["model"]}</div>'
        + intel_section
        + divider
        + _metric("Price", price_str)
        + _metric("Speed", speed_str)
        + _metric("Latency", latency_str)
        + _metric("Context", ctx_str)
    )


# ── Ported callbacks ──────────────────────────────────────────────────────────

def update_overview(providers, min_quality, search, xaxis):
    f = _apply_filters(providers, min_quality, search or "")
    fig = build_quadrant(f) if xaxis == "speed" else build_pareto_scatter(f)
    return fig.to_json()


def update_treemap(providers, min_quality, search):
    return build_treemap(_apply_filters(providers, min_quality, search or "")).to_json()


def update_provider_leaderboard(providers, min_quality, search):
    return build_provider_leaderboard(
        _apply_filters(providers, min_quality, search or "")
    ).to_json()


def update_rankings(providers, min_quality, search, sort_by):
    f = _apply_filters(providers, min_quality, search or "")
    return build_rankings(f, top_n=min(25, len(f)), metric=sort_by or "intelligence").to_json()


def update_value_leaders(providers, min_quality, search):
    return build_value_leaders(
        _apply_filters(providers, min_quality, search or "")
    ).to_json()


def update_compare(providers, min_quality, search, selected_models, triggered):
    f = _apply_filters(providers, min_quality, search or "")
    options = model_options(f)
    if triggered in ("filter-provider", "filter-quality", "model-search"):
        capped = compute_diverse5(f)
    else:
        capped = (selected_models or [])[:5]
    return json.dumps({
        "figure": json.loads(build_radar(f, capped).to_json()),
        "options": options,
        "value": capped,
        "raw_table_html": _build_raw_table_html(f, capped),
    })


def update_cost_calc(monthly_tokens_m, providers, min_quality, search):
    f = _apply_filters(providers, min_quality, search or "")
    tokens = float(monthly_tokens_m) if monthly_tokens_m else 1.0
    return build_cost_calc(f, monthly_tokens_m=tokens).to_json()


def update_table(providers, min_quality, search, sort_col, sort_dir):
    f = _apply_filters(providers, min_quality, search or "").copy()
    f["value"] = f.apply(
        lambda r: r["quality"] / r["price"] if r["price"] > 0 else None, axis=1
    )
    col = sort_col or "quality"
    asc = (sort_dir or "desc") == "asc"
    if col == "context":
        f["_ctx_k"] = f["context"].map(ctx_to_k)
        f = f.sort_values("_ctx_k", ascending=asc, na_position="last")
    else:
        f = f.sort_values(col, ascending=asc, na_position="last")
    cols = ["model", "provider", "quality", "value", "price", "speed", "latency", "context"]
    return json.dumps(f[cols].to_dict("records"))


def export_csv(providers, min_quality, search):
    return _apply_filters(providers, min_quality, search or "").to_csv(index=False)


def model_detail(model_name, provider):
    """Mirror toggle_detail_panel body as an HTML string."""
    rows = _DF[_DF["model"] == model_name]
    if rows.empty:
        return ""
    return _detail_html(rows.iloc[0], provider)
