"""Image Gen and Video Gen chart encodings (2026-10-29 live audit).

- The video Cost vs Arena Elo subtitle and legend promised a dotted Pareto
  line on every render, but text-to-video's frontier is ONE model (HeyGen
  Video 1 is both the cheapest and the best-rated), so no line was ever drawn
  under a legend entry for one. The legend and subtitle now describe what is
  drawn: a line for two or more points, a ring for one, nothing for none.
- The cheapest video model sat on the y axis: the log x range is padded.
- The frontier labels were mode="text", centred ON their marker.
- The ranked video chart's right gutter took 140 of a phone's 356 px.
- Image Gen drew ELO as bars from a ~600 baseline on a different range per
  facet (pinned in test_encoding_calibration: dots, one shared scale).
"""
import math

import pandas as pd
import pytest

from components.charts.image_scatter import build_image_faceted
from components.charts.video_chart import build_video_rankings, build_video_scatter
from data.image_models import get_image_df
from data.video_models import get_video_df


def _video(rows):
    return pd.DataFrame([
        dict(model=m, provider=p, elo=e, price_per_min=c, open_weights=False,
             audio=False, tags_str="", gen_time_s=None)
        for m, p, e, c in rows
    ])


def _frontier(fig):
    return [t for t in fig.data if t.name == "Pareto Frontier"]


def _subtitle(fig):
    return fig.layout.title.text


def test_a_one_model_frontier_is_ringed_not_promised_as_a_line():
    df = _video([("Best", "A", 1200, 1.0), ("Mid", "B", 1100, 3.0), ("Low", "C", 1000, 9.0)])
    fig = build_video_scatter(df, full_df=df)
    (front,) = _frontier(fig)
    assert front.mode == "markers", "a single frontier point drawn as a 'line'"
    assert "circle-open" in front.marker.symbol
    assert "dotted line" not in _subtitle(fig)
    assert "ringed" in _subtitle(fig)


def test_a_real_frontier_is_a_dotted_line_and_says_so():
    df = _video([("Cheap", "A", 1000, 1.0), ("Mid", "B", 1100, 3.0), ("Top", "C", 1200, 9.0),
                 ("Dominated", "D", 1050, 9.5)])
    fig = build_video_scatter(df, full_df=df)
    (front,) = _frontier(fig)
    assert front.mode == "lines" and len(front.x) == 3
    assert "dotted line" in _subtitle(fig)


def test_a_filter_that_hides_the_frontier_promises_nothing():
    full = _video([("Cheap", "A", 1000, 1.0), ("Top", "C", 1200, 9.0),
                   ("Dominated", "D", 1050, 9.5)])
    fig = build_video_scatter(full[full["model"] == "Dominated"], full_df=full)
    assert not _frontier(fig)
    assert "dotted line" not in _subtitle(fig) and "ringed" not in _subtitle(fig)


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
def test_no_video_marker_sits_on_an_axis(mode):
    """Autorange put the cheapest model's centre a few px from the y axis."""
    vdf = get_video_df(mode)
    paid = vdf[vdf["price_per_min"] > 0]
    fig = build_video_scatter(paid, full_df=paid, mode=mode)
    lo, hi = fig.layout.xaxis.range
    logs = [math.log10(v) for v in paid["price_per_min"]]
    span = hi - lo
    assert min(logs) - lo >= 0.04 * span, "cheapest marker on the y axis"
    assert hi - max(logs) >= 0.04 * span, "dearest marker on the right edge"


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
def test_frontier_labels_are_offset_from_their_markers(mode):
    """mode="text" centres the label on the point, i.e. over its marker; a
    transparent marker gives the text Plotly's marker-size offset."""
    vdf = get_video_df(mode)
    paid = vdf[vdf["price_per_min"] > 0]
    fig = build_video_scatter(paid, full_df=paid, mode=mode)
    labels = [t for t in fig.data if t.text is not None and t.hoverinfo == "skip"]
    assert labels, "frontier labels vanished"
    for t in labels:
        assert t.mode == "markers+text"
        assert all(p != "middle center" for p in t.textposition)


def test_the_ranked_video_chart_has_a_phone_layout():
    """At 356 px the 'price · time · open' gutter took 140 px and left the
    dots ~60 px under rotated ticks. The phone patch keeps the price alone."""
    vdf = get_video_df()
    fig = build_video_rankings(vdf, full_df=vdf)
    narrow = fig.layout.meta["narrow"]
    assert narrow["max_width"] >= 500
    lay = narrow["layout"]
    assert lay["margin"]["r"] <= 60
    texts = [a["text"] for a in lay["annotations"]]
    assert len(texts) == len(fig.layout.annotations)
    assert all("·" not in t for t in texts), "phone labels still carry the long form"
    assert max(len(t) for t in lay["yaxis"]["ticktext"]) <= 18


def test_image_facets_stack_on_a_phone():
    """Three facets side by side left each ~70 px on a phone; there they stack
    one under another at full width, without overlapping domains."""
    img = get_image_df()
    fig = build_image_faceted(img, full_df=img)
    lay = fig.layout.meta["narrow"]["layout"]
    doms = [lay[k]["domain"] for k in ("yaxis", "yaxis2", "yaxis3")]
    for k in ("xaxis", "xaxis2", "xaxis3"):
        assert lay[k]["domain"] == [0, 1]
    assert doms[0][0] > doms[1][1] and doms[1][0] > doms[2][1], doms
    assert all(0 <= d[0] < d[1] <= 1 for d in doms)
    assert lay["height"] > fig.layout.height


def test_image_open_weights_are_hollow():
    img = get_image_df()
    fig = build_image_faceted(img[img["open_weights"].eq(True)], full_df=img)
    syms = [s for t in fig.data for s in (t.marker.symbol or [])]
    assert syms and all(s == "circle-open" for s in syms)
    assert "hollow = open weights" in fig.layout.title.text
