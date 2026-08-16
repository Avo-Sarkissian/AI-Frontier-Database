"""Video generation catalogue, read from the live Artificial Analysis scrape.

This file used to BE the dataset: a literal list of 20 models, hand-typed, last
touched 2026-03-13, with "approximate" 0-100 quality scores that had no source
and yet drove a Pareto frontier. It is now a loader over
data/raw/aa_video_models.csv, which data/video_scraper.py refreshes hourly from
the AA Video Arena. See that module's docstring for what the arena publishes.

TWO MODES, NOT ONE DATASET
--------------------------
AA scores text-to-video and image-to-video in separate Elo pools and never
compares across them, and a model's price differs between them (Vidu Q3 Pro is
$6/min from a prompt and $9.60/min from a still). There is therefore no single
"video quality" or "video price" to publish. ``get_video_df(mode)`` projects one
arena at a time onto the shared column names the charts read — `elo`,
`price_per_min`, `tags` — and a model absent from that arena is absent from the
frame, because it was never ranked there. That is a different statement from
"scored zero", and the difference is the whole reason this file no longer
invents numbers.

WHAT AN ELO IS HERE
-------------------
Bradley-Terry maximum likelihood over pairwise human votes, rescaled by AA to an
Elo-like range and recomputed hourly. It is an interval scale with no meaningful
zero, which is why components/charts/video_chart.py plots position, not bar
length — a bar drawn from zero would render the 940-1330 spread as 89 identical
full-width bars.

RETIRED VARIANTS
----------------
AA keeps preview and dated builds ranked after they are superseded and flags the
survivors `isCurrent`. The default frame is current-only, because three of the
top twenty text-to-video entries are superseded Veo previews and a buyer's
leaderboard that leads with withdrawn SKUs is answering the wrong question. It
is a disclosed narrowing, not a silent one: the tab caption says so, the chart
subtitle counts what is shown, ``include_retired=True`` returns everything, and
the CSV export carries every row with its `is_current` flag intact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from components.charts.constants import (
    PROVIDER_COLORS as _SHARED_COLORS, DEFAULT_COLOR as _SHARED_DEFAULT,
)

_CACHE = Path(__file__).parent / "raw" / "aa_video_models.csv"


# ── Modes ─────────────────────────────────────────────────────────────────────
# `value` is the column suffix the scraper writes (elo_t2v, price_per_min_i2v).
# Keep in sync with data/video_scraper._MODES.
VIDEO_MODES: list[dict] = [
    {"value": "t2v", "label": "Text to video"},
    {"value": "i2v", "label": "Image to video"},
]

DEFAULT_MODE = "t2v"

_MODE_VALUES = {m["value"] for m in VIDEO_MODES}


def mode_label(value: str) -> str:
    for m in VIDEO_MODES:
        if m["value"] == value:
            return m["label"]
    return value


def _resolve_mode(mode: str | None) -> str:
    """Unknown modes fall back rather than raising.

    ``mode`` arrives from a URL query string and a browser <select>, so it is
    user input. An unrecognised value must not KeyError inside Pyodide, where
    the only symptom is a chart that never renders.
    """
    return mode if mode in _MODE_VALUES else DEFAULT_MODE


# ── Palette ───────────────────────────────────────────────────────────────────
# Labs that also ship LLMs must wear the same colour they wear everywhere else --
# Google was #60a5fa here and amber on Overview, which is the inconsistency the
# Run Local palette had. Video-only studios keep their own hues, since they have
# no entry in the shared provider palette to inherit.
#
# The live arena carries 31 creators, 11 of which already sit in the shared
# palette. Thirty-one mutually separable hues do not exist, and this file does
# not pretend otherwise: the hues below were assigned by farthest-point search in
# CIE L*a*b*, maximising the minimum distance to every colour that can appear
# beside them on this tab. Every assignment here lands >= 13.9 dE from its
# nearest neighbour and >= 3.0:1 contrast on the #111111 surface. The one pair
# under that bar is OpenAI/Tencent at 4.1 dE, which is inherited from the shared
# palette and not ours to move.
#
# Because 31 is past what colour alone can carry, the two charts do not lean on
# it: the ranked view labels every row with its model name and the scatter gives
# legend entries only to the densest providers, bucketing the tail into a single
# grey "Other" (see components/charts/video_chart.py). Colour groups here; it
# never has to identify on its own.
_VIDEO_ONLY_COLORS: dict[str, str] = {
    "Alibaba-ATH":    "#a8a29e",
    "ByteDance Seed": "#2dd4bf",
    "Genmo":          "#a3e635",
    "Haiper":         "#f87171",
    "HiDream":        "#a5b4fc",
    "KlingAI":        "#0ea5e9",
    "Krea":           "#fdba74",
    "Leonardo.Ai":    "#818cf8",
    "Lightricks":     "#fca5a5",
    "Luma Labs":      "#c084fc",
    "Midjourney":     "#f9a8d4",
    "Moonvalley":     "#67e8f9",
    "Open Source":    "#cbd5e1",
    "Pika Art":       "#facc15",
    "PixVerse":       "#fb923c",
    "Pruna AI":       "#94a3b8",
    "Runway":         "#f472b6",
    "Skywork AI":     "#bef264",
    "Video Rebirth":  "#fcd34d",
    "Vidu":           "#93c5fd",
}

PROVIDER_COLORS: dict[str, str] = {**_VIDEO_ONLY_COLORS, **_SHARED_COLORS}

DEFAULT_COLOR = _SHARED_DEFAULT


# ── Tag vocabulary ────────────────────────────────────────────────────────────
# AA scores every model on 29 categories. Offering all of them would be a
# dropdown where each option selects roughly half the catalogue and any two
# AND-ed select almost nothing. These seven are the ones that decide a real
# choice between video models — the two style families people actually shop for,
# the three subjects every video model is known to fail at, and the two prompt
# formats that separate a clip generator from a shot generator.
#
# Selected, but not hardcoded: each entry names an AA category slug and is
# emitted only if that column exists and carries values in the frame. If AA
# retires a category the tag disappears on its own, which is the failure
# data/image_models.py:204-221 records — a "Fast" option was offered for months
# against a vocabulary that could never produce it, so it matched zero models
# and the UI blamed the user.
_TAG_CATEGORIES: list[tuple[str, str]] = [
    ("photorealistic",    "Photorealistic"),
    ("cartoon-and-anime", "Cartoon & anime"),
    ("physics",           "Physics"),
    ("people",            "People"),
    ("text",              "Text in frame"),
    ("moving-camera",     "Moving camera"),
    ("multi-scene",       "Multi-scene"),
]

# Capability tags. These are facts about the model, not a comparison against
# other models, so they are not median-derived.
_CAPABILITY_LABELS = {
    "open-weights": "Open weights",
    "audio":        "Generates audio",
}

_TAG_LABELS = {**dict(_TAG_CATEGORIES), **_CAPABILITY_LABELS}


def tag_label(slug: str) -> str:
    return _TAG_LABELS.get(slug, slug.replace("_", " ").replace("-", " ").title())


def _category_column(mode: str, category: str) -> str:
    """Mirror of data/video_scraper._col — 'physics' → 'elo_t2v_physics'."""
    return f"elo_{mode}_" + category.replace("-", "_")


def _derive_tags(df: pd.DataFrame, mode: str) -> pd.Series:
    """A model carries a category tag when it beats the field on that category.

    "At or above the median of the models AA scored on this category", the same
    rule data/image_models.py uses. It is a claim about relative standing, which
    is the only claim an Elo supports — nothing here asserts that a model is
    *good* at people, only that it is in the better half.
    """
    medians: dict[str, float] = {}
    for category, _label in _TAG_CATEGORIES:
        col = _category_column(mode, category)
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                medians[category] = float(series.median())

    def _row_tags(row) -> list[str]:
        tags: list[str] = []
        for category, _label in _TAG_CATEGORIES:
            if category not in medians:
                continue
            val = pd.to_numeric(pd.Series([row.get(_category_column(mode, category))]),
                                errors="coerce").iloc[0]
            if pd.notna(val) and val >= medians[category]:
                tags.append(category)
        if row.get("open_weights"):
            tags.append("open-weights")
        if row.get("audio"):
            tags.append("audio")
        return tags

    return df.apply(_row_tags, axis=1)


# ── Public API ────────────────────────────────────────────────────────────────

# The shape every consumer sees, so an absent cache produces an empty frame with
# the right columns rather than a KeyError three call-frames away. There is no
# hand-written fallback dataset on purpose: a stale literal masquerading as live
# data is the exact defect this rewrite removes, and an empty tab under a
# working freshness badge is the honest failure.
_COLUMNS = [
    "model", "slug", "provider", "family", "release_date", "is_current",
    "open_weights", "audio", "elo", "price_per_min", "price_per_min_audio",
    "gen_time_s", "gen_time_host", "tags", "tags_str",
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _COLUMNS})


def load_raw() -> pd.DataFrame | None:
    """The committed catalogue exactly as scraped — every mode, every column."""
    if not _CACHE.exists():
        return None
    df = pd.read_csv(_CACHE)
    for col in ("open_weights", "audio", "is_current"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def get_video_df(mode: str | None = DEFAULT_MODE,
                 include_retired: bool = False) -> pd.DataFrame:
    """One arena, projected onto the column names the charts read.

    Rows are the models AA has actually ranked in ``mode``; `elo` and
    `price_per_min` come from that arena. A null price stays null — see
    data/video_scraper._price for why zero-filling it is not an option.
    """
    mode = _resolve_mode(mode)
    raw = load_raw()
    if raw is None or raw.empty:
        return _empty_frame()

    elo_col   = f"elo_{mode}"
    price_col = f"price_per_min_{mode}"
    if elo_col not in raw.columns:
        return _empty_frame()

    df = raw[pd.to_numeric(raw[elo_col], errors="coerce").notna()].copy()
    if not include_retired and "is_current" in df.columns:
        current = df[df["is_current"]]
        # Never hand back nothing because every model happens to be flagged
        # retired — that would read as a broken tab rather than as a filter.
        if not current.empty:
            df = current

    df["elo"] = pd.to_numeric(df[elo_col], errors="coerce")
    df["price_per_min"] = (pd.to_numeric(df[price_col], errors="coerce")
                           if price_col in df.columns else pd.NA)
    df["tags"] = _derive_tags(df, mode)
    df["tags_str"] = df["tags"].apply(lambda t: ", ".join(t) if t else "—")
    return df.sort_values("elo", ascending=False).reset_index(drop=True)


def filter_video_df(df: pd.DataFrame, providers=None, tags=None) -> pd.DataFrame:
    """The tab's filter semantics, defined once.

    app.py and static_api.py render the same tab through different runtimes, and
    every place this project has let them each spell a filter out has drifted —
    the two still disagree on whether the open-weights tag slug uses a hyphen or
    an underscore (static_api.py:371 vs :389). Both now call this.

    Tags are AND-ed: a model must earn every selected tag.
    """
    if df is None or df.empty:
        return df
    out = df
    if providers:
        out = out[out["provider"].isin(list(providers))]
    for tag in (tags or []):
        if tag == "open-weights":
            out = out[out["open_weights"] == True]        # noqa: E712 — pandas mask
        elif tag == "audio":
            out = out[out["audio"] == True]               # noqa: E712 — pandas mask
        else:
            out = out[out["tags"].apply(
                lambda t, tag=tag: isinstance(t, (list, tuple)) and tag in t)]
    return out


def get_video_tags(mode: str | None = None) -> list[dict]:
    """Tag options that mean something in EVERY mode.

    Built from the vocabulary the pipeline actually emits, and intersected
    across modes: an option the mode switch can silently turn into a dead filter
    is the same defect as one that was never derivable at all. A tag carried by
    every row is dropped too — a filter that promises to narrow the view and
    cannot is the mirror image of one that matches nothing.
    """
    from collections import Counter

    modes = [_resolve_mode(mode)] if mode else [m["value"] for m in VIDEO_MODES]
    usable: set[str] | None = None
    for m in modes:
        df = get_video_df(m)
        total = len(df)
        counts = Counter(
            t for tags in df["tags"] if isinstance(tags, list) for t in tags
        )
        live = {t for t, n in counts.items() if 0 < n < total}
        usable = live if usable is None else (usable & live)

    usable = usable or set()
    return [{"label": tag_label(t), "value": t}
            for t in sorted(usable, key=lambda t: tag_label(t).lower())]


def get_video_providers() -> list[str]:
    """Every provider that appears in any mode, so the control never goes stale
    when the mode switches under it."""
    raw = load_raw()
    if raw is None or raw.empty or "provider" not in raw.columns:
        return []
    return sorted(raw["provider"].dropna().unique().tolist())
