"""
Image generation model dataset.
Primary source: live Artificial Analysis Image Arena API (data/raw/aa_image_models.csv).
Falls back to static _RAW dataset if cache not available.

ELO scores are relative quality rankings from blind human comparisons (higher = better).
Price is USD per 1,000 images at standard 1024×1024 resolution.
"""
from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).parent / "raw" / "aa_image_models.csv"

# Provider color palette — covers all providers seen in live AA data
PROVIDER_COLORS: dict[str, str] = {
    "OpenAI":                       "#34d399",  # emerald
    "Google":                       "#60a5fa",  # blue
    "Black Forest Labs":             "#f97316",  # orange
    "ByteDance Seed":               "#38bdf8",  # sky
    "Bytedance":                    "#38bdf8",  # sky (alt spelling)
    "xAI":                          "#a3e635",  # lime
    "Ideogram":                     "#f472b6",  # pink
    "Recraft":                      "#818cf8",  # indigo
    "Stability AI":                 "#c084fc",  # purple
    "Stability.ai":                 "#c084fc",  # purple (alt spelling)
    "Adobe":                        "#f87171",  # red
    "Midjourney":                   "#e2e8f0",  # light gray
    "Playground":                   "#4ade80",  # green
    "Playground AI":                "#4ade80",  # green (alt spelling)
    "Fal":                          "#fb923c",  # amber-orange
    "KlingAI":                      "#67e8f9",  # cyan
    "Leonardo AI":                  "#fbbf24",  # amber
    "Leonardo.Ai":                  "#fbbf24",  # amber (alt spelling)
    "Alibaba":                      "#a78bfa",  # violet
    "HiDream":                      "#34d399",  # emerald-alt
    "Runway":                       "#f472b6",  # pink-alt
    "Luma Labs":                    "#86efac",  # light green
    "MiniMax":                      "#7dd3fc",  # light blue
    "Sourceful":                    "#d8b4fe",  # lavender
    "Bria":                         "#fda4af",  # rose
    "Reve":                         "#6ee7b7",  # teal
    "Krea":                         "#fcd34d",  # yellow
    "ImagineArt":                   "#c4b5fd",  # light purple
    "DeepSeek":                     "#5eead4",  # turquoise
    "Microsoft Azure":              "#93c5fd",  # blue-200
    "Amazon":                       "#ff9900",  # AWS orange
    "NVIDIA":                       "#76c442",  # NVIDIA green
    "Tencent":                      "#1677ff",  # WeChat blue
    "OpenGVLab":                    "#64748b",  # slate
    "VectorSpaceLab":               "#9ca3af",  # gray
    "Pruna AI":                     "#a1a1aa",  # zinc
    "Eigen AI":                     "#71717a",  # dark gray
    "Meituan":                      "#ffb020",  # meituan yellow
    "Vidu":                         "#2dd4bf",  # teal-alt
    "Z AI":                         "#e879f9",  # fuchsia
}

DEFAULT_COLOR = "#6b7280"


# ── Static fallback dataset ───────────────────────────────────────────────────
_RAW: list[dict] = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    {"model": "GPT Image 1 (high)",   "provider": "OpenAI",
     "elo": 1268, "price_per_1k": 133.0, "gen_time_s": 22, "open_weights": False,
     "tags": ["photorealistic", "general", "text"]},
    {"model": "GPT Image 1 (medium)", "provider": "OpenAI",
     "elo": 1228, "price_per_1k": 42.0,  "gen_time_s": 14, "open_weights": False,
     "tags": ["photorealistic", "general", "text"]},
    {"model": "GPT Image 1 (low)",    "provider": "OpenAI",
     "elo": 1195, "price_per_1k": 11.0,  "gen_time_s": 8,  "open_weights": False,
     "tags": ["photorealistic", "general"]},
    {"model": "DALL-E 3 (HD)",        "provider": "OpenAI",
     "elo": 1095, "price_per_1k": 80.0,  "gen_time_s": 14, "open_weights": False,
     "tags": ["general", "artistic"]},
    {"model": "DALL-E 3",             "provider": "OpenAI",
     "elo": 1075, "price_per_1k": 40.0,  "gen_time_s": 10, "open_weights": False,
     "tags": ["general", "artistic"]},
    # ── Google ────────────────────────────────────────────────────────────────
    {"model": "Imagen 4 Ultra", "provider": "Google",
     "elo": 1173, "price_per_1k": 60.0, "gen_time_s": 26, "open_weights": False,
     "tags": ["photorealistic", "general"]},
    {"model": "Imagen 4",       "provider": "Google",
     "elo": 1138, "price_per_1k": 20.0, "gen_time_s": 14, "open_weights": False,
     "tags": ["photorealistic", "general"]},
    {"model": "Imagen 3",       "provider": "Google",
     "elo": 1112, "price_per_1k": 20.0, "gen_time_s": 16, "open_weights": False,
     "tags": ["photorealistic", "general"]},
    # ── Black Forest Labs (FLUX) ───────────────────────────────────────────────
    {"model": "FLUX.2 [max]",  "provider": "Black Forest Labs",
     "elo": 1206, "price_per_1k": 70.0, "gen_time_s": 22, "open_weights": False,
     "tags": ["photorealistic", "general", "artistic"]},
    {"model": "FLUX.2 [pro]",  "provider": "Black Forest Labs",
     "elo": 1190, "price_per_1k": 30.0, "gen_time_s": 12, "open_weights": False,
     "tags": ["photorealistic", "general", "artistic"]},
    {"model": "FLUX.1 [pro]",  "provider": "Black Forest Labs",
     "elo": 1148, "price_per_1k": 50.0, "gen_time_s": 10, "open_weights": False,
     "tags": ["photorealistic", "general", "artistic"]},
    {"model": "FLUX.1 [dev]",  "provider": "Black Forest Labs",
     "elo": 1112, "price_per_1k": 25.0, "gen_time_s": 12, "open_weights": True,
     "tags": ["photorealistic", "general", "artistic"]},
    {"model": "FLUX.1 [schnell]", "provider": "Black Forest Labs",
     "elo": 1068, "price_per_1k": 3.0,  "gen_time_s": 2,  "open_weights": True,
     "tags": ["general", "fast"]},
    # ── Ideogram ──────────────────────────────────────────────────────────────
    {"model": "Ideogram 3.0", "provider": "Ideogram",
     "elo": 1155, "price_per_1k": 80.0, "gen_time_s": 8, "open_weights": False,
     "tags": ["text", "artistic", "general"]},
    {"model": "Ideogram 2.0", "provider": "Ideogram",
     "elo": 1118, "price_per_1k": 40.0, "gen_time_s": 6, "open_weights": False,
     "tags": ["text", "artistic", "general"]},
    # ── Midjourney ────────────────────────────────────────────────────────────
    {"model": "Midjourney v7",   "provider": "Midjourney",
     "elo": 1198, "price_per_1k": 10.0, "gen_time_s": 30, "open_weights": False,
     "tags": ["artistic", "photorealistic", "general"]},
    {"model": "Midjourney v6.1", "provider": "Midjourney",
     "elo": 1162, "price_per_1k": 10.0, "gen_time_s": 25, "open_weights": False,
     "tags": ["artistic", "photorealistic"]},
    # ── Stability AI ──────────────────────────────────────────────────────────
    {"model": "Stable Diffusion 3.5 Large", "provider": "Stability AI",
     "elo": 1082, "price_per_1k": 13.0, "gen_time_s": 5, "open_weights": True,
     "tags": ["photorealistic", "artistic", "general"]},
]


# ── Tag derivation from category ELOs ────────────────────────────────────────

def _derive_tags(df: pd.DataFrame) -> pd.Series:
    """
    Assign capability tags based on per-category ELO rankings.
    A model gets a tag if its category ELO is at or above the category median.
    """
    tag_rules = [
        ("photorealistic", "elo_general_photorealistic"),
        ("artistic",       "elo_cartoon_illustration"),
        ("text",           "elo_text_typography"),
    ]

    medians: dict[str, float] = {}
    for tag, col in tag_rules:
        if col in df.columns:
            medians[tag] = df[col].median()

    def _row_tags(row) -> list[str]:
        tags = ["general"]
        for tag, col in tag_rules:
            if tag in medians:
                val = row.get(col)
                if pd.notna(val) and val >= medians[tag]:
                    tags.append(tag)
        if row.get("open_weights"):
            tags.append("open_weights")
        return tags

    return df.apply(_row_tags, axis=1)


# ── Public API ────────────────────────────────────────────────────────────────

def get_image_df() -> pd.DataFrame:
    """Load image model data. Uses live CSV cache when available."""
    if _CACHE.exists():
        df = pd.read_csv(_CACHE)
        df["open_weights"]  = df["open_weights"].fillna(False).astype(bool)
        df["price_per_1k"]  = df["price_per_1k"].fillna(0.0)
        df["tags"]          = _derive_tags(df)
        df["tags_str"]      = df["tags"].apply(", ".join)
        # gen_time_s is not in live data — use 0 as placeholder
        if "gen_time_s" not in df.columns:
            df["gen_time_s"] = 0
        return df

    # Fallback to static _RAW
    df = pd.DataFrame(_RAW)
    df["tags_str"] = df["tags"].apply(", ".join)
    return df


def get_image_providers() -> list[str]:
    if _CACHE.exists():
        df = pd.read_csv(_CACHE, usecols=["provider"])
        return sorted(df["provider"].dropna().unique().tolist())
    return sorted({r["provider"] for r in _RAW})
