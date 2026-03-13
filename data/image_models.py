"""
Image generation model dataset.
Metrics sourced from Artificial Analysis Image Arena (ELO) and public pricing pages.
ELO scores are relative quality rankings from blind human comparisons (higher = better).
Price is USD per 1,000 images at standard 1024×1024 resolution.
Gen time is median seconds to generate one image.
"""
import pandas as pd

# Provider color palette (image-gen specific providers + overlap with LLM tab)
PROVIDER_COLORS: dict[str, str] = {
    "OpenAI":             "#34d399",  # emerald
    "Google":             "#60a5fa",  # blue
    "Black Forest Labs":  "#f97316",  # orange
    "ByteDance Seed":     "#38bdf8",  # sky
    "xAI":                "#a3e635",  # lime
    "Ideogram":           "#f472b6",  # pink
    "Recraft":            "#818cf8",  # indigo
    "Stability AI":       "#c084fc",  # purple
    "Adobe":              "#f87171",  # red
    "Midjourney":         "#e2e8f0",  # light gray
    "Playground":         "#4ade80",  # green
    "Fal":                "#fb923c",  # amber-orange
    "Kling":              "#67e8f9",  # cyan
    "Leonardo AI":        "#fbbf24",  # amber
}

DEFAULT_COLOR = "#6b7280"

# ── Dataset ────────────────────────────────────────────────────────────────────
# Fields:
#   model         — display name
#   provider      — company
#   elo           — AA Image Arena ELO score (approx)
#   price_per_1k  — USD per 1,000 images
#   gen_time_s    — median seconds per image (1024×1024)
#   open_weights  — True if weights are publicly available
#   tags          — list of capability tags

_RAW: list[dict] = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    {
        "model": "GPT Image 1 (high)",
        "provider": "OpenAI",
        "elo": 1268, "price_per_1k": 133.0, "gen_time_s": 22,
        "open_weights": False,
        "tags": ["photorealistic", "general", "text"],
    },
    {
        "model": "GPT Image 1 (medium)",
        "provider": "OpenAI",
        "elo": 1228, "price_per_1k": 42.0, "gen_time_s": 14,
        "open_weights": False,
        "tags": ["photorealistic", "general", "text"],
    },
    {
        "model": "GPT Image 1 (low)",
        "provider": "OpenAI",
        "elo": 1195, "price_per_1k": 11.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    {
        "model": "DALL-E 3 (HD)",
        "provider": "OpenAI",
        "elo": 1095, "price_per_1k": 80.0, "gen_time_s": 14,
        "open_weights": False,
        "tags": ["general", "artistic"],
    },
    {
        "model": "DALL-E 3",
        "provider": "OpenAI",
        "elo": 1075, "price_per_1k": 40.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["general", "artistic"],
    },
    # ── Google ────────────────────────────────────────────────────────────────
    {
        "model": "Gemini 3.1 Flash Image",
        "provider": "Google",
        "elo": 1261, "price_per_1k": 67.0, "gen_time_s": 14,
        "open_weights": False,
        "tags": ["photorealistic", "general", "editing"],
    },
    {
        "model": "Gemini 3 Pro Image",
        "provider": "Google",
        "elo": 1217, "price_per_1k": 134.0, "gen_time_s": 22,
        "open_weights": False,
        "tags": ["photorealistic", "general", "editing"],
    },
    {
        "model": "Imagen 4 Ultra",
        "provider": "Google",
        "elo": 1173, "price_per_1k": 60.0, "gen_time_s": 26,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    {
        "model": "Gemini 2.5 Flash Image",
        "provider": "Google",
        "elo": 1164, "price_per_1k": 39.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["photorealistic", "general", "editing"],
    },
    {
        "model": "Imagen 4",
        "provider": "Google",
        "elo": 1138, "price_per_1k": 20.0, "gen_time_s": 14,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    {
        "model": "Imagen 3",
        "provider": "Google",
        "elo": 1112, "price_per_1k": 20.0, "gen_time_s": 16,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    # ── Black Forest Labs (FLUX) ───────────────────────────────────────────────
    {
        "model": "FLUX.2 [max]",
        "provider": "Black Forest Labs",
        "elo": 1206, "price_per_1k": 70.0, "gen_time_s": 22,
        "open_weights": False,
        "tags": ["photorealistic", "general", "artistic"],
    },
    {
        "model": "FLUX.2 [pro]",
        "provider": "Black Forest Labs",
        "elo": 1190, "price_per_1k": 30.0, "gen_time_s": 12,
        "open_weights": False,
        "tags": ["photorealistic", "general", "artistic"],
    },
    {
        "model": "FLUX.2 [flex]",
        "provider": "Black Forest Labs",
        "elo": 1179, "price_per_1k": 60.0, "gen_time_s": 16,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    {
        "model": "FLUX.2 [dev] Turbo",
        "provider": "Black Forest Labs",
        "elo": 1166, "price_per_1k": 8.0, "gen_time_s": 4,
        "open_weights": True,
        "tags": ["photorealistic", "general", "fast"],
    },
    {
        "model": "FLUX.1 [pro]",
        "provider": "Black Forest Labs",
        "elo": 1148, "price_per_1k": 50.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["photorealistic", "general", "artistic"],
    },
    {
        "model": "FLUX.1 [dev]",
        "provider": "Black Forest Labs",
        "elo": 1112, "price_per_1k": 25.0, "gen_time_s": 12,
        "open_weights": True,
        "tags": ["photorealistic", "general", "artistic"],
    },
    {
        "model": "FLUX.1 [schnell]",
        "provider": "Black Forest Labs",
        "elo": 1068, "price_per_1k": 3.0, "gen_time_s": 2,
        "open_weights": True,
        "tags": ["general", "fast"],
    },
    # ── ByteDance Seed ────────────────────────────────────────────────────────
    {
        "model": "Seedream 4.5",
        "provider": "ByteDance Seed",
        "elo": 1172, "price_per_1k": 40.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["photorealistic", "general", "artistic"],
    },
    {
        "model": "Seedream 4.0",
        "provider": "ByteDance Seed",
        "elo": 1187, "price_per_1k": 30.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    {
        "model": "Seedream 3.0",
        "provider": "ByteDance Seed",
        "elo": 1150, "price_per_1k": 3.0, "gen_time_s": 6,
        "open_weights": False,
        "tags": ["general", "artistic"],
    },
    # ── xAI ───────────────────────────────────────────────────────────────────
    {
        "model": "Grok Image",
        "provider": "xAI",
        "elo": 1166, "price_per_1k": 20.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["photorealistic", "general"],
    },
    # ── Ideogram ──────────────────────────────────────────────────────────────
    {
        "model": "Ideogram 3.0",
        "provider": "Ideogram",
        "elo": 1155, "price_per_1k": 80.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["text", "artistic", "general"],
    },
    {
        "model": "Ideogram 2.0",
        "provider": "Ideogram",
        "elo": 1118, "price_per_1k": 40.0, "gen_time_s": 6,
        "open_weights": False,
        "tags": ["text", "artistic", "general"],
    },
    # ── Recraft ───────────────────────────────────────────────────────────────
    {
        "model": "Recraft V4 Pro",
        "provider": "Recraft",
        "elo": 1160, "price_per_1k": 80.0, "gen_time_s": 12,
        "open_weights": False,
        "tags": ["photorealistic", "artistic", "vector"],
    },
    {
        "model": "Recraft V4",
        "provider": "Recraft",
        "elo": 1142, "price_per_1k": 40.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["photorealistic", "artistic", "vector"],
    },
    {
        "model": "Recraft V3",
        "provider": "Recraft",
        "elo": 1102, "price_per_1k": 40.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["artistic", "vector"],
    },
    # ── Stability AI ──────────────────────────────────────────────────────────
    {
        "model": "Stable Diffusion 3.5 Large",
        "provider": "Stability AI",
        "elo": 1082, "price_per_1k": 13.0, "gen_time_s": 5,
        "open_weights": True,
        "tags": ["photorealistic", "artistic", "general"],
    },
    {
        "model": "Stable Diffusion 3.5 Medium",
        "provider": "Stability AI",
        "elo": 1052, "price_per_1k": 3.0, "gen_time_s": 3,
        "open_weights": True,
        "tags": ["general", "fast"],
    },
    {
        "model": "Stable Image Ultra",
        "provider": "Stability AI",
        "elo": 1105, "price_per_1k": 80.0, "gen_time_s": 10,
        "open_weights": False,
        "tags": ["photorealistic", "artistic"],
    },
    # ── Midjourney ────────────────────────────────────────────────────────────
    {
        "model": "Midjourney v7",
        "provider": "Midjourney",
        "elo": 1198, "price_per_1k": 10.0, "gen_time_s": 30,
        "open_weights": False,
        "tags": ["artistic", "photorealistic", "general"],
    },
    {
        "model": "Midjourney v6.1",
        "provider": "Midjourney",
        "elo": 1162, "price_per_1k": 10.0, "gen_time_s": 25,
        "open_weights": False,
        "tags": ["artistic", "photorealistic"],
    },
    # ── Adobe ─────────────────────────────────────────────────────────────────
    {
        "model": "Firefly Image 4 Ultra",
        "provider": "Adobe",
        "elo": 1120, "price_per_1k": 100.0, "gen_time_s": 12,
        "open_weights": False,
        "tags": ["photorealistic", "artistic", "commercial"],
    },
    {
        "model": "Firefly Image 4",
        "provider": "Adobe",
        "elo": 1098, "price_per_1k": 50.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["photorealistic", "artistic", "commercial"],
    },
    # ── Playground ────────────────────────────────────────────────────────────
    {
        "model": "Playground v3",
        "provider": "Playground",
        "elo": 1108, "price_per_1k": 40.0, "gen_time_s": 6,
        "open_weights": False,
        "tags": ["artistic", "general"],
    },
    # ── Kling ─────────────────────────────────────────────────────────────────
    {
        "model": "Kling Image 1.5 Pro",
        "provider": "Kling",
        "elo": 1135, "price_per_1k": 28.0, "gen_time_s": 8,
        "open_weights": False,
        "tags": ["photorealistic", "general", "artistic"],
    },
]


def get_image_df() -> pd.DataFrame:
    df = pd.DataFrame(_RAW)
    df["tags_str"] = df["tags"].apply(lambda t: ", ".join(t))
    return df


def get_image_providers() -> list[str]:
    return sorted({r["provider"] for r in _RAW})
