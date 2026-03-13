"""
Video generation model dataset.
Quality scores are human preference ratings (0-100 scale, approximate).
Price is USD per second of generated video.
Gen time is approximate seconds to generate a ~5s clip.
Sources: public pricing pages, Artificial Analysis, EvalVid, community benchmarks.
"""
import pandas as pd

PROVIDER_COLORS: dict[str, str] = {
    "Google":     "#60a5fa",
    "OpenAI":     "#34d399",
    "Runway":     "#f472b6",
    "Kuaishou":   "#fb923c",
    "Luma AI":    "#c084fc",
    "MiniMax":    "#86efac",
    "Pika":       "#facc15",
    "Alibaba":    "#38bdf8",
    "Zhipu AI":   "#818cf8",
    "Genmo":      "#a3e635",
    "Stability AI": "#a78bfa",
    "ByteDance":  "#67e8f9",
    "Lightricks": "#fca5a5",
}

DEFAULT_COLOR = "#6b7280"

_RAW: list[dict] = [
    # ── Google ────────────────────────────────────────────────────────────────
    {
        "model": "Veo 2",
        "provider": "Google",
        "quality": 90, "price_per_sec": 0.035, "gen_time_s": 120,
        "max_res": "4K", "max_duration_s": 8, "open_weights": False,
        "tags": ["cinematic", "realistic", "high-res"],
    },
    {
        "model": "Veo 3",
        "provider": "Google",
        "quality": 93, "price_per_sec": 0.050, "gen_time_s": 150,
        "max_res": "4K", "max_duration_s": 8, "open_weights": False,
        "tags": ["cinematic", "realistic", "high-res", "audio"],
    },
    # ── OpenAI ────────────────────────────────────────────────────────────────
    {
        "model": "Sora HD",
        "provider": "OpenAI",
        "quality": 88, "price_per_sec": 0.025, "gen_time_s": 180,
        "max_res": "1080p", "max_duration_s": 20, "open_weights": False,
        "tags": ["cinematic", "realistic"],
    },
    {
        "model": "Sora Turbo",
        "provider": "OpenAI",
        "quality": 80, "price_per_sec": 0.010, "gen_time_s": 60,
        "max_res": "720p", "max_duration_s": 20, "open_weights": False,
        "tags": ["realistic", "fast"],
    },
    # ── Kuaishou ──────────────────────────────────────────────────────────────
    {
        "model": "Kling 1.6 Pro",
        "provider": "Kuaishou",
        "quality": 86, "price_per_sec": 0.014, "gen_time_s": 90,
        "max_res": "1080p", "max_duration_s": 10, "open_weights": False,
        "tags": ["cinematic", "realistic"],
    },
    {
        "model": "Kling 1.6 Standard",
        "provider": "Kuaishou",
        "quality": 78, "price_per_sec": 0.007, "gen_time_s": 60,
        "max_res": "720p", "max_duration_s": 10, "open_weights": False,
        "tags": ["realistic", "fast"],
    },
    # ── Runway ────────────────────────────────────────────────────────────────
    {
        "model": "Gen-3 Alpha",
        "provider": "Runway",
        "quality": 83, "price_per_sec": 0.025, "gen_time_s": 60,
        "max_res": "1080p", "max_duration_s": 10, "open_weights": False,
        "tags": ["cinematic", "artistic"],
    },
    {
        "model": "Gen-3 Alpha Turbo",
        "provider": "Runway",
        "quality": 74, "price_per_sec": 0.005, "gen_time_s": 20,
        "max_res": "720p", "max_duration_s": 10, "open_weights": False,
        "tags": ["fast", "artistic"],
    },
    {
        "model": "Gen-4",
        "provider": "Runway",
        "quality": 87, "price_per_sec": 0.030, "gen_time_s": 90,
        "max_res": "1080p", "max_duration_s": 16, "open_weights": False,
        "tags": ["cinematic", "artistic", "realistic"],
    },
    # ── Luma AI ───────────────────────────────────────────────────────────────
    {
        "model": "Dream Machine 1.6",
        "provider": "Luma AI",
        "quality": 80, "price_per_sec": 0.004, "gen_time_s": 60,
        "max_res": "720p", "max_duration_s": 9, "open_weights": False,
        "tags": ["realistic", "fast"],
    },
    {
        "model": "Ray 2",
        "provider": "Luma AI",
        "quality": 85, "price_per_sec": 0.016, "gen_time_s": 90,
        "max_res": "1080p", "max_duration_s": 9, "open_weights": False,
        "tags": ["cinematic", "realistic"],
    },
    # ── MiniMax ───────────────────────────────────────────────────────────────
    {
        "model": "Hailuo AI Video-01",
        "provider": "MiniMax",
        "quality": 79, "price_per_sec": 0.005, "gen_time_s": 60,
        "max_res": "1080p", "max_duration_s": 6, "open_weights": False,
        "tags": ["realistic", "fast"],
    },
    # ── Pika ──────────────────────────────────────────────────────────────────
    {
        "model": "Pika 2.1",
        "provider": "Pika",
        "quality": 76, "price_per_sec": 0.008, "gen_time_s": 45,
        "max_res": "1080p", "max_duration_s": 5, "open_weights": False,
        "tags": ["artistic", "fast"],
    },
    # ── ByteDance ─────────────────────────────────────────────────────────────
    {
        "model": "MagicVideo-V2",
        "provider": "ByteDance",
        "quality": 75, "price_per_sec": 0.006, "gen_time_s": 60,
        "max_res": "720p", "max_duration_s": 5, "open_weights": False,
        "tags": ["realistic"],
    },
    # ── Lightricks ────────────────────────────────────────────────────────────
    {
        "model": "LTX Video 0.9.7",
        "provider": "Lightricks",
        "quality": 72, "price_per_sec": 0.002, "gen_time_s": 15,
        "max_res": "720p", "max_duration_s": 5, "open_weights": True,
        "tags": ["fast", "open-weights"],
    },
    # ── Alibaba ───────────────────────────────────────────────────────────────
    {
        "model": "Wan 2.1 (14B)",
        "provider": "Alibaba",
        "quality": 74, "price_per_sec": 0.003, "gen_time_s": 120,
        "max_res": "720p", "max_duration_s": 10, "open_weights": True,
        "tags": ["open-weights", "realistic"],
    },
    {
        "model": "Wan 2.1 (1.3B)",
        "provider": "Alibaba",
        "quality": 64, "price_per_sec": 0.001, "gen_time_s": 30,
        "max_res": "480p", "max_duration_s": 5, "open_weights": True,
        "tags": ["open-weights", "fast"],
    },
    # ── Zhipu AI ──────────────────────────────────────────────────────────────
    {
        "model": "CogVideoX-5B",
        "provider": "Zhipu AI",
        "quality": 66, "price_per_sec": 0.002, "gen_time_s": 90,
        "max_res": "480p", "max_duration_s": 6, "open_weights": True,
        "tags": ["open-weights"],
    },
    # ── Genmo ─────────────────────────────────────────────────────────────────
    {
        "model": "Mochi 1",
        "provider": "Genmo",
        "quality": 63, "price_per_sec": 0.002, "gen_time_s": 60,
        "max_res": "480p", "max_duration_s": 5, "open_weights": True,
        "tags": ["open-weights"],
    },
    # ── Stability AI ──────────────────────────────────────────────────────────
    {
        "model": "Stable Video Diffusion 1.1",
        "provider": "Stability AI",
        "quality": 58, "price_per_sec": 0.001, "gen_time_s": 30,
        "max_res": "576p", "max_duration_s": 4, "open_weights": True,
        "tags": ["open-weights", "fast"],
    },
]


def get_video_df() -> pd.DataFrame:
    df = pd.DataFrame(_RAW)
    df["tags_str"] = df["tags"].apply(lambda t: ", ".join(t))
    return df


def get_video_providers() -> list[str]:
    return sorted({r["provider"] for r in _RAW})
