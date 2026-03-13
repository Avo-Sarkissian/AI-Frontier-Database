"""
Text embedding model dataset.
MTEB scores from the Massive Text Embedding Benchmark leaderboard (mteb.github.io).
Price is USD per 1M tokens. Dimensions = output vector size.
Max tokens = maximum input context length.
"""
import pandas as pd

PROVIDER_COLORS: dict[str, str] = {
    "OpenAI":      "#34d399",
    "Cohere":      "#f87171",
    "Voyage AI":   "#c084fc",
    "Google":      "#60a5fa",
    "Alibaba":     "#38bdf8",
    "Microsoft":   "#818cf8",
    "BAAI":        "#fb923c",
    "Nomic":       "#a3e635",
    "Mistral":     "#facc15",
    "Sentence Transformers": "#6b7280",
    "Jina AI":     "#f472b6",
    "Amazon":      "#ff9900",
}

DEFAULT_COLOR = "#6b7280"

_RAW: list[dict] = [
    # ── Voyage AI ─────────────────────────────────────────────────────────────
    {
        "model": "Voyage 3 Large",
        "provider": "Voyage AI",
        "mteb": 68.3, "price_per_1m": 0.18, "dimensions": 1024,
        "max_tokens": 32000, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    {
        "model": "Voyage 3",
        "provider": "Voyage AI",
        "mteb": 67.1, "price_per_1m": 0.06, "dimensions": 1024,
        "max_tokens": 32000, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    {
        "model": "Voyage 3 Lite",
        "provider": "Voyage AI",
        "mteb": 64.1, "price_per_1m": 0.02, "dimensions": 512,
        "max_tokens": 32000, "open_weights": False,
        "tags": ["english", "fast", "retrieval"],
    },
    {
        "model": "Voyage Code 3",
        "provider": "Voyage AI",
        "mteb": 65.7, "price_per_1m": 0.12, "dimensions": 1024,
        "max_tokens": 32000, "open_weights": False,
        "tags": ["code", "retrieval"],
    },
    # ── Alibaba ───────────────────────────────────────────────────────────────
    {
        "model": "GTE-Qwen2-7B-Instruct",
        "provider": "Alibaba",
        "mteb": 67.1, "price_per_1m": 0.0, "dimensions": 3584,
        "max_tokens": 32000, "open_weights": True,
        "tags": ["multilingual", "retrieval", "open-weights"],
    },
    {
        "model": "GTE-Qwen2-1.5B-Instruct",
        "provider": "Alibaba",
        "mteb": 65.0, "price_per_1m": 0.0, "dimensions": 1536,
        "max_tokens": 32000, "open_weights": True,
        "tags": ["multilingual", "fast", "open-weights"],
    },
    # ── Microsoft ─────────────────────────────────────────────────────────────
    {
        "model": "E5-mistral-7B-Instruct",
        "provider": "Microsoft",
        "mteb": 66.6, "price_per_1m": 0.0, "dimensions": 4096,
        "max_tokens": 32000, "open_weights": True,
        "tags": ["english", "retrieval", "open-weights"],
    },
    # ── OpenAI ────────────────────────────────────────────────────────────────
    {
        "model": "text-embedding-3-large",
        "provider": "OpenAI",
        "mteb": 64.6, "price_per_1m": 0.13, "dimensions": 3072,
        "max_tokens": 8191, "open_weights": False,
        "tags": ["english", "multilingual", "retrieval"],
    },
    {
        "model": "text-embedding-3-small",
        "provider": "OpenAI",
        "mteb": 62.3, "price_per_1m": 0.02, "dimensions": 1536,
        "max_tokens": 8191, "open_weights": False,
        "tags": ["english", "fast", "retrieval"],
    },
    {
        "model": "text-embedding-ada-002",
        "provider": "OpenAI",
        "mteb": 61.0, "price_per_1m": 0.10, "dimensions": 1536,
        "max_tokens": 8191, "open_weights": False,
        "tags": ["english", "retrieval", "legacy"],
    },
    # ── Cohere ────────────────────────────────────────────────────────────────
    {
        "model": "embed-english-v3.0",
        "provider": "Cohere",
        "mteb": 64.5, "price_per_1m": 0.10, "dimensions": 1024,
        "max_tokens": 512, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    {
        "model": "embed-multilingual-v3.0",
        "provider": "Cohere",
        "mteb": 62.7, "price_per_1m": 0.10, "dimensions": 1024,
        "max_tokens": 512, "open_weights": False,
        "tags": ["multilingual", "retrieval"],
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    {
        "model": "Mistral Embed",
        "provider": "Mistral",
        "mteb": 64.1, "price_per_1m": 0.10, "dimensions": 1024,
        "max_tokens": 8192, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    # ── Google ────────────────────────────────────────────────────────────────
    {
        "model": "text-embedding-004",
        "provider": "Google",
        "mteb": 62.3, "price_per_1m": 0.025, "dimensions": 768,
        "max_tokens": 2048, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    {
        "model": "text-multilingual-embedding-002",
        "provider": "Google",
        "mteb": 61.5, "price_per_1m": 0.025, "dimensions": 768,
        "max_tokens": 2048, "open_weights": False,
        "tags": ["multilingual", "retrieval"],
    },
    # ── BAAI ──────────────────────────────────────────────────────────────────
    {
        "model": "BGE-M3",
        "provider": "BAAI",
        "mteb": 63.3, "price_per_1m": 0.0, "dimensions": 1024,
        "max_tokens": 8192, "open_weights": True,
        "tags": ["multilingual", "retrieval", "open-weights"],
    },
    {
        "model": "BGE-Large-EN-v1.5",
        "provider": "BAAI",
        "mteb": 63.6, "price_per_1m": 0.0, "dimensions": 1024,
        "max_tokens": 512, "open_weights": True,
        "tags": ["english", "retrieval", "open-weights"],
    },
    # ── Nomic ─────────────────────────────────────────────────────────────────
    {
        "model": "Nomic Embed Text v1.5",
        "provider": "Nomic",
        "mteb": 62.4, "price_per_1m": 0.0, "dimensions": 768,
        "max_tokens": 8192, "open_weights": True,
        "tags": ["english", "retrieval", "open-weights"],
    },
    # ── Jina AI ───────────────────────────────────────────────────────────────
    {
        "model": "jina-embeddings-v3",
        "provider": "Jina AI",
        "mteb": 63.7, "price_per_1m": 0.02, "dimensions": 1024,
        "max_tokens": 8192, "open_weights": True,
        "tags": ["multilingual", "retrieval", "open-weights"],
    },
    # ── Amazon ────────────────────────────────────────────────────────────────
    {
        "model": "Titan Embed Text v2",
        "provider": "Amazon",
        "mteb": 61.0, "price_per_1m": 0.02, "dimensions": 1024,
        "max_tokens": 8192, "open_weights": False,
        "tags": ["english", "retrieval"],
    },
    # ── Sentence Transformers ─────────────────────────────────────────────────
    {
        "model": "all-MiniLM-L6-v2",
        "provider": "Sentence Transformers",
        "mteb": 56.3, "price_per_1m": 0.0, "dimensions": 384,
        "max_tokens": 256, "open_weights": True,
        "tags": ["english", "fast", "tiny", "open-weights"],
    },
    {
        "model": "all-mpnet-base-v2",
        "provider": "Sentence Transformers",
        "mteb": 57.8, "price_per_1m": 0.0, "dimensions": 768,
        "max_tokens": 384, "open_weights": True,
        "tags": ["english", "open-weights"],
    },
]


def get_embedding_df() -> pd.DataFrame:
    df = pd.DataFrame(_RAW)
    df["tags_str"] = df["tags"].apply(lambda t: ", ".join(t))
    # For scatter: treat free (0) as a very small price so log scale works
    df["price_plot"] = df["price_per_1m"].apply(lambda p: p if p > 0 else 0.008)
    df["is_free"]    = df["price_per_1m"] == 0.0
    return df


def get_embedding_providers() -> list[str]:
    return sorted({r["provider"] for r in _RAW})
