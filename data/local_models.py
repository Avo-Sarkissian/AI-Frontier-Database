"""
Local / open-weight model catalog for the 'Run Local' tab.

Model data is loaded from data/raw/aa_local_models.csv, which is kept fresh
by data/local_scraper.py running as a background thread (same pattern as the
main model scraper). The hardcoded _MODELS_RAW list is no longer the source
of truth and is kept only as a last-resort fallback when no cache exists.

VRAM formula:  params_b × bytes_per_weight × OVERHEAD_FACTOR
Speed formula: memory_bandwidth_gbps / model_gb × efficiency
               (memory-bandwidth-bound inference, validated against llama.cpp benchmarks)
"""
from pathlib import Path
import pandas as pd

_LOCAL_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"

# ── Quantization ─────────────────────────────────────────────────────────────
QUANT_LEVELS = ["FP16", "Q8", "Q5", "Q4", "Q3", "Q2"]

# Bytes per parameter at each quantization level
QUANT_BYTES: dict[str, float] = {
    "FP16": 2.000,
    "Q8":   1.000,
    "Q5":   0.625,
    "Q4":   0.500,
    "Q3":   0.375,
    "Q2":   0.250,
}

# KV-cache + activation overhead multiplier
_OVERHEAD = 1.18

# Memory efficiency factor by hardware type
# (fraction of theoretical peak bandwidth utilized during inference)
# apple:    0.82 = MLX Metal GPU kernels via Ollama ≥ 0.6 / mlx-lm
# qualcomm: 0.55 = Adreno GPU via llama.cpp Vulkan backend on Windows ARM
# intel:    0.50 = Arc GPU via SYCL/oneAPI backend in llama.cpp
_EFF: dict[str, float] = {
    "nvidia":   0.55,
    "amd":      0.50,
    "apple":    0.82,
    "qualcomm": 0.55,
    "intel":    0.50,
    "cpu":      0.30,
}


# ── Color palette by model family ────────────────────────────────────────────
FAMILY_COLORS: dict[str, str] = {
    "Meta":        "#4c6ef5",
    "Mistral":     "#fd7e14",
    "Google":      "#2f9e44",
    "Microsoft":   "#1c7ed6",
    "Alibaba":     "#e64980",
    "DeepSeek":    "#0c8599",
    "TII":         "#7048e8",
    "Cohere":      "#c92a2a",
    "01.AI":       "#d9480f",
    "Allen AI":    "#5c7cfa",
    "InternLM":    "#74c0fc",
    "IBM":         "#868e96",
    "HuggingFace": "#fab005",
    "Moonshot":    "#a9e34b",
    "SOLAR":       "#ff6b6b",
    "xAI":         "#aaaaaa",
    "OpenAI":      "#10a37f",   # OpenAI brand green
}
DEFAULT_FAMILY_COLOR = "#555555"


# ── Model catalog ─────────────────────────────────────────────────────────────
# Fields:
#   name         - display name
#   family       - model family (for coloring)
#   params_b     - total parameter count in billions (use full count for MoE)
#   active_b     - active params per forward pass (= params_b for dense models)
#   context_k    - max context window in thousands of tokens
#   quality      - AA Intelligence Index score (raw, open-ended; calibrated to AA scale)
#   license      - software license
#   tags         - list of capability tags
#   moe          - True if mixture-of-experts architecture
_MODELS_RAW: list[dict] = [
    # Quality scores are calibrated to the AA Intelligence Index scale (same as
    # the online tab). Scores marked "AA exact" are taken directly from the
    # live dataset; others are estimated proportionally based on size/generation
    # relative to models whose AA scores are known.

    # ── Meta Llama ────────────────────────────────────────────────────────────
    {"name": "Llama 3.2 1B",        "family": "Meta", "params_b": 1.24,  "active_b": 1.24,  "context_k": 128, "quality": 2,  "license": "Llama 3.2", "tags": ["multilingual"]},
    {"name": "Llama 3.2 3B",        "family": "Meta", "params_b": 3.21,  "active_b": 3.21,  "context_k": 128, "quality": 4,  "license": "Llama 3.2", "tags": ["multilingual"]},
    {"name": "Llama 3.1 8B",        "family": "Meta", "params_b": 8.03,  "active_b": 8.03,  "context_k": 128, "quality": 6,  "license": "Llama 3.1", "tags": ["multilingual"]},
    {"name": "Llama 3.3 70B",       "family": "Meta", "params_b": 70.6,  "active_b": 70.6,  "context_k": 128, "quality": 14, "license": "Llama 3.3", "tags": ["multilingual"]},           # AA exact
    {"name": "Llama 3.1 405B",      "family": "Meta", "params_b": 405,   "active_b": 405,   "context_k": 128, "quality": 14, "license": "Llama 3.1", "tags": ["multilingual"]},           # AA exact
    {"name": "Llama 3.2 11B Vision","family": "Meta", "params_b": 11,    "active_b": 11,    "context_k": 128, "quality": 9,  "license": "Llama 3.2", "tags": ["vision", "multilingual"]}, # AA exact
    {"name": "Llama 3.2 90B Vision","family": "Meta", "params_b": 90,    "active_b": 90,    "context_k": 128, "quality": 12, "license": "Llama 3.2", "tags": ["vision", "multilingual"]}, # AA exact
    {"name": "Llama 4 Scout",       "family": "Meta", "params_b": 109,   "active_b": 17,    "context_k": 1000,"quality": 13, "license": "Llama 4",   "tags": ["vision", "multilingual", "reasoning"], "moe": True},  # AA exact
    {"name": "Llama 4 Maverick",    "family": "Meta", "params_b": 400,   "active_b": 17,    "context_k": 1000,"quality": 18, "license": "Llama 4",   "tags": ["vision", "multilingual", "reasoning"], "moe": True},  # AA exact

    # ── Mistral / Mixtral ─────────────────────────────────────────────────────
    {"name": "Mistral 7B v0.3",     "family": "Mistral", "params_b": 7.24,  "active_b": 7.24,  "context_k": 32,  "quality": 5,  "license": "Apache 2.0", "tags": []},
    {"name": "Mistral Nemo 12B",    "family": "Mistral", "params_b": 12.2,  "active_b": 12.2,  "context_k": 128, "quality": 7,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Mistral Small 3.2 22B","family": "Mistral", "params_b": 22.0,  "active_b": 22.0,  "context_k": 128, "quality": 15, "license": "Apache 2.0", "tags": ["multilingual"]},  # AA exact
    {"name": "Mistral Large 2 123B","family": "Mistral", "params_b": 123,   "active_b": 123,   "context_k": 128, "quality": 18, "license": "MNPL",       "tags": ["multilingual"]},  # ~Mistral Large 3 (AA: 23) minus one gen
    {"name": "Mixtral 8x7B",        "family": "Mistral", "params_b": 46.7,  "active_b": 12.9,  "context_k": 32,  "quality": 8,  "license": "Apache 2.0", "tags": ["multilingual"], "moe": True},
    {"name": "Mixtral 8x22B",       "family": "Mistral", "params_b": 141,   "active_b": 39.1,  "context_k": 64,  "quality": 13, "license": "Apache 2.0", "tags": ["multilingual"], "moe": True},

    # ── Google Gemma ──────────────────────────────────────────────────────────
    {"name": "Gemma 2 2B",          "family": "Google", "params_b": 2.61,  "active_b": 2.61,  "context_k": 8,   "quality": 2,  "license": "Gemma ToS", "tags": []},
    {"name": "Gemma 2 9B",          "family": "Google", "params_b": 9.24,  "active_b": 9.24,  "context_k": 8,   "quality": 7,  "license": "Gemma ToS", "tags": []},
    {"name": "Gemma 2 27B",         "family": "Google", "params_b": 27.2,  "active_b": 27.2,  "context_k": 8,   "quality": 11, "license": "Gemma ToS", "tags": []},
    {"name": "Gemma 3 1B",          "family": "Google", "params_b": 1.0,   "active_b": 1.0,   "context_k": 128, "quality": 2,  "license": "Gemma ToS", "tags": ["multilingual"]},
    {"name": "Gemma 3 4B",          "family": "Google", "params_b": 4.0,   "active_b": 4.0,   "context_k": 128, "quality": 5,  "license": "Gemma ToS", "tags": ["multilingual", "vision"]},
    {"name": "Gemma 3 12B",         "family": "Google", "params_b": 12.0,  "active_b": 12.0,  "context_k": 128, "quality": 9,  "license": "Gemma ToS", "tags": ["multilingual", "vision"]},
    {"name": "Gemma 3 27B",         "family": "Google", "params_b": 27.0,  "active_b": 27.0,  "context_k": 128, "quality": 13, "license": "Gemma ToS", "tags": ["multilingual", "vision"]},
    # ── Gemma 4 (Apr 2026) — Apache 2.0, multimodal (text+image+video; E2B/E4B also audio)
    # Sizes: E2B / E4B (MoE+PLE, on-device), 26B-A4B (MoE), 31B (dense)
    # E2B/E4B: PLE gives "effective" active-param footprint; total params ~2-4× larger.
    # Context: 128k (E2B/E4B), 256k (26B-A4B, 31B). Quality = AA exact from live data.
    {"name": "Gemma 4 E2B",     "family": "Google", "params_b": 5.5,  "active_b": 2.0,  "context_k": 128, "quality": 9,  "license": "Apache 2.0", "tags": ["multilingual", "vision", "reasoning"], "moe": True},
    {"name": "Gemma 4 E4B",     "family": "Google", "params_b": 8.2,  "active_b": 4.0,  "context_k": 128, "quality": 15, "license": "Apache 2.0", "tags": ["multilingual", "vision", "reasoning"], "moe": True},
    {"name": "Gemma 4 26B-A4B", "family": "Google", "params_b": 26.0, "active_b": 4.0,  "context_k": 256, "quality": 31, "license": "Apache 2.0", "tags": ["multilingual", "vision", "reasoning"], "moe": True},
    {"name": "Gemma 4 31B",     "family": "Google", "params_b": 31.0, "active_b": 31.0, "context_k": 256, "quality": 39, "license": "Apache 2.0", "tags": ["multilingual", "vision", "reasoning"]},

    # ── Microsoft Phi ─────────────────────────────────────────────────────────
    {"name": "Phi-3 Mini 3.8B",     "family": "Microsoft", "params_b": 3.82,  "active_b": 3.82,  "context_k": 128, "quality": 4,  "license": "MIT", "tags": []},
    {"name": "Phi-3.5 Mini 3.8B",   "family": "Microsoft", "params_b": 3.82,  "active_b": 3.82,  "context_k": 128, "quality": 5,  "license": "MIT", "tags": ["multilingual"]},
    {"name": "Phi-3 Medium 14B",    "family": "Microsoft", "params_b": 14.0,  "active_b": 14.0,  "context_k": 128, "quality": 8,  "license": "MIT", "tags": []},
    {"name": "Phi-4 14B",           "family": "Microsoft", "params_b": 14.7,  "active_b": 14.7,  "context_k": 16,  "quality": 10, "license": "MIT", "tags": ["reasoning"]},  # AA exact
    {"name": "Phi-4 Mini 3.8B",     "family": "Microsoft", "params_b": 3.82,  "active_b": 3.82,  "context_k": 128, "quality": 7,  "license": "MIT", "tags": ["reasoning"]},

    # ── Alibaba Qwen 2.5 (one generation behind Qwen 3) ──────────────────────
    {"name": "Qwen 2.5 0.5B",       "family": "Alibaba", "params_b": 0.50,  "active_b": 0.50,  "context_k": 32,  "quality": 1,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 1.5B",       "family": "Alibaba", "params_b": 1.54,  "active_b": 1.54,  "context_k": 32,  "quality": 2,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 3B",         "family": "Alibaba", "params_b": 3.09,  "active_b": 3.09,  "context_k": 32,  "quality": 4,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 7B",         "family": "Alibaba", "params_b": 7.62,  "active_b": 7.62,  "context_k": 128, "quality": 7,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 14B",        "family": "Alibaba", "params_b": 14.8,  "active_b": 14.8,  "context_k": 128, "quality": 10, "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 32B",        "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 14, "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 72B",        "family": "Alibaba", "params_b": 72.7,  "active_b": 72.7,  "context_k": 128, "quality": 17, "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Qwen 2.5 Coder 7B",   "family": "Alibaba", "params_b": 7.62,  "active_b": 7.62,  "context_k": 128, "quality": 7,  "license": "Apache 2.0", "tags": ["code"]},
    {"name": "Qwen 2.5 Coder 32B",  "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 14, "license": "Apache 2.0", "tags": ["code"]},
    {"name": "QwQ 32B",             "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 17, "license": "Apache 2.0", "tags": ["reasoning"]},

    # ── Alibaba Qwen 3.5 (est. late 2025) ───────────────────────────────────
    # Generation after Qwen3 — quality scores estimated ~20% above Qwen3 peers.
    # Sizes follow the same dense/MoE pattern. Context 128k, reasoning toggle.
    {"name": "Qwen3.5 4B",              "family": "Alibaba", "params_b": 4.00,  "active_b": 4.00,  "context_k": 128, "quality": 11, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3.5 8B",              "family": "Alibaba", "params_b": 8.00,  "active_b": 8.00,  "context_k": 128, "quality": 17, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3.5 14B",             "family": "Alibaba", "params_b": 14.0,  "active_b": 14.0,  "context_k": 128, "quality": 22, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3.5 32B",             "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 29, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3.5 72B",             "family": "Alibaba", "params_b": 72.7,  "active_b": 72.7,  "context_k": 128, "quality": 36, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},

    # ── Alibaba Qwen3-Coder (code-specialized, est. 2025) ────────────────────
    # Code-focused variant of the Qwen3 architecture — strong at code generation,
    # editing, and repo-level reasoning. Quality scores estimated vs Qwen3 base.
    {"name": "Qwen3-Coder 1.5B",        "family": "Alibaba", "params_b": 1.54,  "active_b": 1.54,  "context_k": 128, "quality": 6,  "license": "Apache 2.0", "tags": ["code", "multilingual"]},
    {"name": "Qwen3-Coder 7B",          "family": "Alibaba", "params_b": 7.62,  "active_b": 7.62,  "context_k": 128, "quality": 13, "license": "Apache 2.0", "tags": ["code", "multilingual"]},
    {"name": "Qwen3-Coder 14B",         "family": "Alibaba", "params_b": 14.0,  "active_b": 14.0,  "context_k": 128, "quality": 19, "license": "Apache 2.0", "tags": ["code", "multilingual"]},
    {"name": "Qwen3-Coder 32B",         "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 25, "license": "Apache 2.0", "tags": ["code", "multilingual"]},

    # ── Alibaba Qwen 3 (Apr 2025) ────────────────────────────────────────────
    # Dense models: 0.6B / 1.7B / 4B / 8B / 14B / 32B
    # MoE models: 30B-A3B (30B total, 3B active) / 235B-A22B (235B total, 22B active)
    # Context: 128k tokens, built-in thinking/non-thinking toggle
    # Quality calibrated to AA scale: Qwen3-8B beats Qwen2.5-72B in many benchmarks;
    # Qwen3-32B (thinking) is competitive with o1-mini; 235B approaches top open models.
    {"name": "Qwen3 0.6B",             "family": "Alibaba", "params_b": 0.60,  "active_b": 0.60,  "context_k": 128, "quality": 3,  "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 1.7B",             "family": "Alibaba", "params_b": 1.70,  "active_b": 1.70,  "context_k": 128, "quality": 5,  "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 4B",               "family": "Alibaba", "params_b": 4.00,  "active_b": 4.00,  "context_k": 128, "quality": 9,  "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 8B",               "family": "Alibaba", "params_b": 8.00,  "active_b": 8.00,  "context_k": 128, "quality": 14, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 14B",              "family": "Alibaba", "params_b": 14.0,  "active_b": 14.0,  "context_k": 128, "quality": 18, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 32B",              "family": "Alibaba", "params_b": 32.8,  "active_b": 32.8,  "context_k": 128, "quality": 24, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"]},
    {"name": "Qwen3 30B-A3B",          "family": "Alibaba", "params_b": 30.0,  "active_b": 3.0,   "context_k": 128, "quality": 21, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"], "moe": True},
    {"name": "Qwen3 235B-A22B",        "family": "Alibaba", "params_b": 235.0, "active_b": 22.0,  "context_k": 128, "quality": 36, "license": "Apache 2.0", "tags": ["multilingual", "reasoning", "code"], "moe": True},

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    # DeepSeek V3.2 is AA: 42; V3 (older) estimated at ~36; R1 (reasoning) ~38
    {"name": "DeepSeek V3",                  "family": "DeepSeek", "params_b": 671,  "active_b": 37,   "context_k": 128, "quality": 36, "license": "MIT", "tags": ["code", "multilingual"],              "moe": True},
    {"name": "DeepSeek R1",                  "family": "DeepSeek", "params_b": 671,  "active_b": 37,   "context_k": 128, "quality": 38, "license": "MIT", "tags": ["reasoning", "code", "multilingual"], "moe": True},
    {"name": "DeepSeek R1 Distill Qwen 1.5B","family": "DeepSeek", "params_b": 1.54,  "active_b": 1.54, "context_k": 128, "quality": 4,  "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek R1 Distill Qwen 7B",  "family": "DeepSeek", "params_b": 7.62,  "active_b": 7.62, "context_k": 128, "quality": 8,  "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek R1 Distill Qwen 14B", "family": "DeepSeek", "params_b": 14.8,  "active_b": 14.8, "context_k": 128, "quality": 12, "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek R1 Distill Qwen 32B", "family": "DeepSeek", "params_b": 32.8,  "active_b": 32.8, "context_k": 128, "quality": 16, "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek R1 Distill Llama 8B", "family": "DeepSeek", "params_b": 8.03,  "active_b": 8.03, "context_k": 128, "quality": 9,  "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek R1 Distill Llama 70B","family": "DeepSeek", "params_b": 70.6,  "active_b": 70.6, "context_k": 128, "quality": 15, "license": "MIT", "tags": ["reasoning"]},
    {"name": "DeepSeek Coder V2 Lite",       "family": "DeepSeek", "params_b": 16.0,  "active_b": 2.4,  "context_k": 128, "quality": 10, "license": "MIT", "tags": ["code"],           "moe": True},

    # ── OpenAI open-weight (gpt-oss series, Apache 2.0, Feb 2026) ────────────
    # gpt-oss-120b: near-parity with o4-mini; natively MXFP4 (≈Q4 for sizing)
    # gpt-oss-20b:  similar to o3-mini reasoning; runs on 16 GB edge devices
    # Both support low/medium/high reasoning effort levels.
    {"name": "gpt-oss-120b",         "family": "OpenAI", "params_b": 120,   "active_b": 120,   "context_k": 128, "quality": 38, "license": "Apache 2.0", "tags": ["reasoning", "code"]},  # estimated — near o4-mini
    {"name": "gpt-oss-20b",          "family": "OpenAI", "params_b": 20,    "active_b": 20,    "context_k": 128, "quality": 25, "license": "Apache 2.0", "tags": ["reasoning", "code"]},   # estimated — similar to o3-mini on reasoning

    # ── Moonshot Kimi ─────────────────────────────────────────────────────────
    {"name": "Kimi-VL-A3B",         "family": "Moonshot", "params_b": 16.0,  "active_b": 3.0,   "context_k": 128, "quality": 7,  "license": "MIT", "tags": ["vision", "reasoning"], "moe": True},

    # ── TII Falcon ────────────────────────────────────────────────────────────
    {"name": "Falcon 7B",           "family": "TII",    "params_b": 7.0,   "active_b": 7.0,   "context_k": 2,   "quality": 2,  "license": "Apache 2.0", "tags": []},
    {"name": "Falcon 40B",          "family": "TII",    "params_b": 40.9,  "active_b": 40.9,  "context_k": 2,   "quality": 6,  "license": "Apache 2.0", "tags": []},
    {"name": "Falcon 180B",         "family": "TII",    "params_b": 180,   "active_b": 180,   "context_k": 2,   "quality": 10, "license": "Falcon",     "tags": []},

    # ── Cohere Command R ──────────────────────────────────────────────────────
    {"name": "Command R 35B",       "family": "Cohere", "params_b": 35.0,  "active_b": 35.0,  "context_k": 128, "quality": 9,  "license": "CC-BY-NC",  "tags": ["multilingual"]},
    {"name": "Command R+ 104B",     "family": "Cohere", "params_b": 104,   "active_b": 104,   "context_k": 128, "quality": 13, "license": "CC-BY-NC",  "tags": ["multilingual"]},

    # ── 01.AI Yi ──────────────────────────────────────────────────────────────
    {"name": "Yi-1.5 9B",           "family": "01.AI",  "params_b": 9.0,   "active_b": 9.0,   "context_k": 4,   "quality": 6,  "license": "Apache 2.0", "tags": ["multilingual"]},
    {"name": "Yi-1.5 34B",          "family": "01.AI",  "params_b": 34.0,  "active_b": 34.0,  "context_k": 4,   "quality": 10, "license": "Apache 2.0", "tags": ["multilingual"]},

    # ── Allen AI OLMo ─────────────────────────────────────────────────────────
    # OLMo 3 7B (AA: 8), OLMo 2 is one generation older
    {"name": "OLMo 2 7B",           "family": "Allen AI", "params_b": 7.3,  "active_b": 7.3,  "context_k": 4,   "quality": 5,  "license": "Apache 2.0", "tags": []},
    {"name": "OLMo 2 13B",          "family": "Allen AI", "params_b": 13.0, "active_b": 13.0, "context_k": 4,   "quality": 8,  "license": "Apache 2.0", "tags": []},

    # ── InternLM ──────────────────────────────────────────────────────────────
    {"name": "InternLM 2.5 7B",     "family": "InternLM","params_b": 7.74,  "active_b": 7.74,  "context_k": 32,  "quality": 7,  "license": "Apache 2.0", "tags": ["multilingual", "code"]},
    {"name": "InternLM 2.5 20B",    "family": "InternLM","params_b": 20.0,  "active_b": 20.0,  "context_k": 32,  "quality": 10, "license": "Apache 2.0", "tags": ["multilingual", "code"]},

    # ── IBM Granite ───────────────────────────────────────────────────────────
    # Granite 4.0 H Small (AA: 11); Granite 3.1 is one gen older
    {"name": "Granite 3.1 2B",      "family": "IBM",    "params_b": 2.0,   "active_b": 2.0,   "context_k": 128, "quality": 3,  "license": "Apache 2.0", "tags": ["code"]},
    {"name": "Granite 3.1 8B",      "family": "IBM",    "params_b": 8.0,   "active_b": 8.0,   "context_k": 128, "quality": 6,  "license": "Apache 2.0", "tags": ["code"]},

    # ── HuggingFace SmolLM ────────────────────────────────────────────────────
    {"name": "SmolLM2 135M",        "family": "HuggingFace", "params_b": 0.135, "active_b": 0.135, "context_k": 8,  "quality": 1,  "license": "Apache 2.0", "tags": []},
    {"name": "SmolLM2 360M",        "family": "HuggingFace", "params_b": 0.36,  "active_b": 0.36,  "context_k": 8,  "quality": 1,  "license": "Apache 2.0", "tags": []},
    {"name": "SmolLM2 1.7B",        "family": "HuggingFace", "params_b": 1.71,  "active_b": 1.71,  "context_k": 8,  "quality": 2,  "license": "Apache 2.0", "tags": []},

    # ── SOLAR ─────────────────────────────────────────────────────────────────
    {"name": "Solar 10.7B",         "family": "SOLAR",  "params_b": 10.7,  "active_b": 10.7,  "context_k": 4,   "quality": 6,  "license": "Apache 2.0", "tags": []},
]

# Normalize: fill defaults for non-MoE models
for _m in _MODELS_RAW:
    _m.setdefault("moe", False)
    _m.setdefault("tags", [])


# ── GPU / hardware presets ────────────────────────────────────────────────────
# Fields: name, vram_gb, bandwidth_gbps, hw_type, category
# Bandwidth sources: NVIDIA official specs, Apple silicon specs pages
GPUS: list[dict] = [
    # ── NVIDIA RTX 50 (Blackwell, GDDR7) ─────────────────────────────────────
    # 5090: 512-bit × 28 Gbps = 1792 GB/s | 5080: 256-bit × 30 Gbps = 960 GB/s
    # 5070 Ti: 256-bit × 27 Gbps = 864 GB/s | 5070: 192-bit × 28 Gbps = 672 GB/s
    # 5060 Ti: 128-bit × 28 Gbps = 448 GB/s | 5060: 128-bit × 18 Gbps = 288 GB/s
    {"name": "NVIDIA RTX 5090",          "vram_gb": 32,  "bandwidth_gbps": 1792, "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5080",          "vram_gb": 16,  "bandwidth_gbps": 960,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5070 Ti",       "vram_gb": 16,  "bandwidth_gbps": 864,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5070",          "vram_gb": 12,  "bandwidth_gbps": 672,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060 Ti 16GB",  "vram_gb": 16,  "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060 Ti",       "vram_gb": 8,   "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060",          "vram_gb": 8,   "bandwidth_gbps": 288,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    # ── NVIDIA RTX 40 (Ada Lovelace, GDDR6X) ─────────────────────────────────
    {"name": "NVIDIA RTX 4090",          "vram_gb": 24,  "bandwidth_gbps": 1008, "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4080 Super",    "vram_gb": 16,  "bandwidth_gbps": 736,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4080",          "vram_gb": 16,  "bandwidth_gbps": 717,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Ti Super", "vram_gb": 16,  "bandwidth_gbps": 672,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Ti",       "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Super",    "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070",          "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060 Ti 16GB",  "vram_gb": 16,  "bandwidth_gbps": 288,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060 Ti",       "vram_gb": 8,   "bandwidth_gbps": 288,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060",          "vram_gb": 8,   "bandwidth_gbps": 272,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    # ── NVIDIA RTX 30 (Ampere, GDDR6X) ───────────────────────────────────────
    {"name": "NVIDIA RTX 3090 Ti",       "vram_gb": 24,  "bandwidth_gbps": 1008, "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3090",          "vram_gb": 24,  "bandwidth_gbps": 936,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080 Ti",       "vram_gb": 12,  "bandwidth_gbps": 912,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080 12GB",     "vram_gb": 12,  "bandwidth_gbps": 912,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080",          "vram_gb": 10,  "bandwidth_gbps": 760,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3070 Ti",       "vram_gb": 8,   "bandwidth_gbps": 608,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3070",          "vram_gb": 8,   "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 2080 Ti",       "vram_gb": 11,  "bandwidth_gbps": 616,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    # ── NVIDIA Data Center (Blackwell) ───────────────────────────────────────
    # GB200 NVL2: 2×B200 on NVLink, 384 GB HBM3e, 16 TB/s aggregate bandwidth
    # B200 SXM: 192 GB HBM3e, 8 TB/s
    {"name": "NVIDIA GB200 NVL2",        "vram_gb": 384, "bandwidth_gbps": 16000, "hw_type": "nvidia", "category": "NVIDIA Data Center (Blackwell)"},
    {"name": "NVIDIA B200 SXM",          "vram_gb": 192, "bandwidth_gbps": 8000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Blackwell)"},
    # ── NVIDIA Data Center (Hopper) ──────────────────────────────────────────
    # H200 SXM: 141 GB HBM3e, 4.8 TB/s | H200 PCIe: 141 GB, 3.35 TB/s
    {"name": "NVIDIA H200 SXM",          "vram_gb": 141, "bandwidth_gbps": 4800,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H200 PCIe",         "vram_gb": 141, "bandwidth_gbps": 3350,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H100 SXM",          "vram_gb": 80,  "bandwidth_gbps": 3350,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H100 PCIe",         "vram_gb": 80,  "bandwidth_gbps": 2000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    # ── NVIDIA Data Center (Ada / Ampere) ────────────────────────────────────
    # L40S: 48 GB GDDR6, 864 GB/s (inference-optimised Ada successor to A100)
    # L40: 48 GB GDDR6, 864 GB/s | A40: 48 GB GDDR6, 696 GB/s
    {"name": "NVIDIA L40S",              "vram_gb": 48,  "bandwidth_gbps": 864,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA L40",               "vram_gb": 48,  "bandwidth_gbps": 864,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A40",               "vram_gb": 48,  "bandwidth_gbps": 696,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A100 80GB",         "vram_gb": 80,  "bandwidth_gbps": 2000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A100 40GB",         "vram_gb": 40,  "bandwidth_gbps": 1555,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA V100 32GB",         "vram_gb": 32,  "bandwidth_gbps": 900,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    # ── NVIDIA Professional Workstation ──────────────────────────────────────
    {"name": "NVIDIA RTX 6000 Ada",      "vram_gb": 48,  "bandwidth_gbps": 960,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    {"name": "NVIDIA RTX 5000 Ada",      "vram_gb": 32,  "bandwidth_gbps": 576,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    {"name": "NVIDIA A6000 Ada",         "vram_gb": 48,  "bandwidth_gbps": 864,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    {"name": "NVIDIA A6000",             "vram_gb": 48,  "bandwidth_gbps": 768,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    # ── Apple Silicon ─────────────────────────────────────────────────────────
    # Unified memory = VRAM; bandwidth from Apple silicon spec pages.
    # M3 Max: 14-core GPU (300 GB/s) base / 16-core (400 GB/s) high configs.
    # M3 Ultra (Mac Studio 2025): 2× M3 Max → 819 GB/s, up to 512 GB.
    # M4 Max: 14-core GPU (410 GB/s) base / 16-core (546 GB/s) high configs.
    # M5 (MacBook Air, Mar 2026): 153.6 GB/s, up to 32 GB.
    # M5 Pro (MacBook Pro, Mar 2026): 307 GB/s, up to 64 GB.
    # M5 Max (MacBook Pro, Mar 2026): 614 GB/s, up to 128 GB (single-tier GPU).
    # M5 Ultra: not yet announced — expected Mac Studio mid-2026.
    # ── M1 ──
    {"name": "Apple M1 (8 GB)",          "vram_gb": 8,   "bandwidth_gbps": 68,   "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 68,   "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Pro (16 GB)",     "vram_gb": 16,  "bandwidth_gbps": 200,  "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Pro (32 GB)",     "vram_gb": 32,  "bandwidth_gbps": 200,  "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Max (32 GB)",     "vram_gb": 32,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Ultra (64 GB)",   "vram_gb": 64,  "bandwidth_gbps": 800,  "hw_type": "apple",  "category": "Apple M1"},
    {"name": "Apple M1 Ultra (128 GB)",  "vram_gb": 128, "bandwidth_gbps": 800,  "hw_type": "apple",  "category": "Apple M1"},
    # ── M2 ──
    {"name": "Apple M2 (8 GB)",          "vram_gb": 8,   "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Pro (16 GB)",     "vram_gb": 16,  "bandwidth_gbps": 200,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Pro (32 GB)",     "vram_gb": 32,  "bandwidth_gbps": 200,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Max (32 GB)",     "vram_gb": 32,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Max (96 GB)",     "vram_gb": 96,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Ultra (64 GB)",   "vram_gb": 64,  "bandwidth_gbps": 800,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Ultra (128 GB)",  "vram_gb": 128, "bandwidth_gbps": 800,  "hw_type": "apple",  "category": "Apple M2"},
    {"name": "Apple M2 Ultra (192 GB)",  "vram_gb": 192, "bandwidth_gbps": 800,  "hw_type": "apple",  "category": "Apple M2"},
    # ── M3 ──
    {"name": "Apple M3 (8 GB)",          "vram_gb": 8,   "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 100,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Pro (18 GB)",     "vram_gb": 18,  "bandwidth_gbps": 150,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Pro (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 150,  "hw_type": "apple",  "category": "Apple M3"},
    # M3 Max 14-core GPU: 300 GB/s — MacBook Pro 14" base, MacBook Pro 16" base
    {"name": "Apple M3 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 300,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 300,  "hw_type": "apple",  "category": "Apple M3"},
    # M3 Max 16-core GPU: 400 GB/s — MacBook Pro 16" high, Mac Studio
    {"name": "Apple M3 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (96 GB)",     "vram_gb": 96,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    # M3 Ultra: 2× M3 Max (16-core) → 819 GB/s — Mac Studio (2025), up to 512 GB
    {"name": "Apple M3 Ultra (192 GB)",  "vram_gb": 192, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Ultra (256 GB)",  "vram_gb": 256, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Ultra (512 GB)",  "vram_gb": 512, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    # ── M4 ──
    {"name": "Apple M4 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 120,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 (32 GB)",         "vram_gb": 32,  "bandwidth_gbps": 120,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Pro (24 GB)",     "vram_gb": 24,  "bandwidth_gbps": 273,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Pro (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 273,  "hw_type": "apple",  "category": "Apple M4"},
    # M4 Max 14-core GPU: 410 GB/s — MacBook Pro 14" / 16" base tier
    {"name": "Apple M4 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 410,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 410,  "hw_type": "apple",  "category": "Apple M4"},
    # M4 Max 16-core GPU: 546 GB/s — MacBook Pro 16" high, Mac Studio
    {"name": "Apple M4 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Max (96 GB)",     "vram_gb": 96,  "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    # ── M5 ──
    # M5 base (MacBook Air, Mar 2026): 153.6 GB/s, up to 32 GB
    {"name": "Apple M5 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 154,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 154,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 (32 GB)",         "vram_gb": 32,  "bandwidth_gbps": 154,  "hw_type": "apple",  "category": "Apple M5"},
    # M5 Pro (MacBook Pro 14"/16", Mar 2026): 307 GB/s, up to 64 GB
    {"name": "Apple M5 Pro (24 GB)",     "vram_gb": 24,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Pro (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Pro (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    # M5 Max (MacBook Pro 14"/16", Mar 2026): 614 GB/s, up to 128 GB
    {"name": "Apple M5 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (96 GB)",     "vram_gb": 96,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    # M5 Ultra: not yet announced (expected Mac Studio mid-2026)
    # ── Apple iPhone (on-device inference via llama.cpp Metal / Core ML) ────────
    # Named by chip, not device. Usable RAM ≈ total minus ~2 GB OS reservation.
    # Only models ≤ ~4 GB VRAM fit on phones — filter enforces this automatically.
    # Bandwidth from Apple silicon spec pages.
    {"name": "A16 (iPhone 14 Pro / 16e)", "vram_gb": 6,  "bandwidth_gbps": 60,   "hw_type": "apple",  "category": "Apple — iPhone"},
    {"name": "A17 Pro (iPhone 15 Pro)",   "vram_gb": 6,  "bandwidth_gbps": 68,   "hw_type": "apple",  "category": "Apple — iPhone"},
    {"name": "A18 Pro (iPhone 16 Pro)",   "vram_gb": 6,  "bandwidth_gbps": 75,   "hw_type": "apple",  "category": "Apple — iPhone"},
    # A19 Pro (iPhone 17 Pro, Sep 2025): 12 GB RAM, improved memory bandwidth
    {"name": "A19 Pro (iPhone 17 Pro)",   "vram_gb": 10, "bandwidth_gbps": 84,   "hw_type": "apple",  "category": "Apple — iPhone"},
    # ── AMD RDNA 4 (2025) ─────────────────────────────────────────────────────
    # RX 9070 XT: 16 GB GDDR6, 256-bit, 717 GB/s — RDNA4 flagship mainstream
    # RX 9070:    16 GB GDDR6, 256-bit, 640 GB/s
    # RX 9060 XT: 8/16 GB GDDR6, 128-bit, 384 GB/s — announced, mid-2025
    {"name": "AMD RX 9070 XT",           "vram_gb": 16,  "bandwidth_gbps": 717,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9070",              "vram_gb": 16,  "bandwidth_gbps": 640,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9060 XT (16 GB)",   "vram_gb": 16,  "bandwidth_gbps": 384,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9060 XT (8 GB)",    "vram_gb": 8,   "bandwidth_gbps": 384,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    # ── AMD RDNA 3 ────────────────────────────────────────────────────────────
    {"name": "AMD RX 7900 XTX",          "vram_gb": 24,  "bandwidth_gbps": 960,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7900 XT",           "vram_gb": 20,  "bandwidth_gbps": 800,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7900 GRE",          "vram_gb": 16,  "bandwidth_gbps": 576,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7800 XT",           "vram_gb": 16,  "bandwidth_gbps": 624,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7700 XT",           "vram_gb": 12,  "bandwidth_gbps": 432,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7600 XT",           "vram_gb": 16,  "bandwidth_gbps": 288,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    {"name": "AMD RX 7600",              "vram_gb": 8,   "bandwidth_gbps": 288,  "hw_type": "amd",    "category": "AMD RDNA 3"},
    # ── AMD RDNA 2 ────────────────────────────────────────────────────────────
    {"name": "AMD RX 6900 XT",           "vram_gb": 16,  "bandwidth_gbps": 512,  "hw_type": "amd",    "category": "AMD RDNA 2"},
    {"name": "AMD RX 6800 XT",           "vram_gb": 16,  "bandwidth_gbps": 512,  "hw_type": "amd",    "category": "AMD RDNA 2"},
    {"name": "AMD RX 6800",              "vram_gb": 16,  "bandwidth_gbps": 512,  "hw_type": "amd",    "category": "AMD RDNA 2"},
    {"name": "AMD RX 6700 XT",           "vram_gb": 12,  "bandwidth_gbps": 384,  "hw_type": "amd",    "category": "AMD RDNA 2"},
    # ── Intel Arc (Battlemage, 2024–2025) ─────────────────────────────────────
    # Arc B580: 12 GB GDDR6, 192-bit, 456 GB/s — best value AI card at launch
    # Arc B770: 16 GB GDDR6, 256-bit, 608 GB/s — announced, ships Q2 2025
    # Arc A770: 16 GB GDDR6, 256-bit, 560 GB/s — Alchemist (2022), still relevant
    {"name": "Intel Arc B770 (16 GB)",   "vram_gb": 16,  "bandwidth_gbps": 608,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc B580 (12 GB)",   "vram_gb": 12,  "bandwidth_gbps": 456,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A770 (16 GB)",   "vram_gb": 16,  "bandwidth_gbps": 560,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A770 (8 GB)",    "vram_gb": 8,   "bandwidth_gbps": 560,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A750",           "vram_gb": 8,   "bandwidth_gbps": 512,  "hw_type": "intel",  "category": "Intel Arc"},
    # ── Qualcomm Snapdragon X (Windows ARM laptops, llama.cpp Vulkan) ─────────
    # Bandwidth = LPDDR5X spec; usable RAM ~85% of total (OS overhead).
    # Snapdragon X Elite X1E-84-100: 45/64 GB LPDDR5X, 136 GB/s
    # Snapdragon X Plus X1P-64-100:  32/64 GB LPDDR5X, 120 GB/s
    {"name": "Snapdragon X Elite (64 GB)","vram_gb": 64, "bandwidth_gbps": 136,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Elite (45 GB)","vram_gb": 45, "bandwidth_gbps": 136,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Plus (64 GB)", "vram_gb": 64, "bandwidth_gbps": 120,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Plus (32 GB)", "vram_gb": 32, "bandwidth_gbps": 120,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    # ── CPU Only ─────────────────────────────────────────────────────────────
    {"name": "CPU only — DDR5 laptop",     "vram_gb": 16,  "bandwidth_gbps": 68,  "hw_type": "cpu",    "category": "CPU Only"},
    {"name": "CPU only — DDR5 desktop",    "vram_gb": 32,  "bandwidth_gbps": 80,  "hw_type": "cpu",    "category": "CPU Only"},
    {"name": "CPU only — DDR5 workstation","vram_gb": 128, "bandwidth_gbps": 120, "hw_type": "cpu",    "category": "CPU Only"},
]

# Index by name for fast lookup
GPU_BY_NAME: dict[str, dict] = {g["name"]: g for g in GPUS}


# ── Calculation helpers ───────────────────────────────────────────────────────
def calc_vram_gb(params_b: float, quant: str) -> float:
    """VRAM required in GB for one copy of the model weights."""
    return params_b * QUANT_BYTES[quant] * _OVERHEAD


def calc_speed_tps(
    active_b: float,
    quant: str,
    bandwidth_gbps: float,
    hw_type: str,
) -> float:
    """
    Estimated tokens per second (memory-bandwidth-bound inference).
    Uses active parameter count for MoE (only active experts are read per token).
    """
    active_gb = active_b * QUANT_BYTES[quant] * _OVERHEAD
    if active_gb < 0.01:
        return 0.0
    eff = _EFF.get(hw_type, 0.50)
    return (bandwidth_gbps / active_gb) * eff


# ── Public API ────────────────────────────────────────────────────────────────

def _load_models_raw() -> list[dict]:
    """
    Load open-weight model specs from the scraped CSV cache.
    Falls back to _MODELS_RAW if the cache doesn't exist yet.
    """
    if _LOCAL_CACHE.exists():
        df = pd.read_csv(_LOCAL_CACHE)
        rows = []
        for _, row in df.iterrows():
            tag_str = str(row.get("tags", "")) if pd.notna(row.get("tags")) else ""
            tags = [t for t in tag_str.split(",") if t]
            rows.append({
                "name":      str(row["name"]),
                "family":    str(row["family"]),
                "params_b":  float(row["params_b"]),
                "active_b":  float(row["active_b"]),
                "context_k": int(row["context_k"]),
                "quality":   float(row["quality"]),
                "license":   str(row["license"]),
                "tags":      tags,
                "moe":       bool(row["moe"]),
            })
        return rows
    return _MODELS_RAW


def get_local_df(
    quant: str = "Q4",
    vram_gb: float = 24.0,
    bandwidth_gbps: float = 1008.0,
    hw_type: str = "nvidia",
    tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of all models enriched with hardware-specific columns.

    Columns added:
        vram_req_gb   - VRAM required at selected quantization
        speed_tps     - estimated tokens/second on the given hardware
        fits          - "yes" | "tight" (< 1 GB headroom) | "no"
    """
    models = _load_models_raw()
    rows = []
    for m in models:
        if tags:
            if not any(t in m["tags"] for t in tags):
                continue
        vram_req = calc_vram_gb(m["params_b"], quant)
        speed    = calc_speed_tps(m["active_b"], quant, bandwidth_gbps, hw_type)
        headroom = vram_gb - vram_req
        if headroom >= 1.0:
            fits = "yes"
        elif headroom >= 0:
            fits = "tight"
        else:
            fits = "no"
        rows.append({
            **m,
            "vram_req_gb": round(vram_req, 2),
            "speed_tps":   round(speed, 1),
            "fits":        fits,
            "tags_str":    ", ".join(m["tags"]) if m["tags"] else "general",
        })
    df = pd.DataFrame(rows)
    df["family_color"] = df["family"].map(FAMILY_COLORS).fillna(DEFAULT_FAMILY_COLOR)
    return df


def get_gpu_options() -> list[dict]:
    """Grouped options list for a Dash Dropdown."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for g in GPUS:
        groups[g["category"]].append({"label": g["name"], "value": g["name"]})
    return [
        {"label": cat, "value": cat, "disabled": True, **{}}
        if False else  # group headers not supported in plain Dropdown — flatten instead
        item
        for cat, items in groups.items()
        for item in items
    ]
