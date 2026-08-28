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
import re
from pathlib import Path
from typing import NamedTuple
import pandas as pd

from data.pending_models import merge_pending
from components.charts.constants import (
    PROVIDER_COLORS, DEFAULT_COLOR, canonical_provider,
)

_LOCAL_CACHE = Path(__file__).parent / "raw" / "aa_local_models.csv"

# VRAM assumed when the user has not entered a figure. One constant, because the
# two render paths used to disagree: a cleared box meant 32 GB on the public site
# and 8 GB in Dash, so the same action produced two different answers to "which
# models fit my hardware".
DEFAULT_VRAM_GB = 32.0
DEFAULT_GPU_COUNT = 1
DEFAULT_BANDWIDTH_GBPS = 1792.0
# Peak dense FP16 of the same default card. Named rather than inlined for the
# same reason as the three constants above: the two render paths each used to
# carry their own hardware defaults and answered "which models fit?" two
# different ways. A render path that cannot find this figure falls back to a
# bandwidth-only estimate, which is a quieter wrong answer than a crash.
DEFAULT_FP16_TFLOPS = 209.5


def effective_bandwidth(bandwidth_gbps: float, gpu_count: int) -> float:
    """Memory bandwidth for a single-stream decode across `gpu_count` cards.

    Which is: the bandwidth of ONE card.

    This used to be `bw * (1 + (n-1) * 0.85)`, copy-pasted verbatim into four
    call sites with no comment, predicting 6.95x single-stream throughput on 8
    GPUs. The contradiction needs no benchmark to see — the same four functions
    pool VRAM as `vram_per_gpu * num_gpus`, which is a LAYER split, while the
    bandwidth sum assumes TENSOR parallelism. Two mutually exclusive deployments,
    three lines apart. Under a layer split each card holds different layers and
    they run in sequence, so tokens per second is set by one card's bandwidth;
    what the extra cards buy you is the ability to hold a bigger model at all.

    Kept as a named function so the four sites cannot drift again, and so this
    reasoning has somewhere to live.
    """
    return float(bandwidth_gbps)

# ── Quantization ─────────────────────────────────────────────────────────────
# Q6 is new here: k-quants are the formats readers actually download, and Q6_K
# is the one that sits between "basically lossless" and the Q5/Q4 working range.
QUANT_LEVELS = ["FP16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]

# Levels where the weight loss is severe enough to change answers, not just
# rounding.
#
# WHY THIS EXISTS. Quantisation is the one control on this tab that moves two
# columns and not the third: dropping to Q2 shrinks vram_req_gb and raises
# speed_tps, while `quality` — an Artificial Analysis benchmark run at the
# model's native precision — does not move at all, because nothing in
# get_local_df touches it. The page therefore rendered aggressive quantisation
# as free, and a reader optimising on what was displayed would pick Q2 every
# time.
#
# The missing term is real but it is NOT ours to invent. Degradation is
# model-specific, AA publishes no quantised scores, and a generic penalty curve
# applied to a scraped benchmark would propagate a guessed number into the value
# rankings and the Agent Stack recommender — the hand-set-constant pattern this
# codebase keeps paying for, and exactly what data/pending_models.py refuses to
# do with an unbenchmarked model.
#
# So the cost is disclosed rather than modelled: the option says so at the point
# of choice, and captions.py says what the score does and does not reflect.
QUANT_LOSSY = ("Q3", "Q2")


def quant_options() -> list[dict]:
    """Quantization choices as {label, value}, for both render paths.

    One source, because app.py and docs/app.js built this control separately and
    a warning that lives in only one of them is a warning the deployed site does
    not carry.
    """
    return [
        {"label": f"{q} (lossy)" if q in QUANT_LOSSY else q, "value": q}
        for q in QUANT_LEVELS
    ]

# Bits per weight in a SHIPPED file, not the nominal bit width.
#
# This used to be the nominal width (Q4 = 0.5 bytes exactly). Real GGUF files
# are 6–53% fatter, for two reasons that are both structural rather than
# incidental: k-quant super-blocks carry 6-bit scales and mins on top of the
# packed weights (llama.cpp PR #1684 — Q2_K is 2.5625 bpw, Q3_K 3.4375, Q4_K
# 4.5, Q5_K 5.5, Q6_K 6.5625 BEFORE the mix), and the "_M" mixes every
# distributor ships keep output.weight and token_embd at Q6_K or Q8_0.
#
# MEASURED from bartowski/*-GGUF published file sizes divided by parameter
# count, averaged over Llama-3.1-8B, Qwen2.5-32B and Llama-3.1-70B (HF API,
# 2026-08-26); consistent to ±1.5% across the three sizes.
#
# The single largest consequence: FP16 → Q2 is a 5.24x byte ratio, not 8.0x, so
# calc_vram_gb was understating Q2 by 53% and Q4 by 21%.
GGUF_BPW: dict[str, float] = {
    "FP16": 16.00,   # f16, exact
    "Q8":    8.51,   # Q8_0    nominal 8 → 1.06x (32 int8 + one fp16 scale)
    "Q6":    6.59,   # Q6_K    nominal 6 → 1.10x
    "Q5":    5.69,   # Q5_K_M  nominal 5 → 1.14x
    "Q4":    4.86,   # Q4_K_M  nominal 4 → 1.21x
    "Q3":    3.93,   # Q3_K_M  nominal 3 → 1.31x
    "Q2":    3.06,   # Q2_K    nominal 2 → 1.53x
}

# Bytes per parameter at each quantization level.
QUANT_BYTES: dict[str, float] = {q: b / 8.0 for q, b in GGUF_BPW.items()}

# ── The streaming penalty: why quantising does not buy its byte ratio ────────
#
# tok/s scales as bytes ** -_GAMMA, not bytes ** -1. Low-bit kernels get worse
# memory-level parallelism: the super-block scales live in a separate region
# from the packed nibbles, so the read is two strided streams instead of one.
#
# It is NOT an ALU ceiling. Batch-1 decode has an arithmetic intensity of
# 1.0 FLOP/byte at FP16 and 5.24 at Q2, against a machine balance of 24 (M4
# base) to 295 (H100 SXM) — decode never crosses the roofline knee at any
# quantisation level on any device in GPUS. See decode_roofline().
#
# _GAMMA = 0.77 ± 0.12 (n=9): the uniform-affine / legacy-GGUF sub-population of
# a 15-sweep pooled fit spanning MLX on M3 Ultra, exllamav2 on RTX 4090 and
# 3090 Ti, and llama.cpp on M1 Pro / M1 Max / M2 Max / M2 Ultra / RTX 4080 /
# Xeon 8488C.
#
# WHY NOT THE POOLED 0.62. Because it is arithmetically impossible. _MBU_FP16
# below is FP16 bandwidth utilisation, so a level's achieved MBU is
# _MBU_FP16 * k_q. Back-solving the four MEASURED consumer-NVIDIA Q4_0 MBUs
# (4090 0.706, 3090 Ti 0.649, 3090 0.646, 5090 0.619; llama.cpp discussion
# #15013) through k_q(Q4_0) at gamma=0.62 requires an FP16 MBU of 1.02–1.13 —
# above theoretical peak. At 0.77 it gives 0.82–0.94, which is what a large
# sequential GDDR6X read actually achieves. Independently confirmed by the one
# dataset that measured BOTH formats on the same chips (llama.cpp #4167,
# Apple): the ratio Q4_0-MBU / F16-MBU is 0.810 / 0.744 / 0.918 / 0.651, mean
# 0.78, against 0.751 predicted here. Two independent lines land on 0.77–0.80.
#
# WHAT THIS GETS WRONG, and it is not small. The other sub-population — GGUF
# K-QUANTS specifically (Q6_K…Q2_K) — measures gamma = 0.41 ± 0.13 (n=6),
# because those kernels carry a per-format penalty on top of the byte count. So
# for a reader running ollama with k-quants the Q6/Q5/Q3/Q2 columns read up to
# ~1.7x fast. The FP16 and Q8 columns are right either way. If this dashboard
# ever gains a runtime control, the k-quant branch is one line: _GAMMA = 0.41.
#
# REALITY IS ALSO NOT MONOTONE and this table deliberately is. Reproduced from
# llama.cpp PR #1684: Q5_K_M is ~10% SLOWER than the 14%-larger Q6_K, and
# Q3_K_M ~6% slower than the 20%-larger Q4_K_M — "the memory access pattern is
# more important for performance than the amount of computation" (ikawrakow).
# The ordering also reverses on x86 CPU. No smooth curve can be right about
# individual formats; this one is right about the trend and wrong by up to
# ±25% about any single level.
_GAMMA = 0.77

# k_q = (bpw / 16) ** (1 - _GAMMA). Weight-streaming efficiency of each level as
# a fraction of FP16's. Net speedup vs FP16 is (16/bpw) * k_q, in the comments.
_QUANT_STREAM_EFF: dict[str, float] = {
    q: (bpw / 16.0) ** (1.0 - _GAMMA) for q, bpw in GGUF_BPW.items()
}

# Fraction of PEAK memory bandwidth an FP16 decode achieves.
#
# RENAMED from _EFF and REBASED. The old values were on an implicit Q4 basis —
# the old formula divided by QUANT_BYTES[quant] with no streaming term, so the
# constant absorbed whatever penalty existed at the quant it was tuned at — so
# rows with no direct FP16 measurement are the old value divided by
# _QUANT_STREAM_EFF["Q4"], which leaves them meaning exactly what they meant.
#
# nvidia and apple are no longer estimates. They are the standard MBU metric,
# (weights + KV bytes) / TPOT / peak bandwidth, from published llama-2-7B
# llama-bench scoreboards:
#   apple  0.82 ← F16 MBU measured on 7 chips: M1 Pro 0.859, M3 Max 0.845,
#                 M2 Max 0.830, M5 Max 0.814, M4 Max 0.781, M1 Max 0.776,
#                 M2 Ultra 0.691 (mean 0.80). llama.cpp discussion #4167.
#                 Independently bounded by arXiv:2502.05317, which measures
#                 Apple M1–M4 hitting ~85% of peak on STREAM. The audit's
#                 "apple > nvidia is counter-intuitive" flag is RESOLVED: the
#                 old nvidia=0.55 was simply never measured.
#   nvidia 0.85 ← back-solved from the Q4_0 MBUs above ÷ k_q(Q4_0)=0.751:
#                 0.94 / 0.86 / 0.86 / 0.82, mean 0.87, taken as 0.85.
#                 llama.cpp discussion #15013.
#
# KNOWN OVER-PREDICTION, not fixed here: the same scoreboard gives DATACENTER
# NVIDIA an MBU of 0.31 (H100 SXM) and 0.36 (A100 80GB) on a 3.8 GB model.
# Those cards are not bandwidth-limited at 270+ tok/s; they are latency-bound on
# ~10 kernel launches per layer (arXiv:2605.30571 measures 3.05 ms/step of
# launch tax on an H100, 20.6% of step time). One constant cannot span 0.87 and
# 0.31, and the 0.31 is a small-model artefact that must NOT be generalised to a
# 70B — so this model over-predicts SMALL models on datacenter parts. A
# per-preset MBU is the fix when someone measures one; splitting it on a guess
# here would be the same unmeasured constant twice.
#
# amd / intel / qualcomm / cpu remain UNVALIDATED ESTIMATES, rebased only.
# Treat the ordering as indicative and any absolute tok/s as ±30%.
_MBU_FP16: dict[str, float] = {
    "nvidia":   0.85,   # MEASURED (consumer)
    "apple":    0.82,   # MEASURED
    "amd":      0.66,   # 0.50 / 0.760, unvalidated
    "qualcomm": 0.72,   # 0.55 / 0.760, unvalidated
    "intel":    0.66,   # 0.50 / 0.760, unvalidated
    "cpu":      0.39,   # 0.30 / 0.760, unvalidated
}

# Effective bandwidth of the KV-cache read, as a fraction of peak. Lower than
# the weight stream because it is a paged/strided read, and because it silently
# absorbs the growth of attention COMPUTE with context, which is not a bandwidth
# term at all — so treat it as an empirical lumped constant, not a physical
# bandwidth. Two independent fits: a 3-parameter fit of the 42-cell Qwen-32B
# grid on M3 Ultra (ml-explore/mlx #3209) gives 395 GB/s = 49% of peak (mean
# residual 0.46%, max 1.64%); a first-party 5-point context sweep on an M1 Max
# gives 228–250 GB/s = 57–62% across three quant levels. 0.55 is the midpoint.
_KV_MBU = 0.55

# Achieved / peak dense FLOPs for the decode GEMMs. FIT, not measured: least
# squares over 86 inter-token-latency points from 11 published NVIDIA NIM
# concurrency sweeps. Weakly identified — dropping the compute term entirely
# only worsens that fit from 12.6% to 13.2% mean error, because in NVIDIA's
# benchmark band the sublinearity is almost entirely the KV read.
#
# NOTE this is NOT Pope et al.'s 0.14 "low-batch decode MFU" (MLSys 2023). That
# figure is an OUTPUT of a memory-bound regime, not a ceiling — and this model
# reproduces it: an 8B at Q4 on a 4090 predicts ~85 tok/s memory-bound against
# a ~6,200 tok/s compute roof, i.e. ~1.4% of peak FLOPs delivered. Use 0.14 as
# a regression target, never as an input.
_MFU_DECODE = 0.60

# Compute roof for a dequantize-then-multiply CPU runtime (llama.cpp on x86 or
# ARM CPU). There the per-weight instruction stream is 2 FLOPs PLUS unpack work
# that GROWS as bit width shrinks, so a FLOP-based roof does not describe it.
# The measured invariant is a near-constant rate of low-bit weights per second
# which FALLS with bit width: T-MAC (EuroSys '25, arXiv:2407.00088, Table 7)
# gives Snapdragon X Elite CPU 71.7 Gweights/s at 4-bit and 63.3 at 2-bit.
#
# This is the ONE place decode is genuinely compute-bound. Going 4-bit → 2-bit
# halves the bytes but makes llama.cpp SLOWER on three separate edge CPUs
# (0.88x, 0.84x, 0.81x) — impossible under a memory-bound model — and swapping
# in T-MAC's lookup-table kernel, same hardware and same bytes with only the
# dequant removed, recovers 1.47–1.63x. That is a categorical refutation of
# bandwidth-only for this corner.
#
# TRANSPLANTED, and flagged: measured on one ARM CPU family, applied to three
# generic DDR5 presets that name no CPU. It is a max() guard, not a headline —
# at _MBU_FP16["cpu"] = 0.39 the memory roof still binds for every model in the
# catalogue, and this term only stops a tiny model on a wide bus from charting a
# figure no CPU reaches.
_LOWBIT_WEIGHT_RATE_GW_S: dict[str, dict[float, float]] = {
    "cpu": {16.0: 71.7, 4.0: 71.7, 2.0: 63.3},
}

# ── VRAM sizing constants ────────────────────────────────────────────────────
GIB = 1024 ** 3

# Runtime overhead, REPLACING the old _OVERHEAD = 1.18 multiplier.
#
# 1.18 was the wrong SHAPE, not the wrong value. Real overhead is close to
# CONSTANT; a multiplier makes it proportional to weight bytes, so it added
# 0.11 GB to a 1B/Q4 model (~10x too little against a ~1–1.5 GiB floor) and
# 36.9 GB to a 405B/Q4 (~5–9x too much). It landed within 2x only in the 7–16B
# band, which is presumably where it was eyeballed. It also silently scaled
# _EFF by 1/1.18 in the SPEED path, where 18% of the weights being re-read
# every token is not a thing that happens.
#
# CUDA_CTX_GIB: the driver context, PER GPU, allocated outside the framework's
#   own accounting. vLLM issue #12059 reports 384 MiB on an RTX 4090 and
#   ~520 MiB on an H100, and its author proposes budgeting a flat 0.5 GiB while
#   noting "the exact discrepancy depends on the GPU type and is hard to
#   measure". ESTIMATE, and someone else's.
# WORKSPACE_GIB: activation / compute buffers. Sized by batch and prefill chunk,
#   NOT by parameter count. vLLM PR #10511's own profiling log for a Llama-3-8B
#   BF16 run: "non_torch_memory takes 0.18GiB; PyTorch activation peak memory
#   takes 1.26GiB". llama.cpp discussion #9936 prints a 507.00 MiB compute
#   buffer at n_batch 2048 and 126.75 MiB at n_batch 128. 1.0 GiB is a midpoint
#   for a single-stream local run; the honest range is 0.5–2.0 GiB. ESTIMATE.
CUDA_CTX_GIB = 0.5
WORKSPACE_GIB = 1.0

# Bytes per cached KV element. llama.cpp block formats include the per-block
# scale, so 8-bit KV is 1.0625 bytes and not 1.0 — using 1.0 understates it 6%.
#   q8_0 = 32 int8 + one fp16 scale = 34 B / 32 elems = 1.0625
#   q5_1 = 32x5 bits + two fp16 scales = 24 B / 32     = 0.75
#   q4_0 = 32 int4 + one fp16 scale = 18 B / 32        = 0.5625
# vLLM's kv_cache_dtype="fp8" is a flat 1.0 (no block scale).
KV_BYTES_PER_ELEM: dict[str, float] = {
    "FP16": 2.0000, "FP8": 1.0000, "Q8": 1.0625, "Q5": 0.7500, "Q4": 0.5625,
}

# The KV cache is FP16 in llama.cpp, MLX and vLLM BY DEFAULT and is not affected
# by weight quantisation. It is deliberately NOT wired to the quant control:
# `-ctk q8_0` is a separate flag most readers never set, and coupling them would
# make aggressive weight quantisation look like it shrinks the cache too — the
# exact "quantising is free" defect QUANT_LOSSY exists to disclose.
DEFAULT_KV_QUANT = "FP16"

# Context assumed when the reader has not chosen one. 8k is a working session,
# NOT the model's advertised maximum: `context_k` is a ceiling, and sizing at it
# makes Llama 4 Scout demand 183 GiB of KV and renders almost the whole
# catalogue unrunnable. Every per-model figure is capped at that model's own
# context_k regardless, because a 4k model cannot run 32k.
DEFAULT_CONTEXT_TOKENS = 8192
CONTEXT_CHOICES = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ── Concurrency ──────────────────────────────────────────────────────────────
# Per-session decode floors, tokens/sec. These are MLPerf Inference latency
# constraints, not a number this project chose.
#
# Practitioners do not maximise throughput; they maximise throughput SUBJECT TO
# a per-token latency bound. Reporting the throughput-maximising batch size
# without one recommends a configuration nobody would actually run.
SLO_FLOORS_TPS: dict[str, float] = {
    # MLPerf Inference v4.0/v5.0, Llama-2-70B Server: p99 TTFT <= 2 s,
    # p99 TPOT <= 200 ms. (Llama-3.1-405B uses 175 ms = 5.7 tok/s.) This floor
    # coincides with actual human silent reading rather than a multiple of it:
    # Brysbaert 2019, a meta-analysis of 190 studies / 18,573 participants, puts
    # adult non-fiction silent reading at 238 wpm = ~5.3 tok/s at 0.75 words
    # per token.
    "batch":       5.0,
    # DEFAULT. MLPerf Inference v5.1, Llama-3.1-8B Server: p99 TPOT <= 100 ms.
    # MLCommons' stated rationale: "A TPOT of 100 ms is effectively ~480 words
    # per minute, significantly faster than typical human reading rates."
    "interactive": 10.0,
    # MLPerf Inference v5.1, Llama-3.1-8B Interactive: p99 TPOT <= 30 ms, for
    # "use cases where user engagement and responsiveness are paramount, such as
    # code assistants and real-time creative tools".
    "realtime":    33.3,
}
DEFAULT_SLO = "interactive"

# NOT A CONTROL, deliberately. The tab briefly shipped an SLO dropdown and it
# was the wrong question to put in front of a reader: it was labelled SESSIONS
# but set a latency policy, so choosing "Interactive" produced a session count
# nobody had asked for and could not predict from the label. The floor is fixed
# at DEFAULT_SLO now and named in the caption — the reader picks what they want
# to SEE, and the floor is a stated assumption behind it, which is the shape
# every other assumption on this dashboard takes. All three floors stay because
# optimal_concurrency() still takes the argument.

# ── What the speed column MEANS ──────────────────────────────────────────────
# One number on screen at a time, and a control that says which one. The tab
# briefly showed BOTH at once — a single-stream figure and an aggregate figure,
# in the same gutter, differing by 2-4x. That is the ambiguity
# test_the_tok_s_figures_say_which_of_the_two_they_are exists to prevent, moved
# from "unlabelled" to "labelled twice" rather than fixed.
SPEED_MODES: dict[str, str] = {
    "single":     "Single stream",
    "throughput": "Max throughput",
}
DEFAULT_SPEED_MODE = "single"


def context_options() -> list[dict]:
    """Context-length choices as {label, value}, for both render paths.

    Same one-source rule as quant_options(): app.py and docs/app.js built the
    quantisation control separately once and the lossy warning reached only one
    of them, so a control that exists in two shells is declared in neither.
    """
    # "default" travels with the options so the browser does not have to name a
    # number of its own. A JS-side `|| 8192` would be a second copy of
    # DEFAULT_CONTEXT_TOKENS, which is the drift this project keeps paying for.
    return [{"label": f"{n // 1024}k", "value": n,
             "default": n == DEFAULT_CONTEXT_TOKENS} for n in CONTEXT_CHOICES]


def speed_mode_options() -> list[dict]:
    """Speed-metric choices as {label, value}, for both render paths — same
    one-source rule as quant_options(), for the same reason."""
    return [{"label": v, "value": k, "default": k == DEFAULT_SPEED_MODE}
            for k, v in SPEED_MODES.items()]


def speed_columns(speed_mode: str) -> tuple[str, str]:
    """(value column, label) for the metric the reader selected.

    One place, because the scatter's bubble size, the compat chart's gutter and
    both hovers all have to agree about which of the two tok/s figures is the
    headline — and they are 2-4x apart.
    """
    if speed_mode == "throughput":
        return "total_tps", "max throughput"
    return "speed_tps", "single stream"


# NOT A CONSTANT ANY MORE, and the reason is worth keeping.
#
# vLLM's documented gpu_memory_utilization default of 0.92 reserves the last 8%
# of VRAM for driver context, fragmentation and activation scratch. This file
# already models exactly that, itemised, as CUDA_CTX_GIB + WORKSPACE_GIB — so
# applying both charged for the same thing twice. That is not a rounding
# quibble: `fits` used the raw capacity while optimal_concurrency() used the
# discounted one, and 3.4% of the rows the chart drew as runnable came back
# with sessions=0, putting "Speed: 484 tok/s single stream" and "Sessions: x0
# concurrent" in the same tooltip. Both now spend out of the same wallet.

# Display bound, NOT a measured limit. Every published sweep found stops at or
# below it (NVIDIA NIM <= 250, llama.cpp batched-bench <= 256). Past it the
# model is extrapolating beyond all available evidence.
MAX_REPORTED_CONCURRENCY = 256


# ── Color palette by model family ────────────────────────────────────────────
# A "family" in the open-weight catalog is a lab — the same entity the rest of
# the dashboard colours by provider. This used to be a separate hand-kept palette,
# which meant one lab read as two different colours depending on the tab (Meta was
# indigo here and blue on Overview) and, worse, every lab added upstream since the
# list was written fell through to grey: 40 of 97 catalog rows, including Kimi,
# MiniMax, Z AI, Xiaomi and NVIDIA. Deriving it from PROVIDER_COLORS keeps the two
# in step by construction.
_FAMILY_ALIASES: dict[str, str] = {
    "Allen AI": "Allen Institute for AI",
    "Moonshot": "Kimi",          # Moonshot AI ships the Kimi models
}

# Labs that only ever appear in the open-weight catalog, so they have no provider
# entry to inherit. Kept distinct from each other and from the spotlight palette.
# Labs that only ever appear in the open-weight catalog, so they have no provider
# entry to inherit.
#
# The block below the original six arrived with the move to the leaderboard
# source: it carries the 77 open-weight models no API host sells, and most of
# those come from labs the hosted catalogue never mentions. Hues were assigned by
# farthest-point search in CIE L*a*b* against every colour that can appear beside
# them on this tab, each >= 3.0:1 contrast on the #111111 surface. Separation
# degrades down the list — 44 families is well past what colour alone can carry,
# which is why the compat chart labels every bar and the scatter leans on
# position rather than hue.
_LOCAL_ONLY_COLORS: dict[str, str] = {
    "TII":         "#7048e8",
    "01.AI":       "#d9480f",
    "InternLM":    "#74c0fc",
    "HuggingFace": "#fab005",
    "SOLAR":       "#ff6b6b",
    "Liquid AI":   "#38bdf8",
    "Ornith AI":   "#f0abfc",
    # ── unhosted open-weight labs, added 2026-08-17 ──
    "AI9Stars":                              "#f97316",
    "MBZUAI Institute of Foundation Models": "#f472b6",
    "Motif Technologies":                    "#fdba74",
    "Nanbeige":                              "#a3e635",
    "Naver":                                 "#60a5fa",
    "OpenBMB":                               "#34d399",
    "Prime Intellect":                       "#f0abfc",
    "SK Telecom":                            "#c084fc",
    "ServiceNow":                            "#22d3ee",
    "TII UAE":                               "#facc15",
    "Trillion Labs":                         "#cbd5e1",
}

FAMILY_COLORS: dict[str, str] = {
    **_LOCAL_ONLY_COLORS,
    **PROVIDER_COLORS,
    **{alias: PROVIDER_COLORS[target]
       for alias, target in _FAMILY_ALIASES.items() if target in PROVIDER_COLORS},
}
DEFAULT_FAMILY_COLOR = DEFAULT_COLOR


def family_color(family: str) -> str:
    """Colour for a catalog family, resolving upstream renames the same way the
    provider palette does (e.g. xAI → SpaceXAI)."""
    name = _FAMILY_ALIASES.get(family, family)
    return FAMILY_COLORS.get(name) or PROVIDER_COLORS.get(
        canonical_provider(name), DEFAULT_FAMILY_COLOR
    )


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
    # 5070 Ti: 256-bit × 28 Gbps = 896 GB/s | 5070: 192-bit × 28 Gbps = 672 GB/s
    # 5060 Ti: 128-bit × 28 Gbps = 448 GB/s | 5060: 128-bit × 28 Gbps = 448 GB/s
    #   (the old 864 and 288 assumed 27 and 18 Gbps; GDDR7 ships neither)
    {"name": "NVIDIA RTX 5090",          "vram_gb": 32,  "bandwidth_gbps": 1792, "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5080",          "vram_gb": 16,  "bandwidth_gbps": 960,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5070 Ti",       "vram_gb": 16,  "bandwidth_gbps": 896,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5070",          "vram_gb": 12,  "bandwidth_gbps": 672,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060 Ti 16GB",  "vram_gb": 16,  "bandwidth_gbps": 448,  "si": "GB206", "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060 Ti",       "vram_gb": 8,   "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    {"name": "NVIDIA RTX 5060",          "vram_gb": 8,   "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 50"},
    # ── NVIDIA RTX 40 (Ada Lovelace, GDDR6X) ─────────────────────────────────
    {"name": "NVIDIA RTX 4090",          "vram_gb": 24,  "bandwidth_gbps": 1008, "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4080 Super",    "vram_gb": 16,  "bandwidth_gbps": 736,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4080",          "vram_gb": 16,  "bandwidth_gbps": 717,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Ti Super", "vram_gb": 16,  "bandwidth_gbps": 672,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Ti",       "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070 Super",    "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4070",          "vram_gb": 12,  "bandwidth_gbps": 504,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060 Ti 16GB",  "vram_gb": 16,  "bandwidth_gbps": 288,  "si": "AD106", "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060 Ti",       "vram_gb": 8,   "bandwidth_gbps": 288,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    {"name": "NVIDIA RTX 4060",          "vram_gb": 8,   "bandwidth_gbps": 272,  "hw_type": "nvidia", "category": "NVIDIA RTX 40"},
    # ── NVIDIA RTX 30 (Ampere, GDDR6X) ───────────────────────────────────────
    {"name": "NVIDIA RTX 3090 Ti",       "vram_gb": 24,  "bandwidth_gbps": 1008, "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3090",          "vram_gb": 24,  "bandwidth_gbps": 936,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080 Ti",       "vram_gb": 12,  "bandwidth_gbps": 912,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080 12GB",     "vram_gb": 12,  "bandwidth_gbps": 912,  "si": "GA102-3080", "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3080",          "vram_gb": 10,  "bandwidth_gbps": 760,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3070 Ti",       "vram_gb": 8,   "bandwidth_gbps": 608,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 3070",          "vram_gb": 8,   "bandwidth_gbps": 448,  "hw_type": "nvidia", "category": "NVIDIA RTX 30"},
    {"name": "NVIDIA RTX 2080 Ti",       "vram_gb": 11,  "bandwidth_gbps": 616,  "hw_type": "nvidia", "category": "NVIDIA RTX 20"},
    # ── NVIDIA Data Center (Blackwell) ───────────────────────────────────────
    # B300 SXM (Blackwell Ultra, 2025): 288 GB HBM3e, 8 TB/s — refreshed B200
    # B200 SXM: 192 GB HBM3e, 8 TB/s. NVIDIA ships B200 only as SXM on an HGX
    #   baseboard — a 1,000 W part cannot be fed by a PCIe slot — so the old
    #   "B200 PCIe" row was a duplicate of a card that does not exist, and
    #   "GB200 NVL2" was a TWO-GPU aggregate (384 GB / 16 TB/s) handed to a
    #   function whose docstring refuses to sum bandwidth across cards.
    {"name": "NVIDIA B300 SXM",          "vram_gb": 288, "bandwidth_gbps": 8000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Blackwell)"},
    {"name": "NVIDIA B200 SXM",          "vram_gb": 192, "bandwidth_gbps": 8000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Blackwell)"},
    # ── NVIDIA Data Center (Hopper) ──────────────────────────────────────────
    # H200 SXM and H200 NVL/PCIe are both 141 GB @ 4.8 TB/s — only the TDP
    #   differs. The old 3.35 TB/s on the PCIe row was the H100 SXM figure
    #   copied from the line below, under-predicting that card by 30%.
    # H100 NVL: NVIDIA now lists this as ONE 94 GB / 3.9 TB/s card; the old
    #   188 GB / 7.8 TB/s was the two-card pair's aggregate.
    # GH200 (Grace Hopper Superchip, 144 GB HBM3e variant): 144 GB, 4.9 TB/s HBM
    {"name": "NVIDIA GH200 144GB",       "vram_gb": 144, "bandwidth_gbps": 4900,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H200 SXM",          "vram_gb": 141, "bandwidth_gbps": 4800,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H200 PCIe",         "vram_gb": 141, "bandwidth_gbps": 4800,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H100 NVL",          "vram_gb": 94, "bandwidth_gbps": 3938,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H100 SXM",          "vram_gb": 80,  "bandwidth_gbps": 3350,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    {"name": "NVIDIA H100 PCIe",         "vram_gb": 80,  "bandwidth_gbps": 2000,  "hw_type": "nvidia", "category": "NVIDIA Data Center (Hopper)"},
    # ── NVIDIA Data Center (Ada / Ampere) ────────────────────────────────────
    # L40S: 48 GB GDDR6, 864 GB/s (inference-optimised Ada successor to A100)
    # L40: 48 GB GDDR6, 864 GB/s | A40: 48 GB GDDR6, 696 GB/s
    # L4: 24 GB GDDR6, 300 GB/s — Ada inference card, common on GCP
    # A10: 24 GB GDDR6, 600 GB/s — Ampere inference card, common on AWS
    # T4:  16 GB GDDR6, 320 GB/s — Turing inference workhorse, still ubiquitous
    {"name": "NVIDIA L40S",              "vram_gb": 48,  "bandwidth_gbps": 864,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA L40",               "vram_gb": 48,  "bandwidth_gbps": 864,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA L4",                "vram_gb": 24,  "bandwidth_gbps": 300,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A40",               "vram_gb": 48,  "bandwidth_gbps": 696,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A100 80GB SXM",         "vram_gb": 80,  "bandwidth_gbps": 2039,  "si": "GA100", "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A100 40GB",         "vram_gb": 40,  "bandwidth_gbps": 1555,  "si": "GA100", "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA A10",               "vram_gb": 24,  "bandwidth_gbps": 600,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA V100 32GB",         "vram_gb": 32,  "bandwidth_gbps": 900,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    {"name": "NVIDIA T4",                "vram_gb": 16,  "bandwidth_gbps": 320,   "hw_type": "nvidia", "category": "NVIDIA Data Center (Ada/Ampere)"},
    # ── AMD Instinct (CDNA, Data Center) ─────────────────────────────────────
    # MI355X (CDNA 4, 2025): 288 GB HBM3e, 8 TB/s — Blackwell-class memory
    # MI325X (CDNA 3, late 2024): 256 GB HBM3e, 6 TB/s
    # MI300X (CDNA 3, 2023):     192 GB HBM3,   5.3 TB/s
    # MI250X (CDNA 2):           128 GB HBM2e, 3.2 TB/s — Frontier supercomputer
    {"name": "AMD Instinct MI355X",      "vram_gb": 288, "bandwidth_gbps": 8000,  "hw_type": "amd",    "category": "AMD Instinct (Data Center)"},
    {"name": "AMD Instinct MI325X",      "vram_gb": 256, "bandwidth_gbps": 6000,  "hw_type": "amd",    "category": "AMD Instinct (Data Center)"},
    {"name": "AMD Instinct MI300X",      "vram_gb": 192, "bandwidth_gbps": 5300,  "hw_type": "amd",    "category": "AMD Instinct (Data Center)"},
    {"name": "AMD Instinct MI250X",      "vram_gb": 128, "bandwidth_gbps": 3200,  "hw_type": "amd",    "category": "AMD Instinct (Data Center)"},
    # ── Intel Gaudi (Data Center) ────────────────────────────────────────────
    # Gaudi 3 (2024): 128 GB HBM2e, 3.7 TB/s
    # Gaudi 2 (2022): 96 GB HBM2e,  2.45 TB/s
    {"name": "Intel Gaudi 3",            "vram_gb": 128, "bandwidth_gbps": 3700,  "hw_type": "intel",  "category": "Intel Gaudi (Data Center)"},
    {"name": "Intel Gaudi 2",            "vram_gb": 96,  "bandwidth_gbps": 2450,  "hw_type": "intel",  "category": "Intel Gaudi (Data Center)"},
    # ── NVIDIA Professional Workstation ──────────────────────────────────────
    {"name": "NVIDIA RTX 6000 Ada",      "vram_gb": 48,  "bandwidth_gbps": 960,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    {"name": "NVIDIA RTX 5000 Ada",      "vram_gb": 32,  "bandwidth_gbps": 576,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    {"name": "NVIDIA RTX A6000",             "vram_gb": 48,  "bandwidth_gbps": 768,  "hw_type": "nvidia", "category": "NVIDIA Professional"},
    # ── Apple Silicon ─────────────────────────────────────────────────────────
    # Unified memory = VRAM; bandwidth from Apple silicon spec pages.
    #
    # THE RECURRING DEFECT IN THIS BLOCK, now fixed, is assuming a bigger memory
    # tier implies the higher-bandwidth GPU bin. Apple does not sell them that
    # way. M3 Max 96 GB shipped ONLY with the 14C-CPU/30C-GPU part (300 GB/s)
    # while 48 GB is 16C/40C (400 GB/s) — counter-intuitive and correct. Rows
    # where the tier picks the LOWER bin carry an explicit "si" so gpu_compute()
    # does not hand them the big die's FLOPS.
    #
    # M3 Max: 30-core GPU 300 GB/s / 40-core 400 GB/s. Tiers 36/48/64/96/128.
    # M3 Ultra (Mac Studio 2025): 2× M3 Max → 819 GB/s. Tiers 96/256/512 —
    #   there was never a 192 GB M3 Ultra; 192 was the M2 Ultra maximum.
    # M4 Max: 32-core GPU 410 GB/s / 40-core 546 GB/s. Tiers 36/48/64/128.
    # M5 (MacBook Air, Mar 2026): 153 GB/s, up to 32 GB.
    # M5 Pro (MacBook Pro / Mac mini): 307 GB/s, up to 64 GB.
    # M5 Max: 32-core GPU 460 GB/s (36 GB only) / 40-core 614 GB/s. Tiers
    #   36/48/64/128 — NOT a single-tier GPU, as this comment used to claim.
    # M5 Ultra (Mac Studio, Aug 2026): 1.2 TB/s, tiers 96/256/512. See below.
    # M6 (Mac mini, Aug 2026): first 2 nm chip, 170 GB/s, tiers 16/24/32.
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
    {"name": "Apple M3 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 300,  "si": "M3 Max 30c", "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    # M3 Max 16-core GPU: 400 GB/s — MacBook Pro 16" high, Mac Studio
    {"name": "Apple M3 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (96 GB)",     "vram_gb": 96,  "bandwidth_gbps": 300,  "si": "M3 Max 30c", "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 400,  "hw_type": "apple",  "category": "Apple M3"},
    # M3 Ultra: 2× M3 Max (16-core) → 819 GB/s — Mac Studio (2025), up to 512 GB
    {"name": "Apple M3 Ultra (96 GB)",  "vram_gb": 96, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Ultra (256 GB)",  "vram_gb": 256, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    {"name": "Apple M3 Ultra (512 GB)",  "vram_gb": 512, "bandwidth_gbps": 819,  "hw_type": "apple",  "category": "Apple M3"},
    # ── M4 ──
    {"name": "Apple M4 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 120,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 120,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 (32 GB)",         "vram_gb": 32,  "bandwidth_gbps": 120,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Pro (24 GB)",     "vram_gb": 24,  "bandwidth_gbps": 273,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Pro (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 273,  "hw_type": "apple",  "category": "Apple M4"},
    # M4 Max 14-core GPU: 410 GB/s — MacBook Pro 14" / 16" base tier
    {"name": "Apple M4 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 410,  "si": "M4 Max 32c", "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    # M4 Max 16-core GPU: 546 GB/s — MacBook Pro 16" high, Mac Studio
    {"name": "Apple M4 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    {"name": "Apple M4 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 546,  "hw_type": "apple",  "category": "Apple M4"},
    # ── M5 ──
    # M5 base (MacBook Air, Mar 2026): 153.6 GB/s, up to 32 GB
    {"name": "Apple M5 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 153,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 153,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 (32 GB)",         "vram_gb": 32,  "bandwidth_gbps": 153,  "hw_type": "apple",  "category": "Apple M5"},
    # M5 Pro (MacBook Pro 14"/16", Mar 2026): 307 GB/s, up to 64 GB
    {"name": "Apple M5 Pro (24 GB)",     "vram_gb": 24,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Pro (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Pro (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 307,  "hw_type": "apple",  "category": "Apple M5"},
    # M5 Max (MacBook Pro 14"/16", Mar 2026): 614 GB/s, up to 128 GB
    {"name": "Apple M5 Max (36 GB)",     "vram_gb": 36,  "bandwidth_gbps": 460,  "si": "M5 Max 32c", "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (48 GB)",     "vram_gb": 48,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (64 GB)",     "vram_gb": 64,  "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Max (128 GB)",    "vram_gb": 128, "bandwidth_gbps": 614,  "hw_type": "apple",  "category": "Apple M5"},
    # ── M5 Ultra ── (Mac Studio, announced 2026-08-25; ships 22 Sep 2026)
    # Quad-die: two dual-die M5 Max joined by next-generation UltraFusion.
    # 36-core CPU (12 super + 24 performance), up to 80-core GPU with a Neural
    # Accelerator in every core, 32-core Neural Engine. 1.2 TB/s is the SAME on
    # both GPU bins (64-core base, 80-core top), so unlike M5 Max the memory
    # tier does not pick the bandwidth here — but 256 GB and 512 GB require the
    # 80-core bin, so per this file's convention every tier carries it.
    # Tiers Apple sells are 96 / 256 / 512 GB. There is NO 128 GB M5 Ultra.
    # https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
    #   ("1.2TB/s of unified memory bandwidth", "up to 512GB of unified memory")
    {"name": "Apple M5 Ultra (96 GB)",   "vram_gb": 96,  "bandwidth_gbps": 1200, "si": "M5 Ultra 80c", "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Ultra (256 GB)",  "vram_gb": 256, "bandwidth_gbps": 1200, "si": "M5 Ultra 80c", "hw_type": "apple",  "category": "Apple M5"},
    {"name": "Apple M5 Ultra (512 GB)",  "vram_gb": 512, "bandwidth_gbps": 1200, "si": "M5 Ultra 80c", "hw_type": "apple",  "category": "Apple M5"},
    # ── M6 ── (Mac mini, announced 2026-08-25; ships 22 Sep 2026)
    # Apple's first 2 nm chip. 12-core CPU (2 super + 4 performance + 6
    # efficiency), 12-core GPU with a Neural Accelerator in every core, dual
    # 16-core Neural Engine, 170 GB/s. The GPU is a SINGLE non-configurable bin,
    # so all three memory tiers share one silicon row. Tiers: 16 / 24 / 32 GB.
    # M6 has shipped only in the Mac mini — there is no M6 Pro/Max/Ultra yet.
    # https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
    #   ("170GB/s of unified memory bandwidth — a 10 percent increase over M5
    #     and a 2.5x increase over M1"; 68 × 2.5 = 170 confirms both ends)
    {"name": "Apple M6 (16 GB)",         "vram_gb": 16,  "bandwidth_gbps": 153,  "hw_type": "apple",  "category": "Apple M6"},
    {"name": "Apple M6 (24 GB)",         "vram_gb": 24,  "bandwidth_gbps": 170,  "hw_type": "apple",  "category": "Apple M6"},
    {"name": "Apple M6 (32 GB)",         "vram_gb": 32,  "bandwidth_gbps": 170,  "hw_type": "apple",  "category": "Apple M6"},
    # M5 Ultra: not yet announced (expected Mac Studio mid-2026)
    # ── Apple iPhone (on-device inference via llama.cpp Metal / Core ML) ────────
    # Named by chip, not device. Usable RAM ≈ total minus ~2 GB OS reservation.
    # Only models ≤ ~4 GB VRAM fit on phones — filter enforces this automatically.
    # Bandwidth from Apple silicon spec pages.
    {"name": "A16 (iPhone 14 Pro / 16e)", "vram_gb": 6,  "bandwidth_gbps": 51.2,   "hw_type": "apple",  "category": "Apple — iPhone"},
    {"name": "A17 Pro (iPhone 15 Pro)",   "vram_gb": 6,  "bandwidth_gbps": 51.2,   "hw_type": "apple",  "category": "Apple — iPhone"},
    {"name": "A18 Pro (iPhone 16 Pro)",   "vram_gb": 6,  "bandwidth_gbps": 60,   "hw_type": "apple",  "category": "Apple — iPhone"},
    # A19 Pro (iPhone 17 Pro, Sep 2025): 12 GB RAM, improved memory bandwidth
    {"name": "A19 Pro (iPhone 17 Pro)",   "vram_gb": 10, "bandwidth_gbps": 76.8,   "hw_type": "apple",  "category": "Apple — iPhone"},
    # ── AMD RDNA 4 (2025) ─────────────────────────────────────────────────────
    # RX 9070 XT: 16 GB GDDR6, 256-bit, 717 GB/s — RDNA4 flagship mainstream
    # RX 9070:    16 GB GDDR6, 256-bit, 640 GB/s
    # RX 9060 XT: 8/16 GB GDDR6, 128-bit, 384 GB/s — announced, mid-2025
    {"name": "AMD RX 9070 XT",           "vram_gb": 16,  "bandwidth_gbps": 640,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9070",              "vram_gb": 16,  "bandwidth_gbps": 640,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9060 XT (16 GB)",   "vram_gb": 16,  "bandwidth_gbps": 320,  "hw_type": "amd",    "category": "AMD RDNA 4"},
    {"name": "AMD RX 9060 XT (8 GB)",    "vram_gb": 8,   "bandwidth_gbps": 320,  "hw_type": "amd",    "category": "AMD RDNA 4"},
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
    {"name": "Intel Arc B770 (16 GB, unreleased)",   "vram_gb": 16,  "bandwidth_gbps": 608,  "si": "Intel Arc B770 (16 GB)", "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc B580 (12 GB)",   "vram_gb": 12,  "bandwidth_gbps": 456,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A770 (16 GB)",   "vram_gb": 16,  "bandwidth_gbps": 560,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A770 (8 GB)",    "vram_gb": 8,   "bandwidth_gbps": 512,  "hw_type": "intel",  "category": "Intel Arc"},
    {"name": "Intel Arc A750",           "vram_gb": 8,   "bandwidth_gbps": 512,  "hw_type": "intel",  "category": "Intel Arc"},
    # ── Qualcomm Snapdragon X (Windows ARM laptops, llama.cpp Vulkan) ─────────
    # Bandwidth = LPDDR5X spec; usable RAM ~85% of total (OS overhead).
    # Snapdragon X Elite X1E-84-100: 45/64 GB LPDDR5X, 136 GB/s
    # Snapdragon X Plus X1P-64-100:  32/64 GB LPDDR5X, 120 GB/s
    {"name": "Snapdragon X Elite (64 GB)","vram_gb": 64, "bandwidth_gbps": 135,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Elite (32 GB)","vram_gb": 32, "bandwidth_gbps": 135,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Plus (64 GB)", "vram_gb": 64, "bandwidth_gbps": 135,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    {"name": "Snapdragon X Plus (32 GB)", "vram_gb": 32, "bandwidth_gbps": 135,  "hw_type": "qualcomm","category": "Qualcomm Snapdragon X"},
    # ── CPU Only ─────────────────────────────────────────────────────────────
    {"name": "CPU only — DDR5 laptop",     "vram_gb": 16,  "bandwidth_gbps": 89.6,  "hw_type": "cpu",    "category": "CPU Only"},
    {"name": "CPU only — DDR5 desktop",    "vram_gb": 32,  "bandwidth_gbps": 96,  "hw_type": "cpu",    "category": "CPU Only"},
    {"name": "CPU only — DDR5 workstation","vram_gb": 128, "bandwidth_gbps": 153.6, "hw_type": "cpu",    "category": "CPU Only"},
]

# Index by name for fast lookup
GPU_BY_NAME: dict[str, dict] = {g["name"]: g for g in GPUS}


# ── Peak compute, per SILICON ────────────────────────────────────────────────
# A preset is a (silicon, memory tier) pair: `vram_gb` and `bandwidth_gbps`
# belong to the tier, FLOPS belong to the die. Fourteen Apple M3/M4/M5 Max
# presets are six dies; the 5060 Ti 16GB/8GB pair is one. Keeping FLOPS here
# rather than on every preset line means the RTX 4090's figure is written once,
# so it can be corrected once.
#
# fp16_tflops is DENSE FP16/BF16 tensor throughput with FP32 ACCUMULATE — the
# rate cuBLASLt HGEMM, and therefore vLLM/llama.cpp prefill, actually hits. On
# every NVIDIA part since Turing the FP32-accumulate rate is HALF the
# FP16-accumulate rate and NVIDIA publishes both; using the wrong row overstates
# by exactly 2x. int8_tops is dense INT8 tensor TOPS.
#
# EVERY vendor headline in this range is a SPARSE number and often a different
# dtype. Stripped here:
#   RTX 50 "AI TOPS"  = FP4 with sparsity  = 16x the real dense FP16 (the
#                       5090's 3352 against its actual 209.5)
#   RTX 40 "AI TOPS"  = INT8 with sparsity = 16x FP32
#   Workstation Ada "Tensor performance" = FP8 with sparsity — halve it
#   L40S/L40/A40/A10 datasheets print "dense | sparse*" — the UNSTARRED number
#                       is ALREADY dense; do NOT halve 362.05 or 149.7
#   Turing (2080 Ti, T4) and Volta (V100) have NO structured sparsity at all
#
# APPLE IS DERIVED — Apple has published no GPU FLOPS figure since M2.
#   M1-M4:  cores x 512 FP16 FLOPS/clk (128 FP32 ALUs per core, FP16 at double
#           rate; Apple WWDC 2020). M1/M2 use Apple's own published FP32 x2.
#           This is a scalar-ALU ROOFLINE, not achievable matmul:
#           arXiv:2606.12765 states the M4 Max fp16 roofline as "~32 TFLOP/s"
#           (matching exactly) but measures real tiled matmul peaking at 14.8.
#   M5/M6:  cores x 1024, because M5 adds a Neural Accelerator to every GPU
#           core. The 1024 is measured, not assumed (tzakharko: 7.4 TFLOPS on
#           the 5-core A19 at ~1.46 GHz -> 1014 ~= 1024). M5's 1.578 GHz is a
#           measured sustained clock. M6's 1.709 GHz is INFERRED by inverting
#           two Apple ratios that agree to 1% ("nearly 30 percent increase in
#           peak GPU compute for AI compared to M5" -> 1.30 x 16.16 = 21.0;
#           "more than 8x compared to M1" -> 21.0 / 2.6 = 8.08x). The 21.0 is
#           solid; never quote 1.709 GHz anywhere as a spec.
#   iPhone: GPU, NOT the Neural Engine. The ANE is ~8x faster and unreachable
#           from llama.cpp Metal. If anyone "corrects" these upward with
#           Apple's marketing TOPS, iPhone figures overstate by ~8x.
#
# `None` means no figure could be sourced. It is not a zero and not a licence to
# interpolate: decode_roofline() drops the compute roof and reports bound
# "memory?" so the UI can say the ceiling is unknown for that device.
SILICON: dict[str, dict] = {
    # -- NVIDIA GeForce RTX 50 (Blackwell) — whitepaper v1.1 App. A Tbl 3 ----------
    "NVIDIA RTX 5090":                     {"fp16_tflops":   209.5, "int8_tops":     838},
    "NVIDIA RTX 5080":                     {"fp16_tflops":   112.6, "int8_tops":   450.2},
    "NVIDIA RTX 5070 Ti":                  {"fp16_tflops":    87.9, "int8_tops":   351.5},
    "NVIDIA RTX 5070":                     {"fp16_tflops":    61.7, "int8_tops":   246.9},
    "GB206":                               {"fp16_tflops":    47.4, "int8_tops":   189.8},
    "NVIDIA RTX 5060 Ti":                  {"fp16_tflops":    47.4, "int8_tops":   189.8},
    "NVIDIA RTX 5060":                     {"fp16_tflops":    38.4, "int8_tops":   153.5},
    # -- NVIDIA GeForce RTX 40 (Ada) — whitepaper v2.02 App. A Tbl 2 ---------------
    "NVIDIA RTX 4090":                     {"fp16_tflops":   165.2, "int8_tops":   660.6},
    "NVIDIA RTX 4080 Super":               {"fp16_tflops":   104.5, "int8_tops":     418},
    "NVIDIA RTX 4080":                     {"fp16_tflops":    97.5, "int8_tops":   389.9},
    "NVIDIA RTX 4070 Ti Super":            {"fp16_tflops":    88.3, "int8_tops":     353},
    "NVIDIA RTX 4070 Ti":                  {"fp16_tflops":    80.2, "int8_tops":   320.7},
    "NVIDIA RTX 4070 Super":               {"fp16_tflops":      71, "int8_tops":     284},
    "NVIDIA RTX 4070":                     {"fp16_tflops":    58.3, "int8_tops":   233.2},
    "AD106":                               {"fp16_tflops":    44.1, "int8_tops":   176.5},
    "NVIDIA RTX 4060 Ti":                  {"fp16_tflops":    44.1, "int8_tops":   176.5},
    "NVIDIA RTX 4060":                     {"fp16_tflops":    30.3, "int8_tops":     121},
    # -- NVIDIA GeForce RTX 30 (Ampere) — GA102 whitepaper v2 App. A ---------------
    "NVIDIA RTX 3090 Ti":                  {"fp16_tflops":      80, "int8_tops":     320},
    "NVIDIA RTX 3090":                     {"fp16_tflops":    71.2, "int8_tops":   284.7},
    "NVIDIA RTX 3080 Ti":                  {"fp16_tflops":    68.2, "int8_tops":   272.8},
    "GA102-3080":                          {"fp16_tflops":    61.3, "int8_tops":   245.1},
    "NVIDIA RTX 3080":                     {"fp16_tflops":    59.5, "int8_tops":   238.1},
    "NVIDIA RTX 3070 Ti":                  {"fp16_tflops":    43.5, "int8_tops":     174},
    "NVIDIA RTX 3070":                     {"fp16_tflops":    40.6, "int8_tops":   162.6},
    # -- NVIDIA GeForce RTX 20 (Turing) — no structured sparsity at all ------------
    "NVIDIA RTX 2080 Ti":                  {"fp16_tflops":    56.9, "int8_tops":   227.7},
    # -- NVIDIA data center — Blackwell --------------------------------------------
    "B200":                                {"fp16_tflops":    2500, "int8_tops":    5000},
    "NVIDIA B300 SXM":                     {"fp16_tflops":    2500, "int8_tops":    None},
    "NVIDIA B200 SXM":                     {"fp16_tflops":    2500, "int8_tops":    5000},
    # -- NVIDIA data center — Hopper -----------------------------------------------
    "NVIDIA GH200":                        {"fp16_tflops":     990, "int8_tops":    1979},
    "NVIDIA H200 SXM":                     {"fp16_tflops":   989.5, "int8_tops":    1979},
    "NVIDIA H200 PCIe":                    {"fp16_tflops":   835.5, "int8_tops":  1670.5},
    "NVIDIA H100 NVL":                     {"fp16_tflops":   835.5, "int8_tops":    1671},
    "NVIDIA H100 SXM":                     {"fp16_tflops":   989.5, "int8_tops":    1979},
    "NVIDIA H100 PCIe":                    {"fp16_tflops":   756.5, "int8_tops":    1513},
    # -- NVIDIA data center — Ada / Ampere -----------------------------------------
    "NVIDIA L40S":                         {"fp16_tflops":  362.05, "int8_tops":     733},
    "NVIDIA L40":                          {"fp16_tflops":  181.05, "int8_tops":     362},
    "NVIDIA L4":                           {"fp16_tflops":     121, "int8_tops":   242.5},
    "NVIDIA A40":                          {"fp16_tflops":   149.7, "int8_tops":   299.3},
    "GA100":                               {"fp16_tflops":     312, "int8_tops":     624},
    "NVIDIA A10":                          {"fp16_tflops":     125, "int8_tops":     250},
    "NVIDIA V100":                         {"fp16_tflops":     125, "int8_tops":    None},
    "NVIDIA T4":                           {"fp16_tflops":      65, "int8_tops":     130},
    # -- AMD Instinct (CDNA) -------------------------------------------------------
    "AMD Instinct MI355X":                 {"fp16_tflops":    2500, "int8_tops":    5000},
    "AMD Instinct MI325X":                 {"fp16_tflops":    1300, "int8_tops":    2600},
    "AMD Instinct MI300X":                 {"fp16_tflops":    1300, "int8_tops":    2600},
    "AMD Instinct MI250X":                 {"fp16_tflops":     383, "int8_tops":     383},
    # -- Intel Gaudi ---------------------------------------------------------------
    "Intel Gaudi 3":                       {"fp16_tflops":    1678, "int8_tops":    None},
    "Intel Gaudi 2":                       {"fp16_tflops":     432, "int8_tops":    None},
    # -- NVIDIA professional workstation -------------------------------------------
    "NVIDIA RTX 6000 Ada":                 {"fp16_tflops":   364.2, "int8_tops":   728.5},
    "NVIDIA RTX 5000 Ada":                 {"fp16_tflops":   261.2, "int8_tops":   522.2},
    "NVIDIA RTX A6000":                    {"fp16_tflops":   154.8, "int8_tops":   309.7},
    # -- Apple M1 — cores x 512 FP16 FLOPS/clk (128 FP32 ALUs, FP16 2x) ------------
    "Apple M1":                            {"fp16_tflops":     5.2, "int8_tops":    None},
    "Apple M1 Pro":                        {"fp16_tflops":    10.4, "int8_tops":    None},
    "Apple M1 Max":                        {"fp16_tflops":    20.8, "int8_tops":    None},
    "Apple M1 Ultra":                      {"fp16_tflops":      42, "int8_tops":    None},
    # -- Apple M2 ------------------------------------------------------------------
    "Apple M2":                            {"fp16_tflops":     7.2, "int8_tops":    None},
    "Apple M2 Pro":                        {"fp16_tflops":    13.6, "int8_tops":    None},
    "Apple M2 Max":                        {"fp16_tflops":    27.2, "int8_tops":    None},
    "Apple M2 Ultra":                      {"fp16_tflops":    54.4, "int8_tops":    None},
    # -- Apple M3 ------------------------------------------------------------------
    "Apple M3":                            {"fp16_tflops":     7.1, "int8_tops":    None},
    "Apple M3 Pro":                        {"fp16_tflops":    12.7, "int8_tops":    None},
    "M3 Max 30c":                          {"fp16_tflops":    21.2, "int8_tops":    None},
    "Apple M3 Max":                        {"fp16_tflops":    28.3, "int8_tops":    None},
    "Apple M3 Ultra":                      {"fp16_tflops":    56.5, "int8_tops":    None},
    # -- Apple M4 ------------------------------------------------------------------
    "Apple M4":                            {"fp16_tflops":     8.1, "int8_tops":    None},
    "Apple M4 Pro":                        {"fp16_tflops":    16.2, "int8_tops":    None},
    "M4 Max 32c":                          {"fp16_tflops":    25.9, "int8_tops":    None},
    "Apple M4 Max":                        {"fp16_tflops":    32.3, "int8_tops":    None},
    # -- Apple M5 — cores x 1024 (Neural Accelerator per core), 1.578 GHz ----------
    "Apple M5":                            {"fp16_tflops":    16.2, "int8_tops":    None},
    "Apple M5 Pro":                        {"fp16_tflops":    32.3, "int8_tops":    None},
    "M5 Max 32c":                          {"fp16_tflops":    51.7, "int8_tops":    None},
    "Apple M5 Max":                        {"fp16_tflops":    64.6, "int8_tops":    None},
    "M5 Ultra 80c":                        {"fp16_tflops":   129.3, "int8_tops":    None},
    # -- Apple M6 — 12 cores x 1024 x 1.709 GHz (clock INFERRED, see below) --------
    "Apple M6":                            {"fp16_tflops":      21, "int8_tops":    None},
    # -- Apple A-series — GPU, NOT the Neural Engine -------------------------------
    "A16 (iPhone 14 Pro / 16e)":           {"fp16_tflops":    1.79, "int8_tops":    None},
    "A17 Pro (iPhone 15 Pro)":             {"fp16_tflops":    2.06, "int8_tops":    None},
    "A18 Pro (iPhone 16 Pro)":             {"fp16_tflops":    2.26, "int8_tops":    None},
    "A19 Pro (iPhone 17 Pro)":             {"fp16_tflops":    9.83, "int8_tops":    None},
    # -- AMD Radeon RDNA 4 — AMD publishes the DENSE figure first ------------------
    "AMD RX 9070 XT":                      {"fp16_tflops":     195, "int8_tops":     389},
    "AMD RX 9070":                         {"fp16_tflops":     145, "int8_tops":     289},
    "AMD RX 9060 XT":                      {"fp16_tflops":     103, "int8_tops":     205},
    # -- AMD Radeon RDNA 3 — WMMA; AMD lists INT8 equal to FP16 here ---------------
    "AMD RX 7900 XTX":                     {"fp16_tflops":     123, "int8_tops":     123},
    "AMD RX 7900 XT":                      {"fp16_tflops":     103, "int8_tops":     103},
    "AMD RX 7900 GRE":                     {"fp16_tflops":      92, "int8_tops":      92},
    "AMD RX 7800 XT":                      {"fp16_tflops":    74.6, "int8_tops":    74.6},
    "AMD RX 7700 XT":                      {"fp16_tflops":    70.3, "int8_tops":    70.3},
    "AMD RX 7600 XT":                      {"fp16_tflops":    45.1, "int8_tops":    45.1},
    "AMD RX 7600":                         {"fp16_tflops":    43.5, "int8_tops":    43.5},
    # -- AMD Radeon RDNA 2 — no matrix cores, so FP16 is packed-FP32 rate ----------
    "AMD RX 6900 XT":                      {"fp16_tflops":   46.08, "int8_tops":    None},
    "AMD RX 6800 XT":                      {"fp16_tflops":   41.47, "int8_tops":    None},
    "AMD RX 6800":                         {"fp16_tflops":   32.33, "int8_tops":    None},
    "AMD RX 6700 XT":                      {"fp16_tflops":   26.43, "int8_tops":    None},
    # -- Intel Arc (Xe / Xe2 XMX) --------------------------------------------------
    "Intel Arc B770 (16 GB)":              {"fp16_tflops":   186.8, "int8_tops":   373.6},
    "Intel Arc B580":                      {"fp16_tflops":   116.5, "int8_tops":     233},
    "Intel Arc A770":                      {"fp16_tflops":   137.6, "int8_tops":   275.1},
    "Intel Arc A750":                      {"fp16_tflops":   117.4, "int8_tops":   234.9},
    # -- Qualcomm Snapdragon X — Adreno GPU ----------------------------------------
    "Snapdragon X Elite":                  {"fp16_tflops":     9.2, "int8_tops":    None},
    "Snapdragon X Plus":                   {"fp16_tflops":     7.6, "int8_tops":    None},
    # -- x86 CPU — peak AVX-512/AMX; the LOWBIT roof is what actually binds --------
    "CPU only — DDR5 laptop":              {"fp16_tflops":    2.05, "int8_tops":     4.1},
    "CPU only — DDR5 desktop":             {"fp16_tflops":    9.01, "int8_tops":      18},
    "CPU only — DDR5 workstation":         {"fp16_tflops":    73.7, "int8_tops":   147.5},
}

# Presets whose memory tier does NOT pick the silicon carry an explicit "si".
# Everything else resolves by stripping the memory parenthetical, which is true
# for 133 of 140 rows.
_MEM_SUFFIX_RE = re.compile(r"\s*\((?:\d+(?:\.\d+)?\s*GB)(?:,[^)]*)?\)$|\s+\d+GB$")


def _silicon_key(gpu: dict) -> str:
    """Silicon a preset runs on.

    The explicit "si" overrides are real hardware facts, not naming noise: Apple
    sells the SAME "M5 Max" name as a 32-core / 460 GB/s bin at 36 GB and a
    40-core / 614 GB/s bin at 48 GB and up, so the memory tier IS the bin, and a
    name-only rule would hand the 36 GB machine 25% more FLOPS than it has.
    test_every_preset_resolves_to_a_silicon_row fails loudly on a preset that
    matches neither path.
    """
    if "si" in gpu:
        return gpu["si"]
    return _MEM_SUFFIX_RE.sub("", gpu["name"]).strip()


def gpu_compute(gpu: dict | None) -> tuple[float | None, float | None]:
    """(fp16_tflops, int8_tops) for a preset, or (None, None) if unsourced.

    int8_tops is CARRIED AND READ BY NOTHING today. It is here because the
    research is done and a prefill / time-to-first-token model needs it — not as
    dead weight to be silently repurposed as a decode figure.
    """
    if not gpu:
        return None, None
    si = SILICON.get(_silicon_key(gpu))
    if si is None:
        return None, None            # bandwidth-only; NEVER interpolate a value
    return si.get("fp16_tflops"), si.get("int8_tops")


def tflops_for_gpu(gpu_name: str) -> float | None:
    """fp16_tflops for a preset NAME, for the two render paths' hw-meta dicts."""
    return gpu_compute(GPU_BY_NAME.get(gpu_name))[0]


# ── Calculation helpers ───────────────────────────────────────────────────────

# ── Attention geometry ───────────────────────────────────────────────────────
# The KV cache is the term this file used to assert was free, and it is the
# single largest error the Run Local tab shipped: a Qwen-32B at Q2 and 128k
# context charted 67.9 tok/s against 9.3 measured, and Llama 3.1 8B at Q4 was
# told to fit a 12 GB card at its advertised 128k context when it needs ~21 GiB.
#
# Modelling it needs n_layers / n_kv_heads / head_dim, which aa_local_models.csv
# does not carry and the AA leaderboard payload does not publish. Two options
# were on the table and BOTH have a failure mode this codebase has paid for
# before: refusing to price context leaves the tab asserting the cache is free
# (an error up to +1171%), and a silent fitted estimate is the hand-set-constant
# pattern. So neither, exactly: the values below are read from each model's own
# published config.json — data/pending_models.py's doctrine is that a
# MEASUREMENT may never be invented but a published architectural FACT may be
# curated — the fitted estimator runs only for models not in this table, and
# every row carries a kv_source column saying which it got.
#
# Formulas are DeepSeek-V2 (arXiv:2405.04434) Table 1, "Comparison of the KV
# cache per token among different attention mechanisms".
#
# Matched exact-then-longest-prefix on a case-folded name, so "Qwen 2.5 7B
# Instruct" finds "qwen 2.5 7b".
KV_ARCH: dict[str, dict] = {
    # ── dense GQA ──
    "llama 3.2 1b":        {"n_layers": 16, "n_kv_heads":  8, "head_dim":  64},
    "llama 3.2 3b":        {"n_layers": 28, "n_kv_heads":  8, "head_dim": 128},
    "llama 3.1 8b":        {"n_layers": 32, "n_kv_heads":  8, "head_dim": 128},
    "llama 3.3 70b":       {"n_layers": 80, "n_kv_heads":  8, "head_dim": 128},
    "llama 3.1 70b":       {"n_layers": 80, "n_kv_heads":  8, "head_dim": 128},
    # 8 KV heads, as the Llama 3 paper's Table 3 says (arXiv:2407.21783).
    #
    # This row said 16 for one commit, on the strength of a k_proj shape read out
    # of SillyTilly/Meta-Llama-3.1-405B-Instruct. That mirror is a repack, not
    # the model, and the parameter counts settle it without needing to trust any
    # single header: meta-llama/Llama-3.1-405B publishes 405,853,388,800
    # parameters and so does NousResearch's mirror, while SillyTilly's publishes
    # 410,081,247,232. The difference is 4,227,858,432, which is EXACTLY
    # 126 layers x 2 tensors x 8 extra KV heads x 128 head_dim x 16384 hidden.
    # The repack carries eight KV heads the released model does not have.
    #
    # Caught by test_scraped_architecture_agrees_with_the_hand_curated_table,
    # which exists for precisely this: two published sources disagreeing means
    # one of them is being read wrong. The cache is 504 KiB/token, not 1008.
    "llama 3.1 405b":      {"n_layers": 126, "n_kv_heads": 8, "head_dim": 128},
    "qwen 2.5 0.5b":       {"n_layers": 24, "n_kv_heads":  2, "head_dim":  64},
    "qwen 2.5 1.5b":       {"n_layers": 28, "n_kv_heads":  2, "head_dim": 128},
    "qwen 2.5 3b":         {"n_layers": 36, "n_kv_heads":  2, "head_dim": 128},
    "qwen 2.5 7b":         {"n_layers": 28, "n_kv_heads":  4, "head_dim": 128},
    "qwen 2.5 14b":        {"n_layers": 48, "n_kv_heads":  8, "head_dim": 128},
    "qwen 2.5 32b":        {"n_layers": 64, "n_kv_heads":  8, "head_dim": 128},
    "qwen 2.5 72b":        {"n_layers": 80, "n_kv_heads":  8, "head_dim": 128},
    "qwq 32b":             {"n_layers": 64, "n_kv_heads":  8, "head_dim": 128},
    "qwen3 4b":            {"n_layers": 36, "n_kv_heads":  8, "head_dim": 128},
    "qwen3 8b":            {"n_layers": 36, "n_kv_heads":  8, "head_dim": 128},
    "qwen3 14b":           {"n_layers": 40, "n_kv_heads":  8, "head_dim": 128},
    "qwen3 32b":           {"n_layers": 64, "n_kv_heads":  8, "head_dim": 128},
    "mistral 7b":          {"n_layers": 32, "n_kv_heads":  8, "head_dim": 128},
    "mistral nemo 12b":    {"n_layers": 40, "n_kv_heads":  8, "head_dim": 128},
    "mistral small 3":     {"n_layers": 40, "n_kv_heads":  8, "head_dim": 128},
    "phi-4 14b":           {"n_layers": 40, "n_kv_heads": 10, "head_dim": 128},
    "phi-3 medium 14b":    {"n_layers": 40, "n_kv_heads": 10, "head_dim": 128},
    # Gemma runs head_dim 256 at the small end — one of only two families that
    # break the head_dim=128 rule, and it doubles the cache.
    "gemma 2 9b":          {"n_layers": 42, "n_kv_heads":  8, "head_dim": 256},
    "gemma 2 27b":         {"n_layers": 46, "n_kv_heads": 16, "head_dim": 128},
    "gemma 3 4b":          {"n_layers": 34, "n_kv_heads":  4, "head_dim": 256},
    "gemma 3 12b":         {"n_layers": 48, "n_kv_heads":  8, "head_dim": 256},

    # ── hybrid local/global ──
    # These interleave sliding-window layers with full-attention ones, and the
    # window layers stop growing at `window` tokens. Ignoring that overstates
    # Gemma 3 27B at 128k by 6.0x and Llama 4 Scout at 1M by 3.9x.
    "gemma 3 27b":  {"n_layers": 62, "n_kv_heads": 16, "head_dim": 128,
                     "global_layers": 10, "window": 1024},   # 5 local : 1 global
    "gpt-oss-120b": {"n_layers": 36, "n_kv_heads":  8, "head_dim":  64,
                     "global_layers": 18, "window":  128},
    "gpt-oss-20b":  {"n_layers": 24, "n_kv_heads":  8, "head_dim":  64,
                     "global_layers": 12, "window":  128},

    # ── MoE (GQA attention) ──
    "mixtral 8x7b":     {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
    "mixtral 8x22b":    {"n_layers": 56, "n_kv_heads": 8, "head_dim": 128},
    "qwen3 30b-a3b":    {"n_layers": 48, "n_kv_heads": 4, "head_dim": 128},
    "qwen3 235b-a22b":  {"n_layers": 94, "n_kv_heads": 4, "head_dim": 128},
    "llama 4 scout":    {"n_layers": 48, "n_kv_heads": 8, "head_dim": 128},

    # ── MLA ── ONE latent cached, no factor of 2. Verified against the weights:
    # DeepSeek-V3's kv_a_proj_with_mqa.weight is [576, 7168] and
    # 576 = kv_lora_rank(512) + qk_rope_head_dim(64). DeepSeek V3 caches
    # 68.6 KB/token, barely half of Llama-3.1-8B's 128 KB despite being 671B.
    # Routing it through the GQA estimator is +232% wrong.
    "deepseek v3": {"attn": "mla", "n_layers": 61, "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64},
    "deepseek r1": {"attn": "mla", "n_layers": 61, "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64},
    "kimi k2":     {"attn": "mla", "n_layers": 61, "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64},
}

# ── The estimator: fallback only, for models not in KV_ARCH ──────────────────
# Fitted over the 33 published architectures above and their siblings.
# IN-SAMPLE accuracy on that table: mean |error| 29.6%, median 15.0%, p90 55.6%,
# max 171% (Qwen 2.5 7B, which uses 4 KV heads at 7.6B where its peers use 8).
#
# OUT OF SAMPLE IT IS MUCH WORSE, and now that data/arch_scraper.py resolves most
# of the catalogue we can measure it rather than warn about it: against the 147
# rows whose real geometry is known, the estimator's error is mean 89%, median
# 33%, p90 272%, max 885%. The tail is structural, not noise — the estimator
# assumes every layer caches the full sequence, so it cannot see sliding-window
# attention (Gemma 4 E4B: +618%; Muse Glimmer: +885%) or latent attention
# (Sarvam 105B: +367%). Those are the two things it is worst at and they are
# increasingly common.
#
# This is the FALLBACK now, not the main path. Roughly one row in five reaches
# it, and every one of those is labelled "architecture estimated" on screen.
#
# DO NOT SURFACE THE ESTIMATED n_layers / n_kv_heads AS FACTS. The fit works
# partly through compensating errors: on Qwen3 235B-A22B it gets layers 46% too
# low and KV width 100% too high and the product lands within 8.5%. Show the KV
# bytes per token; never show the shape.
LAYERS_A, LAYERS_B = 21.6, 0.281        # n_layers = A * P_eff ** B
LAYERS_MIN, LAYERS_MAX = 16, 128        # observed range, Llama 3.2 1B .. 405B
EST_HEAD_DIM = 128                      # exactly 128 in 26 of 33 architectures
# n_kv_heads is a STEP function of size: the observed values are powers of two
# (8 in 21 of 33), never a smooth curve. A grid search preferring edges at
# 2/8/100 was rejected as fitting noise — it puts Llama 3.1 8B, Mistral 7B and
# Qwen3 8B, all truly 8, into the 4-head band.
KV_HEAD_BANDS = ((2.0, 2), (4.0, 4), (200.0, 8), (float("inf"), 16))


_KV_NAME_NOISE = re.compile(
    r"\((?:reasoning|non-reasoning|thinking|low|medium|high|xhigh|max|max effort|"
    r"high effort|max_effort)[^)]*\)|\b(?:instruct|chat|it|preview|base)\b",
    re.I,
)


def _kv_normalise(name: str) -> str:
    """Fold a catalogue row's display name toward a KV_ARCH key.

    The scrape names a model by product, not by architecture: the same weights
    appear as "gpt-oss-120b (high)" and "gpt-oss-120b (low)", and a finetune
    carries its base model's attention config unchanged ("Hermes 4 -
    Llama-3.1 70B" is Llama 3.1 70B's 80/8/128). Effort settings and
    instruct/chat suffixes never change n_layers, n_kv_heads or head_dim, so
    folding them is a fact, not a guess.
    """
    key = _KV_NAME_NOISE.sub(" ", str(name).lower())
    key = key.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", key).strip()


# KV_ARCH keys folded through the same normaliser as the catalogue names, so a
# key written "gpt-oss-120b" still matches a row that arrives as "gpt-oss-120b
# (high)". Built once at import; KV_ARCH stays the human-readable source.
_KV_ARCH_NORM: dict[str, dict] = {}


def _kv_arch_lookup(name: str) -> dict | None:
    """Published geometry for a catalogue row: exact, then prefix, then contains.

    The `contains` pass is what catches finetunes, which are most of what the
    leaderboard carries above 100B. It is deliberately last and takes the
    LONGEST match, so "llama 3.1 405b" wins over "llama 3.1 8b" inside
    "Hermes 4 - Llama-3.1 405B" and a short key can never shadow a long one.
    A row that matches nothing falls to the estimator and is labelled.
    """
    key = _kv_normalise(name)
    if not key:
        return None
    hit = _KV_ARCH_NORM.get(key)
    if hit is not None:
        return hit
    for pick in (lambda k: key.startswith(k), lambda k: k in key):
        cands = [k for k in _KV_ARCH_NORM if pick(k)]
        if cands:
            return _KV_ARCH_NORM[max(cands, key=len)]
    return None


_KV_ARCH_NORM.update({_kv_normalise(k): v for k, v in KV_ARCH.items()})


def estimate_attention_shape(params_b: float, active_b: float | None = None,
                             is_moe: bool = False) -> tuple[int, int, int]:
    """Estimated (n_layers, n_kv_heads, head_dim). ESTIMATE, NOT MEASUREMENT.

    P_eff uses ACTIVE params for MoE. The KV cache is a function of the
    attention config alone — expert count and routing top-k do not enter it —
    but MoE models are shallow relative to their total size (DeepSeek V3 is 671B
    across 61 layers; Llama 3.1 405B is 405B across 126), so feeding total
    params to the depth law over-predicts KV by +353% on average and +814% at
    worst (gpt-oss-120b: 658 KB/token predicted against 72 real). Active params
    gives +31% mean. Measured on 7 GQA MoE models — a small sample, and a deep
    MoE would break it, but the direction is not in doubt.
    """
    p_eff = active_b if (is_moe and active_b) else params_b
    p_eff = max(float(p_eff or 0.0), 0.1)
    n_layers = int(min(max(round(LAYERS_A * p_eff ** LAYERS_B), LAYERS_MIN),
                       LAYERS_MAX))
    n_kv_heads = next(h for edge, h in KV_HEAD_BANDS if p_eff < edge)
    return n_layers, n_kv_heads, EST_HEAD_DIM


def kv_cache_bytes(arch: dict, seq_len: int, kv_quant: str = DEFAULT_KV_QUANT,
                   batch: int = 1) -> float:
    """Resident KV cache in bytes for `batch` sequences of `seq_len` tokens.

    Three shapes, because three real architectures need three formulas:
      MHA / GQA / MQA : 2 * n_layers * n_kv_heads * head_dim   (K and V)
      MLA             : n_layers * (kv_lora_rank + qk_rope)    (ONE latent)
      hybrid          : global layers pay seq_len, window layers stop at `window`
    """
    b = KV_BYTES_PER_ELEM[kv_quant]
    seq_len = max(int(seq_len), 0)
    if arch.get("attn") == "mla":
        return (arch["n_layers"] * (arch["kv_lora_rank"] + arch["qk_rope_head_dim"])
                * seq_len * batch * b)
    width = 2 * arch["n_kv_heads"] * arch["head_dim"] * b
    g = arch.get("global_layers")
    if g is not None:
        local = arch["n_layers"] - g
        # What the non-full layers are decides what they cost. A sliding-window
        # layer stops growing at `window`; a LINEAR / recurrent layer (Gated
        # DeltaNet, Mamba) holds a fixed-size state and contributes nothing that
        # grows with context at all. Qwen3.8-Flash-Next is 12 full-attention
        # layers out of 48 — charging the other 36 as full overstates it 4x.
        if arch.get("local_kind") == "linear":
            per_local = 0
        else:
            per_local = min(seq_len, arch.get("window") or seq_len)
        return width * batch * (g * seq_len + local * per_local)
    return width * arch["n_layers"] * seq_len * batch


# ── Scraped attention geometry ───────────────────────────────────────────────
# data/arch_scraper.py resolves each catalogue row to its HuggingFace repo and
# reads n_layers / n_kv_heads / head_dim straight out of the model's own
# config.json, refusing any repo whose published parameter count disagrees with
# the catalogue's. That is the same class of fact as KV_ARCH above — a published
# architectural fact, which data/pending_models.py's doctrine permits curating —
# so it is priced identically and labelled identically. KV_ARCH stays ahead of
# it only because those rows were read and cross-checked by hand.
_ARCH_CACHE = Path(__file__).parent / "raw" / "aa_local_arch.csv"
_SCRAPED_ARCH: dict[str, dict] | None = None


def _load_scraped_arch() -> dict[str, dict]:
    """Name -> geometry from the scrape, memoised. Missing file is not an error:
    the estimator answers for everything absent, and says so."""
    global _SCRAPED_ARCH
    if _SCRAPED_ARCH is not None:
        return _SCRAPED_ARCH
    out: dict[str, dict] = {}
    if _ARCH_CACHE.exists():
        try:
            df = pd.read_csv(_ARCH_CACHE)
        except Exception:
            df = None
        for _, r in (df.iterrows() if df is not None else ()):
            def _i(k):
                v = r.get(k)
                try:
                    return int(v) if pd.notna(v) and str(v) != "" else None
                except (TypeError, ValueError):
                    return None
            n_layers = _i("n_layers")
            if not n_layers:
                continue
            if str(r.get("attn", "")).lower() == "mla":
                latent = _i("kv_lora_rank")
                if not latent:
                    continue
                geo = {"attn": "mla", "n_layers": n_layers,
                       "kv_lora_rank": latent,
                       "qk_rope_head_dim": _i("qk_rope_head_dim") or 0}
            else:
                n_kv, head_dim = _i("n_kv_heads"), _i("head_dim")
                if not (n_kv and head_dim):
                    continue
                geo = {"n_layers": n_layers, "n_kv_heads": n_kv,
                       "head_dim": head_dim}
                g, w = _i("global_layers"), _i("sliding_window")
                kind = str(r.get("local_kind") or "").strip().lower()
                # A linear-attention hybrid needs NO window to be modelled —
                # its non-full layers cache nothing that grows. Requiring one
                # (as this did) silently charged every such model full price on
                # every layer.
                if g and 0 < g < n_layers and (w or kind == "linear"):
                    geo["global_layers"] = g
                    geo["local_kind"] = kind or "sliding"
                    if w:
                        geo["window"] = w
            out[str(r["name"])] = geo
    _SCRAPED_ARCH = out
    return out


def resolve_attention(name: str, params_b: float, active_b: float | None = None,
                      is_moe: bool = False) -> tuple[dict, str]:
    """(geometry, source) where source is "config", "hf" or "estimated".

    Order is hand-curated, then scraped, then fitted. The first two are both
    published facts and the UI presents them the same way; the estimator is the
    only one the reader is warned about, because it is the only one that is a
    guess.
    """
    hit = _kv_arch_lookup(name)
    if hit is not None:
        return hit, "config"
    scraped = _load_scraped_arch().get(str(name))
    if scraped is not None:
        return scraped, "hf"
    L, H, D = estimate_attention_shape(params_b, active_b, is_moe)
    return {"n_layers": L, "n_kv_heads": H, "head_dim": D}, "estimated"


def kv_bytes_per_token(name: str, params_b: float, active_b: float | None = None,
                       is_moe: bool = False, ctx_tokens: int = 1,
                       kv_quant: str = DEFAULT_KV_QUANT) -> tuple[float, str]:
    """Average KV bytes per token at this context, and where the shape came from.

    Averaged rather than marginal so that hybrid models — whose window layers
    stop growing — are priced correctly when this is multiplied back by
    ctx_tokens. For a pure GQA/MLA model the two are identical.
    """
    arch, src = resolve_attention(name, params_b, active_b, is_moe)
    n = max(int(ctx_tokens), 1)
    return kv_cache_bytes(arch, n, kv_quant) / n, src


# ── VRAM ─────────────────────────────────────────────────────────────────────
class VramBreakdown(NamedTuple):
    total_gib: float
    weights_gib: float
    kv_gib: float
    overhead_gib: float
    kv_bytes_tok: float
    kv_source: str
    ctx_used: int


def vram_breakdown(params_b: float, quant: str, ctx_tokens: int = 0, *,
                   name: str = "", active_b: float | None = None,
                   is_moe: bool = False, max_context_tokens: int | None = None,
                   gpu_count: int = 1,
                   kv_quant: str = DEFAULT_KV_QUANT) -> VramBreakdown:
    """Weights + KV cache + runtime overhead, in GiB, itemised.

    UNITS ARE GiB (2**30 bytes), which is what a vendor means by "24 GB" on the
    box and therefore what GPUS[...]["vram_gb"] means. The old body computed
    params_b * bytes = DECIMAL GB (10**9) and compared it against a GiB card — a
    7.4% error, in the safe direction, but the KV term has to compose with it so
    both are GiB now. decode_roofline() stays DECIMAL GB, because bandwidth is
    quoted in decimal GB/s. Do not unify them.

    KV is sized at min(ctx_tokens, the model's own context_k) — a 4k model
    cannot run 32k, and pricing it as though it could made every small-context
    model look artificially expensive at the long settings — and at
    DEFAULT_KV_QUANT (FP16), which is what every runtime uses unless the reader
    passes `-ctk`. `-ctk q8_0` or vLLM's FP8 KV halve the term and make this
    pessimistic at long context by up to 2x.
    """
    weights = params_b * QUANT_BYTES[quant] * 1e9 / GIB
    ctx = max(int(ctx_tokens or 0), 0)
    if max_context_tokens:
        ctx = min(ctx, int(max_context_tokens))
    kv_tok, src = (0.0, "none")
    kv_gib = 0.0
    if ctx > 0:
        kv_tok, src = kv_bytes_per_token(name, params_b, active_b, is_moe, ctx,
                                         kv_quant)
        kv_gib = kv_tok * ctx / GIB
    overhead = max(int(gpu_count), 1) * CUDA_CTX_GIB + WORKSPACE_GIB
    return VramBreakdown(weights + kv_gib + overhead, weights, kv_gib, overhead,
                         kv_tok, src, ctx)


def calc_vram_gb(params_b: float, quant: str, ctx_tokens: int = 0, **kw) -> float:
    """Total VRAM in GiB: weights + KV cache + runtime overhead.

    WHAT CHANGED. This used to be `params_b * QUANT_BYTES[quant] * 1.18` with no
    context parameter at all, verified identical to the cent across all 14
    distinct context_k values in the catalogue. The old comment on _OVERHEAD
    said so plainly and said the honest move was to stop implying otherwise.
    Doing better did not need a scrape: it needed a curated table of published
    config.json values (KV_ARCH), a fitted fallback, and a kv_source column so
    the reader can tell which one they are looking at.

    The size of what was missing: Llama 3.1 8B at Q4 with its advertised 128k
    context needs ~21 GiB, not 4.74. The tab told a 12 GB RTX 4070 owner that
    model fit at its stated context. It fits at about 20k tokens.

    ctx_tokens defaults to 0, so the two-argument form still answers the
    weights-plus-overhead question it used to answer.
    """
    return vram_breakdown(params_b, quant, ctx_tokens, **kw).total_gib


# ── Speed: a roofline, not a division ────────────────────────────────────────
class DecodeEstimate(NamedTuple):
    tps: float
    bound: str            # "memory" | "compute" | "dequant" | "memory?"
    t_weights: float
    t_kv: float
    t_compute: float


def _dequant_seconds(active_b: float, quant: str, hw_type: str) -> float:
    """Seconds per step for a dequantize-then-multiply CPU runtime, or 0."""
    table = _LOWBIT_WEIGHT_RATE_GW_S.get(hw_type)
    if not table:
        return 0.0
    bpw = GGUF_BPW[quant]
    lo = max((b for b in table if b <= bpw), default=min(table))
    hi = min((b for b in table if b >= bpw), default=max(table))
    rate = table[lo] if lo == hi else (
        table[lo] + (table[hi] - table[lo]) * (bpw - lo) / (hi - lo))
    return active_b / rate if rate > 0 else 0.0


def decode_roofline(active_b: float, quant: str, bandwidth_gbps: float,
                    hw_type: str, *, ctx_tokens: int = 0,
                    kv_bytes_tok: float = 0.0, fp16_tflops: float | None = None,
                    batch: int = 1, compute_b: float | None = None) -> DecodeEstimate:
    """Decode throughput as the SLOWEST of three roofs.

        t_weights = W / (BW * u_hw * k_q)               # flat in the batch
        t_kv      = B * ctx * kv_bytes_tok / (BW * _KV_MBU)
        t_compute = B * 2 * active_params / (FLOPS * _MFU_DECODE)
        t_dequant = active_params / weight_rate(hw_type, quant)   # CPU only
        tok/s     = batch / max(t_weights + t_kv, t_compute, t_dequant)

    WHAT THIS REPLACES AND WHY. The old body was

        (bandwidth_gbps / (active_b * QUANT_BYTES[quant] * 1.18)) * _EFF[hw_type]

    which asserts three things that are not true. (1) tok/s scales as bytes**-1,
    so FP16 -> Q2 was an 8x speedup; measured across 15 same-model/same-hardware
    sweeps it is 1.9x-4.0x, and ~1.3-1.7x at 128k context. (2) The KV cache is
    free, so a Qwen-32B at Q2 and 128k context charted 67.9 tok/s against 9.3
    measured — 630% high. (3) The 1.18 activation allowance is bytes re-read
    from VRAM every token, which is not a thing; it is a VRAM SIZING factor and
    it has moved to vram_breakdown() where it belongs. Its accidental effect was
    to scale _EFF by 1/1.18, which is the only reason the FP16 column was within
    4% of measurement while Q2 was 375% high — the bug was invisible at the
    default quantisation and catastrophic at the extreme.

    UNITS are DECIMAL GB (1e9 bytes), because `bandwidth_gbps` is the vendor's
    decimal GB/s. calc_vram_gb works in GiB. Do not let the two meet.

    fp16_tflops=None means no compute ceiling could be sourced for this device.
    The roof is DROPPED and `bound` comes back "memory?" so the caller can say
    the ceiling is unknown rather than implying one was checked. Do not fill
    those in by scaling core counts.

    THE COMPUTE ROOF WILL ESSENTIALLY NEVER BIND AT batch=1, and that is the
    correct answer rather than a bug: batch-1 decode is a GEMV at 2 FLOP per
    weight, so its arithmetic intensity is 1.00 FLOP/byte at FP16 and 5.24 at
    Q2, against a machine balance of 24 (M4 base) to 295 (H100 SXM). It is
    carried because it becomes load-bearing at optimal_concurrency()'s batch
    sizes, and because a model that cannot say WHICH roof binds is not a
    roofline.

    MoE still reads active_b, so this is still an UPPER BOUND for MoE and the
    gap still widens as a model approaches the VRAM limit and experts page.
    The roofline does not fix that; the old note stands.
    """
    batch = max(int(batch), 1)
    weight_gb = active_b * QUANT_BYTES[quant]
    if weight_gb < 1e-9 or bandwidth_gbps <= 0:
        return DecodeEstimate(0.0, "memory", 0.0, 0.0, 0.0)

    u_hw = _MBU_FP16.get(hw_type, 0.66)
    k_q = _QUANT_STREAM_EFF[quant]
    t_weights = weight_gb / (bandwidth_gbps * u_hw * k_q)

    kv_gb = batch * max(int(ctx_tokens), 0) * max(kv_bytes_tok, 0.0) / 1e9
    t_kv = kv_gb / (bandwidth_gbps * _KV_MBU) if kv_gb else 0.0

    # compute_b, not active_b. Under batching an MoE reads the UNION of the
    # experts its sequences route to, but it still only MULTIPLIES by the ones
    # each token actually routes to: an expert whose weights were fetched for
    # another sequence contributes zero multiply-accumulates to this token. The
    # union is a bytes-moved quantity and belongs to the weight-stream roof
    # alone. Passing it here charged a 21B-A3.6B MoE exactly the decode FLOPs of
    # a DENSE 21B at batch >= 32, asserting that expert sparsity buys no FLOP
    # reduction under concurrency, and flipped real rows to bound="compute".
    flops_b = active_b if compute_b is None else compute_b
    t_compute = 0.0
    if fp16_tflops and fp16_tflops > 0:
        t_compute = (batch * 2.0 * flops_b * 1e9) / (fp16_tflops * 1e12 * _MFU_DECODE)
    t_dequant = _dequant_seconds(active_b, quant, hw_type)

    t_mem = t_weights + t_kv
    t_step = max(t_mem, t_compute, t_dequant)
    if t_step <= 0:
        return DecodeEstimate(0.0, "memory", t_weights, t_kv, t_compute)
    if t_step == t_dequant and t_dequant > t_mem:
        bound = "dequant"
    elif t_compute > t_mem:
        bound = "compute"
    else:
        bound = "memory" if fp16_tflops else "memory?"
    return DecodeEstimate(batch / t_step, bound, t_weights, t_kv, t_compute)


def calc_speed_tps(active_b: float, quant: str, bandwidth_gbps: float,
                   hw_type: str, **kw) -> float:
    """Single-stream decode tok/s. Thin float wrapper on decode_roofline().

    Signature-compatible with the four-positional-argument form on purpose: it
    is called from get_local_df, reached from components/stack_recommender.py
    through that frame, and pinned by six test modules.
    """
    return decode_roofline(active_b, quant, bandwidth_gbps, hw_type, **kw).tps


# ── Concurrency ──────────────────────────────────────────────────────────────
class ConcurrencyEstimate(NamedTuple):
    sessions: int
    total_tps: float
    per_session_tps: float
    single_tps: float
    binding: str          # "kv_vram" | "latency" | "throughput_turnover"


def _moe_read_fraction(active_b: float, params_b: float, batch: int) -> float:
    """Fraction of an MoE's weights read per step at batch size B.

    At B=1 only the routed experts are read, but different sequences route to
    different experts, so the union grows as 1 - (1 - f)**B with
    f = active_b / params_b. For Qwen3 30B-A3B (f=0.10) that is 10% of the
    weights at B=1 but 82% at B=16 — an MoE's decode advantage largely
    EVAPORATES under concurrency, which is why it shows ~1.4x batching gain
    against a dense 8B's 3.8x on the same card.

    APPROXIMATION: assumes balanced routing, so it is optimistic at small B and
    near-exact at large. No published measurement of effective expert-read
    fraction against batch size was found to check it against. Still strictly
    better than the old code's implied assumption that the advantage is
    unbounded.
    """
    if params_b <= 0 or active_b <= 0 or active_b >= params_b:
        return 1.0
    f = active_b / params_b
    return min(1.0, 1.0 - (1.0 - f) ** max(int(batch), 1))


def _step_seconds(*, params_b, active_b, quant, bandwidth_gbps, hw_type,
                  fp16_tflops, ctx_tokens, kv_bytes_tok, batch, moe):
    """Seconds for one decode step that emits `batch` tokens."""
    read_b = active_b
    if moe:
        read_b = params_b * _moe_read_fraction(active_b, params_b, batch)
    e = decode_roofline(read_b, quant, bandwidth_gbps, hw_type,
                        ctx_tokens=ctx_tokens, kv_bytes_tok=kv_bytes_tok,
                        fp16_tflops=fp16_tflops, batch=batch,
                        # read_b is the weight stream; the FLOPs stay on the
                        # routed experts. For a dense model these are equal.
                        compute_b=active_b)
    return (batch / e.tps) if e.tps > 0 else float("inf")


def optimal_concurrency(*, params_b: float, active_b: float, quant: str,
                        vram_gb: float, bandwidth_gbps: float, hw_type: str,
                        ctx_tokens: int, kv_bytes_tok: float,
                        weights_gib: float, fp16_tflops: float | None = None,
                        slo: str = DEFAULT_SLO, gpu_count: int = 1,
                        moe: bool = False) -> ConcurrencyEstimate | None:
    """Best number of concurrent sessions, and the aggregate tok/s it buys.

    WHY BATCHING WORKS AT ALL, and where the usual explanation stops short. One
    decode step reads the whole weight set ONCE and emits B tokens, so the
    weight term is flat in B and arithmetic intensity rises linearly with it.
    The KV read is NOT amortised — every sequence has its own cache and
    attention streams all of it every step:

        t_step(B) = W/(BW*u)  +  B*ctx*kv/(BW*u_kv)  +  2*P_active*B/(F*mfu)
                    flat         linear in B           linear in B

    Fitted to 86 measured inter-token-latency points across 11 published NVIDIA
    NIM concurrency sweeps (Llama-3.1-8B and Llama-3.3-70B; H100/H200/A100/L40S;
    fp8 and bf16; TP1-TP8): 1.4%-13% RMS per curve, 12.6% mean absolute error.
    Held out, and it does NOT flatter the model: Llama-3.1-8B at ctx 1500 on
    this file's H100 SXM preset (80 GB, 3350 GB/s) predicts 8,319 tok/s at
    B=256 against NIM's measured 11,527.6 at C=250 — 28% LOW. The +3.4% this
    docstring used to claim was real arithmetic against the wrong card: 4800
    GB/s reproduces 11,920, and 4800 is the H200. Under-prediction here is the
    expected direction and has a named cause — see the datacenter-MBU note on
    _MBU_FP16, which measures 0.31 on an H100 where this model assumes 0.85.

    OPTIMAL means the largest B such that
      (1) the KV cache for B sequences at ctx_tokens fits the leftover VRAM, and
      (2) per-session decode stays at or above the SLO floor,
    then clipped to argmax_B B/t_step(B), because aggregate throughput itself
    turns over. That clip is not theoretical: llama.cpp batched-bench on an
    RTX 3090 with Phi-4-mini Q4_K_M peaks at 3,973 tok/s at B=128 and FALLS to
    3,419 at B=256.

    Returns None when the model does not fit at all, or when kv_bytes_tok is
    unknown. Do NOT substitute a guess: a 2x error in KV bytes is a 2x error in
    optimal concurrency.

    MULTI-GPU. bandwidth comes from effective_bandwidth(), which returns ONE
    card's bandwidth because this project models a llama.cpp/Ollama LAYER split
    rather than tensor parallelism. Against NVIDIA's TP2 curve that
    under-predicts by ~30% — correctly, because it is a different deployment.
    The hover says so, or multi-GPU rows look wrong to anyone who has run vLLM.

    DECODE ONLY. Every measured curve shows time-to-first-token blowing up long
    before throughput does: Llama-3.1-8B at 20k input on an H100 goes from 11 s
    TTFT at C=50 to 279 s at C=250 while throughput is flat at ~1,335 tok/s. A
    reader with long prompts will meet queueing this says nothing about.
    """
    if kv_bytes_tok <= 0 or ctx_tokens <= 0 or bandwidth_gbps <= 0:
        return None
    floor = SLO_FLOORS_TPS.get(slo, SLO_FLOORS_TPS[DEFAULT_SLO])

    # (a) the KV-cache memory ceiling.
    #
    # The budget is the SAME one `fits` uses, and that is not a detail. This
    # used to be `vram_gb * GPU_MEMORY_UTILIZATION` minus the runtime overhead,
    # which charges for the driver context and the activation scratch TWICE —
    # vLLM's 0.92 is a reserve for exactly what CUDA_CTX_GIB + WORKSPACE_GIB
    # already models. The visible symptom was worse than the 8%: 3.4% of rows
    # the chart drew as runnable came back with sessions=0, so the same tooltip
    # read "Speed: 484 tok/s single stream" and "Sessions: x0 concurrent",
    # because `fits` and this function were answering out of different wallets.
    free_gib = vram_gb - weights_gib - (max(int(gpu_count), 1) * CUDA_CTX_GIB
                                        + WORKSPACE_GIB)
    kv_per_seq_gib = ctx_tokens * kv_bytes_tok / GIB
    if free_gib <= 0 or kv_per_seq_gib <= 0:
        return None
    b_mem = int(free_gib // kv_per_seq_gib)
    if b_mem < 1:
        return None
    b_mem = min(b_mem, MAX_REPORTED_CONCURRENCY)

    kw = dict(params_b=params_b, active_b=active_b, quant=quant,
              bandwidth_gbps=bandwidth_gbps, hw_type=hw_type,
              fp16_tflops=fp16_tflops, ctx_tokens=ctx_tokens,
              kv_bytes_tok=kv_bytes_tok, moe=moe)
    single = 1.0 / _step_seconds(batch=1, **kw)

    # (b) the latency ceiling and (c) the throughput turnover, in one scan.
    # Per-session speed is monotone decreasing in B, so the first SLO failure
    # ends it; aggregate throughput is not monotone, so it is an argmax.
    best_b, best_total, binding = 0, 0.0, "kv_vram"
    for b in range(1, b_mem + 1):
        t = _step_seconds(batch=b, **kw)
        if t == float("inf"):
            break
        per = 1.0 / t
        if per < floor:
            if best_b == 0:
                # Even one stream misses the floor. Report it honestly rather
                # than pretending the hardware clears an SLO it does not.
                return ConcurrencyEstimate(1, single, single, single, "latency")
            binding = "latency"
            break
        total = b / t
        # A relative tolerance, not `>=`. On a compute-bound plateau b/t_step(b)
        # is mathematically constant but not bit-constant, so a 1-ULP downward
        # blip took the else branch, stopped the scan early and labelled the
        # stop "throughput_turnover" when nothing had turned over. 0.1% is far
        # below any real turnover (the measured RTX 3090 curve falls 14% from
        # B=128 to B=256) and far above float noise.
        if total >= best_total * (1.0 - 1e-3):
            if total > best_total:
                best_total = total
            best_b = b
        else:
            binding = "throughput_turnover"
            break
    if best_b == 0:
        return None
    if best_b == b_mem:
        binding = "kv_vram"
    return ConcurrencyEstimate(best_b, best_total, best_total / best_b, single,
                               binding)


def total_throughput_tps(sessions: int, **kw) -> float:
    """Aggregate tok/s across `sessions` concurrent streams."""
    sessions = max(int(sessions), 1)
    t = _step_seconds(batch=sessions, **kw)
    return sessions / t if t not in (0, float("inf")) else 0.0


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
    # The shared constants, not a fourth set of numbers. These defaults used to
    # read 24 GB / 1008 GB/s, so a bare get_local_df() — the form the chart
    # contract tests call — validated the charts against hardware the product
    # never presents (47 fitting models at 72.3 tok/s median rather than 49 at
    # 128.5), and a regression at the real default was invisible to them.
    vram_gb: float = DEFAULT_VRAM_GB,
    bandwidth_gbps: float = DEFAULT_BANDWIDTH_GBPS,
    hw_type: str = "nvidia",
    tags: list[str] | None = None,
    include_pending: bool = True,
    ctx_tokens: int = DEFAULT_CONTEXT_TOKENS,
    fp16_tflops: float | None = None,
    slo: str = DEFAULT_SLO,
    gpu_count: int = DEFAULT_GPU_COUNT,
) -> pd.DataFrame:
    """
    Return a DataFrame of all models enriched with hardware-specific columns.

    Columns added:
        vram_req_gb     - total VRAM in GiB: weights + KV cache + runtime
        weights_gb      - the weights alone
        kv_gb           - the KV cache at ctx_used
        kv_bytes_tok    - KV bytes per token, the figure the hover should show
        kv_source       - "config" (published) | "estimated" (±30%) | "none"
        ctx_used        - context actually priced, capped at the model's own max
        speed_tps       - single-stream decode tok/s on the given hardware
        bound           - which roof binds: memory | compute | dequant | memory?
        sessions        - optimal concurrent sessions at the chosen SLO floor
        per_session_tps - decode tok/s each of those sessions still gets
        total_tps       - aggregate tok/s across all of them
        concurrency_bound - what stopped it: kv_vram | latency | turnover
        fits            - "yes" | "tight" (< 1 GB headroom) | "no"

    ctx_tokens defaults to DEFAULT_CONTEXT_TOKENS rather than 0 so that a bare
    get_local_df() — the form the chart-contract tests call — validates the
    charts against the hardware the product actually presents. The same reason
    vram_gb and bandwidth_gbps stopped defaulting to 24 GB / 1008 GB/s.
    """
    # Curated open-weight releases AA has not benchmarked yet are folded in
    # here, carrying quality=None. See data/pending_models.py for what an entry
    # may claim and why quality must never be invented.
    models = merge_pending(_load_models_raw()) if include_pending else \
        [{**m, "pending": False} for m in _load_models_raw()]
    rows = []
    for m in models:
        if tags:
            if not any(t in m["tags"] for t in tags):
                continue
        vb = vram_breakdown(
            m["params_b"], quant, ctx_tokens,
            name=m["name"], active_b=m["active_b"], is_moe=bool(m.get("moe")),
            max_context_tokens=int(m["context_k"]) * 1000,
            gpu_count=gpu_count,
        )
        est = decode_roofline(
            m["active_b"], quant, bandwidth_gbps, hw_type,
            ctx_tokens=vb.ctx_used, kv_bytes_tok=vb.kv_bytes_tok,
            fp16_tflops=fp16_tflops,
        )
        headroom = vram_gb - vb.total_gib
        if headroom >= 1.0:
            fits = "yes"
        elif headroom >= 0:
            fits = "tight"
        else:
            fits = "no"
        # Concurrency is only meaningful for a model that fits at all — a
        # session count for a model the reader cannot load is noise, and
        # optimal_concurrency returns None for it anyway.
        conc = None
        if fits != "no":
            conc = optimal_concurrency(
                params_b=m["params_b"], active_b=m["active_b"], quant=quant,
                vram_gb=vram_gb, bandwidth_gbps=bandwidth_gbps, hw_type=hw_type,
                ctx_tokens=vb.ctx_used, kv_bytes_tok=vb.kv_bytes_tok,
                weights_gib=vb.weights_gib, fp16_tflops=fp16_tflops, slo=slo,
                gpu_count=gpu_count, moe=bool(m.get("moe")),
            )
        rows.append({
            **m,
            "vram_req_gb":  round(vb.total_gib, 2),
            "weights_gb":   round(vb.weights_gib, 2),
            "kv_gb":        round(vb.kv_gib, 2),
            # Carried rather than hardcoded in the hovers: the driver context is
            # PER GPU, so this is 1.5 GiB at x1 and 5.0 at x8, and both hovers
            # printed a flat "runtime 1.5 GB" under a total that included 5.0.
            "overhead_gb":  round(vb.overhead_gib, 2),
            "kv_bytes_tok": round(vb.kv_bytes_tok, 1),
            "kv_source":    vb.kv_source,
            "ctx_used":     vb.ctx_used,
            "speed_tps":    round(est.tps, 1),
            "bound":        est.bound,
            "sessions":         conc.sessions if conc else 0,
            "per_session_tps":  round(conc.per_session_tps, 1) if conc else 0.0,
            "total_tps":        round(conc.total_tps, 1) if conc else 0.0,
            "concurrency_bound": conc.binding if conc else "",
            "fits":         fits,
            "tags_str":     ", ".join(m["tags"]) if m["tags"] else "general",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        # A tag that matches nothing gave a column-less frame, so df["family"]
        # raised KeyError('family') — and neither caller surfaced it: Dash kept
        # the previous figures and the browser logged to console. Not reachable
        # while the tag options are derived from the data, but 'code' is down to
        # 3 of 97 rows, so it is one scrape away.
        df = pd.DataFrame(columns=[
            "name", "family", "params_b", "active_b", "context_k", "quality",
            "license", "tags", "moe", "vram_req_gb", "speed_tps", "fits", "tags_str",
            "pending",
            # Every column the loop above emits has to appear here too, or a
            # filter that matches nothing raises KeyError in a chart builder
            # instead of rendering the empty state — the exact bug this list
            # was written to fix, one new column later.
            "weights_gb", "kv_gb", "overhead_gb", "kv_bytes_tok", "kv_source", "ctx_used",
            "bound", "sessions", "per_session_tps", "total_tps",
            "concurrency_bound",
        ])
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


def unhosted_ranking_rows(exclude_names=None) -> pd.DataFrame:
    """Open-weight models AA has scored that no API host sells, in the HOSTED
    catalogue's schema so they can join an intelligence ranking.

    WHY THIS EXISTS
    ---------------
    data/scraper.py builds the hosted catalogue from host x model rows, so a
    model nobody sells has no row and cannot appear on any tab driven by that
    frame. For the price and speed views that is correct — there is no price to
    plot and no host to measure. For a leaderboard captioned "top 25 by
    intelligence" it is not: intelligence is a property of the model, and
    Qwen3.8 27B scoring 52.0 is 24th in the catalogue whether or not anyone
    rents it out.

    Price, speed and latency come back as NaN rather than zero, deliberately.
    Every downstream metric filters on ``> 0``, and NaN fails that comparison,
    so these rows are admitted to the intelligence ranking and excluded from
    Value and Speed without a single call site needing to know they exist. Zero
    would have placed them at the origin of a value axis as infinitely good
    deals — the "no published price is not the same as free" defect that
    components/charts/image_scatter.py already documents.

    ``self_host`` marks them so the renderer can say what they are.
    """
    df = get_local_df(include_pending=False)
    if df.empty:
        return pd.DataFrame()
    exclude = {str(n) for n in (exclude_names or ())}
    rows = df[~df["name"].isin(exclude) & df["quality"].notna()]
    if rows.empty:
        return pd.DataFrame()

    def _ctx(k):
        try: k = int(k)
        except (TypeError, ValueError): return ""
        return f"{k / 1000:g}m" if k >= 1000 else f"{k}k"

    return pd.DataFrame({
        "model":     rows["name"].astype(str),
        "provider":  rows["family"].astype(str),
        "context":   rows["context_k"].map(_ctx),
        "quality":   pd.to_numeric(rows["quality"], errors="coerce"),
        "price":     float("nan"),
        "speed":     float("nan"),
        "latency":   float("nan"),
        "price_in":  float("nan"),
        "price_out": float("nan"),
        "self_host": True,
    }).reset_index(drop=True)
