"""Chart captions — ONE source for both renderings.

app.py rendered 14 of these; docs/app.js rendered exactly one, so the public
site — the one that actually deploys — shipped nine tabs with no explanation of
what the reader was looking at. Two copies of the same prose would have drifted
the way the palettes and the price semantics did, so the text lives here, app.py
reads it directly, and build_static.py ships it in the manifest for the browser.
"""

CAPTIONS: dict[str, str] = {
    'overview_price': (
"Each bubble is one model FAMILY — the best-scoring variant, so five Claude Opus 5 effort tiers "
        "collapse to one point rather than five near-duplicates at the same price. X = price "
        "per 1M tokens (log scale), Y = AA Intelligence Index. Bubble size = throughput (tok/s). "
        "Dotted line = Pareto frontier. Click any bubble for full details."
    ),
    'recommend': (
        "Build your Claude Code model stack. API Only = all cloud. Hybrid — Fast "
        "local = free local for sub-tasks, API for Balanced + Reasoning. Hybrid — "
        "Fast + Balanced local = local for both workhorse tiers, API only for the "
        "top orchestrator. Local Only = fully offline. Fast = high-volume "
        "sub-tasks. Balanced = coding and writing. Reasoning = planning and "
        "delegation."
    ),
    'landscape_treemap': (
        "AI ecosystem by provider. Tile area = number of models in the dataset. "
        "Color intensity = average intelligence score."
    ),
    'landscape_leaderboard': (
        "Provider leaderboard: bar length = best model's intelligence. Tick mark = "
        "provider average. Right labels show model count and top model."
    ),
    'rankings_intelligence': (
        "Intelligence = AA Intelligence Index (composite benchmark). Value = "
        "Intelligence ÷ Price (higher = more score per dollar). Speed = throughput "
        "in tokens/second. Models within ±2 points of each other are effectively "
        "tied — small deltas are within measurement variance."
    ),
    'rankings_value': (
        "Top 15 models ranked by intelligence per dollar (quality score ÷ price per "
        "1M tokens). Quality floor of 20 so ultra-cheap but weak models don't "
        "pollute the list. Bar length = value score. Labels show raw quality and "
        "price."
    ),
    'compare': (
        "Radar comparing up to 5 models across 5 dimensions, each scaled 0–100% "
        "between the weakest and strongest model in the full catalogue — so a shape "
        "means the same thing whatever the filter. Speed, price, context and "
        "latency are heavy-tailed, so those four use a log scale: one step is a 10× "
        "change, not a fixed amount. Affordability = inverted price (100% = "
        "cheapest). Latency = inverted TTFT (100% = fastest). Raw values for each "
        "model are shown in the table below the chart."
    ),
    'budget': (
        "Estimate monthly API cost. Price is our own blend of Artificial Analysis's "
        "per-token rates, weighted 3 parts output to 1 part input (AA's own "
        "published blend uses the opposite weighting, so their site quotes a lower "
        "figure). Enter volume in millions — 1M tokens ≈ 750,000 words or ~1,500 "
        "pages. Chart sorts cheapest-first."
    ),
    'table': (
        "Full sortable model table. Score = AA Intelligence Index (composite "
        "benchmark, higher = better). Value = Score ÷ Price (quality per dollar). "
        "Price = blended $/M tokens (3:1 output/input). Latency = "
        "time-to-first-token (TTFT) in seconds."
    ),
    'local': (
        "Model specs come from Artificial Analysis, plus a short curated list of "
        "notable open-weight releases AA has not benchmarked yet — those are drawn "
        "with an outlined bar and no score, never a guessed one, and are excluded "
        "from the Agent Stack recommendations. "
        "VRAM covers model WEIGHTS, the KV cache at the context you select, and "
        "about 1.5 GB of runtime overhead. The KV cache is the term that used to "
        "be missing and it is not small: Llama 3.1 8B at Q4 needs about 7 GB at 8k "
        "and about 22 GB at its advertised 128k — the KV cache alone goes from "
        "1 GB to 16 GB. Attention geometry comes from "
        "each model's published config where this dashboard carries it and is "
        "estimated from parameter count otherwise — the hover says which. About "
        "four in five models now carry a published figure; the rest are estimates, and an estimate here can be badly wrong for a "
        "model with sliding-window or latent attention, which the estimator cannot see. "
        "Scores are benchmarked at the model's native precision and do NOT move "
        "with the quantization control: lowering it here shrinks VRAM and raises "
        "speed at no visible cost, but real quality does fall, steeply at the "
        "levels marked lossy. Artificial Analysis does not publish quantized "
        "scores and this dashboard will not guess one. "
        "Speed is a roofline estimate (±30%) for token generation, not prompt "
        "processing: the slower of the memory stream (weights + KV cache) and the "
        "compute ceiling, and on every GPU here it is the memory stream, by 13x to "
        "198x. Quantizing does not buy its full byte ratio — FP16 to Q2 is about "
        "3.6x here and 1.9x–4.0x in published measurements, not the 8x a linear "
        "model implies — and the gain shrinks further at long context, because the "
        "KV cache stays FP16 whatever the weights are. It remains an upper bound "
        "for MoE models. The SPEED control picks which tok/s figure the chart shows: "
        "single stream is one conversation at a time; max throughput is the total "
        "across as many concurrent sessions as the KV cache fits in VRAM while each "
        "one still decodes at 10 tok/s or better (MLPerf's Llama-3.1-8B Server "
        "floor, ~480 words/minute). The two are 2-4x apart, so only one is on "
        "screen at a time and the hover carries the other. Both are decode-only "
        "and say nothing about queueing or time-to-first-token. "
        "Find open-weight models you can run on your own hardware. Select your GPU "
        "(or enter VRAM manually), choose a quantization level and a context "
        "length, and see which models fit — with estimated throughput."
    ),
    'image': (
        "Compare image generation models by quality and style. ELO scores from "
        "Artificial Analysis Image Arena — blind human comparisons. Each column "
        "shows the best models for that style. Annotations show generation time."
    ),
    'video': (
        "Compare video generation models on quality and cost, scraped hourly from the "
        "Artificial Analysis Video Arena. Elo comes from blind pairwise human votes, so "
        "it ranks models against each other and has no meaningful zero — read position, "
        "not distance from the axis. Text-to-video and image-to-video are separate "
        "arenas with separate prices, so MODE switches between them rather than blending "
        "them. Price is per minute of video at each provider's default settings; "
        "generation time is measured end-to-end and AA currently publishes it for only a "
        "handful of models. The ranked view shows current models; superseded preview "
        "builds stay in the ↓CSV export."
    ),
    'overview_speed': (
        "Speed (tok/s) vs. AA Intelligence Index. Top-right = fast and smart. "
        "Bubble size = affordability (larger = cheaper). Click any bubble for full "
        "details."
    ),
    'overview_price_dyn': (
"Each bubble is one model FAMILY — the best-scoring variant, so five Claude Opus 5 effort tiers "
        "collapse to one point rather than five near-duplicates at the same price. X = price "
        "per 1M tokens (log scale), Y = AA Intelligence Index. Bubble size = throughput (tok/s). "
        "Dotted line = Pareto frontier. Click any bubble for full details."
    ),
}
