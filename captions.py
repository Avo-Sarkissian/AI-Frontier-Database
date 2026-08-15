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
        "Find open-weight models you can run on your own hardware. Select your GPU "
        "(or enter VRAM manually), choose a quantization level, and see which "
        "models fit — with estimated inference speed."
    ),
    'image': (
        "Compare image generation models by quality and style. ELO scores from "
        "Artificial Analysis Image Arena — blind human comparisons. Each column "
        "shows the best models for that style. Annotations show generation time."
    ),
    'video': (
        "Compare video generation models on quality, speed, and cost. CURATED DATASET, "
        "NOT LIVE-SCRAPED: unlike every other tab, this list is maintained by hand and "
        "its models are 2025-era, so the \"Updated N ago\" badge in the header does not "
        "describe this tab. Quality scores are human preference ratings; price is per "
        "second of generated video; open-weights models can be self-hosted for free."
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
