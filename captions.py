"""Chart captions — ONE source for both renderings.

app.py rendered 14 of these; docs/app.js rendered exactly one, so the public
site — the one that actually deploys — shipped nine tabs with no explanation of
what the reader was looking at. Two copies of the same prose would have drifted
the way the palettes and the price semantics did, so the text lives here, app.py
reads it directly, and build_static.py ships it in the manifest for the browser.

Two layers, since the 2026-10-29 audit: the longest captions had grown to 3-4k
characters (about 1,000px of text above the Serve chart on a phone).
  CAPTIONS  one or two plain sentences: what the chart shows, how to read it.
  METHODS   everything else — method, sources, dates, caveats, attribution —
            as paragraphs inside a closed "Method & sources" disclosure.
Nothing was cut in the split, only moved; CAPTION_TEXT joins both layers so a
test can still demand that a fact, a source or a caveat is published.
"""

import re

CAPTIONS: dict[str, str] = {
    'overview_price': (
        "One bubble per model family: price per 1M tokens at your MIX (log scale) "
        "against the AA Intelligence Index, so up and to the left is better. "
        "Bubble size = speed, hollow = open weights, and the dotted line joins the "
        "families nothing beats on both price and score."
    ),
    'overview_speed': (
        "One bubble per model family: output speed (tok/s) against the AA "
        "Intelligence Index, so top-right is fast and smart. "
        "Larger bubbles are cheaper; hollow = open weights."
    ),
    'overview_task': (
        "One bubble per model family: what Artificial Analysis measured it costs to "
        "run its Intelligence Index (log scale) against the score, so up and to the "
        "left is better. MIX does not move this axis, and uncosted models are "
        "counted in the title, never estimated."
    ),
    'recommend': (
        "A suggested Claude Code model stack, one card per tier: Fast for "
        "high-volume sub-tasks, Balanced for coding and writing, Reasoning for "
        "planning and delegation. WORKFLOW picks which tiers run on your own GPU."
    ),
    'landscape_treemap': (
        "AI ecosystem by provider. Tile area = number of models in the dataset; "
        "colour intensity = average intelligence score."
    ),
    'landscape_leaderboard': (
        "Provider leaderboard: bar length = best model's intelligence, tick mark = "
        "provider average. Right labels show model count and top model."
    ),
    'rankings_intelligence': (
        "One bar per model family, ranked by the metric you pick: intelligence, "
        "value (score per dollar) or speed. Outlined bars are open weights; scores "
        "within ±2 points of each other are effectively tied."
    ),
    'rankings_value': (
        "The 15 model families with the most intelligence per dollar at your MIX. "
        "Bar length = value score; labels give raw quality and price."
    ),
    'compare': (
        "Up to five models on five dimensions, each scaled 0–100% across the full "
        "catalogue, so a shape means the same under any filter and bigger is "
        "better on every axis. Raw values are in the table below the chart."
    ),
    'budget': (
        "Estimated monthly API cost per model for the volume you enter, priced at "
        "the MIX in the filter bar and sorted cheapest first. "
        "1M tokens ≈ 750,000 words or ~1,500 pages; * = no cache discount published."
    ),
    'table': (
        "Every model in the current filter, sorted with the two controls above. "
        "Price is the effective $/1M tokens at your MIX, named in its column header."
    ),
    'local': (
        "Open-weight models against the VRAM they need on your GPU at the chosen "
        "quantization and context: everything left of the dashed line fits. "
        "The bars below rank the models that fit by intelligence, with estimated speed."
    ),
    'image': (
        "Image models by arena ELO in three styles, one dot per model on one shared "
        "scale, so further right is preferred more often. "
        "Hollow = open weights; the label is price per 1,000 images."
    ),
    'video': (
        "Video models ranked by arena Elo, then Elo against price per minute of "
        "video; MODE switches between the text-to-video and image-to-video arenas. "
        "Read position, not distance from the axis."
    ),
    'economics_serve': (
        "What it costs to serve each open-weights model on datacenter GPUs against "
        "its API output price, both per 1M output tokens: points below the dashed "
        "1× line are priced above their GPU cost. Diamonds are measured "
        "InferenceX runs; circles are modelled (read them as ±2×)."
    ),
    'economics_compute': (
        "What a GPU-hour costs to own (the stacked bar: capital, power and "
        "colocation) and what the market rents it for (the ladder and the weekly "
        "series). The GPU, preset and boxes override every default."
    ),
    'economics_labs': (
        "What the labs earn, keep, commit to and serve, one chart each, then each "
        "layer of the stack's dollars per year. Filled markers are disclosed, open "
        "markers reported, open diamonds estimated; a line never joins two definitions."
    ),
    'economics_open_gap': (
        "How far open-weight models trail closed ones, three ways: the score gap "
        "over time, the wait until an open model matched each new closed high, and "
        "the price of the cheapest closed model that scores as well."
    ),
}
# The Overview callback re-renders the price caption under its own key.
CAPTIONS['overview_price_dyn'] = CAPTIONS['overview_price']


METHODS: dict[str, list[str]] = {
    'overview_price': [
        "Each bubble is one model FAMILY — the best-scoring variant, so five Claude "
        "Opus 5 effort tiers collapse to one point rather than five near-duplicates "
        "at the same price.",
        "X = price per 1M tokens at the MIX setting (log scale; default = our 1:3 "
        "in:out blend, not AA's), Y = AA Intelligence Index. Bubble size = "
        "throughput (tok/s). Hollow = open weights, filled = closed (Artificial "
        "Analysis's flag; WEIGHTS filters them). Dotted line = Pareto frontier, "
        "recomputed for the mix. Click any bubble for full details.",
    ],
    'overview_speed': [
        "Speed (tok/s) vs. AA Intelligence Index, one bubble per model family "
        "(best-scoring variant). Top-right = fast and smart. Bubble size = "
        "affordability (larger = cheaper, log-scaled). Hollow = open weights. "
        "Click any bubble for full details.",
    ],
    'overview_task': [
        "Each bubble is one model FAMILY — its best-scoring variant that Artificial "
        "Analysis has costed (with EFFORT set, the nearest tier AA has costed).",
        "X = AA's cost per Intelligence Index task (USD, log scale): the token "
        "counts the provider reported for the run (a canonical tokenizer where none "
        "were reported), reasoning included, with cache reads at AA's live "
        "measurement of the model's typical cache-hit rate rather than the hits "
        "seen in the run. MIX does not move it. Models AA has not costed are "
        "counted in the title, never estimated.",
        "Y = AA Intelligence Index. Bubble size = throughput (tok/s). Dotted line = "
        "Pareto frontier on cost per task. Token burn in the hover and detail panel "
        "= cost per task ÷ list price per token. Click any bubble for full details.",
        "Source: artificialanalysis.ai.",
    ],
    'recommend': [
        "WORKFLOW: API Only = all cloud. Hybrid — Fast local = local for sub-tasks, "
        "API for Balanced + Reasoning. Hybrid — Fast + Balanced local = local for "
        "both workhorse tiers, API only for the top orchestrator. Local Only = fully "
        "offline.",
        "TIERS: Fast = high-volume sub-tasks. Balanced = coding and writing. "
        "Reasoning = planning and delegation.",
        "COST OF LOCAL PICKS: on a datacenter GPU each carries a $/M chip, "
        "SemiAnalysis InferenceX's renting $/GPU-hr for that card over the "
        "throughput this tab models, at full use. Consumer cards show no $/M — "
        "your own hardware has no published hourly rate.",
    ],
    'landscape_treemap': [],
    'landscape_leaderboard': [],
    'rankings_intelligence': [
        "One bar per model family — its best-scoring effort tier unless EFFORT "
        "picks another. Intelligence = AA Intelligence Index (composite benchmark). "
        "Value = Intelligence ÷ Price (higher = more score per dollar). Speed = "
        "throughput in tokens/second.",
        "Models within ±2 points of each other are effectively tied — small deltas "
        "are within measurement variance.",
        "About half these scores are estimated by Artificial Analysis rather than "
        "measured: since its 2026-09 rebuild it extrapolates the index for models it "
        "has not yet run the full evaluation suite on. Click any model for which it is.",
        "Outlined bars = open weights (Artificial Analysis's flag, or a name match "
        "to its open-weights list where a row has no flag; the hover says which), "
        "solid = closed.",
    ],
    'rankings_value': [
        "Value = quality score ÷ price per 1M tokens at the MIX setting, so an "
        "input-heavy or cached workload can re-rank the list; one bar per family.",
        "A quality floor excludes the weakest 5% of model families so ultra-cheap "
        "but weak models stay out; it is a percentile, not a fixed score, so it "
        "follows the index through a rescale.",
    ],
    'compare': [
        "Radar comparing up to 5 models across 5 dimensions, each scaled 0–100% "
        "between the weakest and strongest model in the full catalogue — so a shape "
        "means the same thing whatever the filter.",
        "Speed, price, context and latency are heavy-tailed, so those four use a log "
        "scale: one step is a 10× change, not a fixed amount. Affordability = "
        "inverted price (100% = cheapest). Latency = inverted TTFT (100% = "
        "fastest). Rings mark 25% steps; hover a shape for its exact percentages.",
    ],
    'budget': [
        "Price is the effective $/1M tokens for the workload MIX in the filter bar: "
        "input:output ratio and cache-hit rate applied to Artificial Analysis's "
        "per-token rates. The default is our own blend, 1 part input to 3 parts "
        "output with no cache — not AA's, whose published blend weights it the "
        "other way.",
        "Presets: chat 1:1, AA's 3:1, 15:1 observed (late-2025 averages in the "
        "a16z × OpenRouter State of AI study, quoted; a lower bound), and agentic "
        "300:1 with 90% cached (a SemiAnalysis estimate), or type your own.",
        "Cache-hit prices are Artificial Analysis's; where it publishes none, cache "
        "hits are billed at the full input price (no discount assumed), marked * "
        "and counted on the chart. Cache-write premiums are not modelled.",
        "Enter volume in millions of total tokens — 1M tokens ≈ 750,000 words or "
        "~1,500 pages. Chart sorts cheapest-first.",
    ],
    'table': [
        "Score = AA Intelligence Index (composite benchmark, higher = better) — "
        "about half are estimated by Artificial Analysis rather than measured, which "
        "the ↓CSV export marks per row.",
        "Value = Score ÷ Price (quality per dollar). Price = effective $/M tokens at "
        "the MIX setting; the column header names it (default: our 1:3 in:out "
        "blend, not AA's). Latency = time-to-first-token (TTFT) in seconds.",
    ],
    'local': [
        "COVERAGE: model specs come from Artificial Analysis, plus a short curated "
        "list of notable open-weight releases AA has not benchmarked yet — those are "
        "listed last in the ranking as \"unscored\" with no bar, never a guessed "
        "score, and are excluded from the Agent Stack recommendations. Each model "
        "appears once, as its best-scoring reasoning-effort variant (the tiers share "
        "weights, VRAM and speed); the ranking shows the top 30 that fit and counts "
        "the rest, and ↓CSV carries every row. Colour names a lab only where it "
        "stays tellable apart; other labs share grey.",
        "VRAM covers model WEIGHTS, the KV cache at the context you select, and "
        "about 1.5 GB of runtime overhead. The KV cache is the term that used to be "
        "missing and it is not small: Llama 3.1 8B at Q4 needs about 7 GB at 8k and "
        "about 22 GB at its advertised 128k — the KV cache alone goes from 1 GB to "
        "16 GB. Attention geometry comes from each model's published config where "
        "this dashboard carries it and is estimated from parameter count otherwise "
        "— the hover says which. About four in five models now carry a published "
        "figure; the rest are estimates, and an estimate here can be badly wrong for "
        "a model with sliding-window or latent attention, which the estimator "
        "cannot see.",
        "SCORES are benchmarked at the model's native precision and do NOT move "
        "with the quantization control: lowering it here shrinks VRAM and raises "
        "speed at no visible cost, but real quality does fall, steeply at the "
        "levels marked lossy. Artificial Analysis does not publish quantized scores "
        "and this dashboard will not guess one.",
        "SPEED is a roofline estimate (±30%) for token generation, not prompt "
        "processing: the slower of the memory stream (weights + KV cache) and the "
        "compute ceiling, and on every GPU here it is the memory stream, by 13x to "
        "198x. Quantizing does not buy its full byte ratio — FP16 to Q2 is about "
        "3.6x here and 1.9x–4.0x in published measurements, not the 8x a linear "
        "model implies — and the gain shrinks further at long context, because the "
        "KV cache stays FP16 whatever the weights are. It remains an upper bound "
        "for MoE models.",
        "The SPEED control picks which tok/s figure the chart shows: single stream "
        "is one conversation at a time; max throughput is the total across as many "
        "concurrent sessions as the KV cache fits in VRAM while each one still "
        "decodes at 10 tok/s or better (MLPerf's Llama-3.1-8B Server floor, ~480 "
        "words/minute). The two are 2-4x apart, so only one is on screen at a time "
        "and the hover carries the other. Both are decode-only and say nothing "
        "about queueing or time-to-first-token.",
        "$/GPU-HR turns the max-throughput figure into a serving cost per 1M output "
        "tokens at full use (the hover's Serving cost line): picking one of the "
        "seven datacenter cards InferenceX prices (H100, H200, B200, B300 SXM; "
        "MI300X, MI325X, MI355X) fills it with SemiAnalysis InferenceX's "
        "large-hyperscaler-volume owning or renting rate (OWN / RENT); other cards "
        "have no published rate, so type your own. It is decode only and "
        "uncalibrated, at this tab's 10 tok/s floor; Economics › Serve has the "
        "calibrated datacenter figure at 35 tok/s per user.",
        "To use it: select your GPU (or enter VRAM manually), choose a quantization "
        "level and a context length, and see which models fit — with estimated "
        "throughput.",
    ],
    'image': [
        "ELO scores from the Artificial Analysis Image Arena — blind pairwise human "
        "votes, so they rank models against each other and have no meaningful zero.",
        "Overall ELO and price refresh hourly. The per-style scores each column "
        "ranks on do not: Artificial Analysis retired its public per-category "
        "breakdown on 2026-09-07 and now publishes it only to API-key holders, so "
        "those are the last published figures and a model released since then "
        "carries none.",
    ],
    'video': [
        "Scraped hourly from the Artificial Analysis Video Arena. Elo comes from "
        "blind pairwise human votes, so it ranks models against each other and has "
        "no meaningful zero — read position, not distance from the axis.",
        "Text-to-video and image-to-video are separate arenas with separate prices, "
        "so MODE switches between them rather than blending them. Price is per "
        "minute of video at each provider's default settings; generation time is "
        "measured end-to-end and AA currently publishes it for only a handful of "
        "models.",
        "The ranked view shows current models; superseded preview builds stay in "
        "the ↓CSV export.",
    ],
    'economics_serve': [
        "GPU cost per 1M output tokens = $/GPU-hr ÷ (output tok/s per GPU × 3,600) "
        "× 1M ÷ use.",
        "Diamonds are MEASURED: SemiAnalysis InferenceX™ runs at its 8k-in / 1k-out "
        "workload with the MEDIAN user at 35 tok/s or faster (InferenceX's "
        "calculator target, applied to the median interactivity it publishes, so up "
        "to half the users of a run decode slower), the best configuration of any "
        "kind (FP4, MTP, disaggregated); click one to open its run. A model measured "
        "on a sibling checkpoint says so in its hover.",
        "Circles are MODELLED: this dashboard's Run Local roofline in a "
        "tensor-parallel FP8 mode on up to 8 GPUs, every session at 35 tok/s or "
        "faster and no more sessions than InferenceX itself swept (1,024 on up to 4 "
        "GPUs, 4,096 on 8), with a per-GPU bandwidth correction fitted to the "
        "like-for-like InferenceX runs (FP8, no speculative decoding, one node) — "
        "read them as ±2×; each hover gives its fit and residual. Large "
        "mixture-of-experts models served in FP4 or across nodes in practice come "
        "out conservative here.",
        "$/GPU-hr is InferenceX's OWN or RENT mark, which reflects "
        "large-hyperscaler-volume ownership and rental pricing "
        "(inferencex.semianalysis.com/about): the default, OWN, is about the "
        "cheapest cost any operator has, and most providers pay more. USE is the "
        "share of paid hours spent serving, 60% by default (InferenceX's "
        "profit-estimator default).",
        "The dashed 1× line compares OUTPUT price with cost: points below it are "
        "priced above their GPU cost on output price alone. That markup is a floor: "
        "InferenceX charges prefill time to output tokens, so the title and hover "
        "also give the same-mix markup (the bill for 8k input + 1k output tokens "
        "over the same cost). The price is Artificial Analysis's single reference "
        "price, not the cheapest host.",
        "Sanity anchor, from DeepSeek's CC0 disclosure for 2025-02-28 ($2 per "
        "H800-hour assumed): on this chart's basis, every node over the day "
        "($87,072 ÷ 168B output tokens) is about $0.52 per 1M output tokens against "
        "R1's $2.19 list price, about 4.2×; the decode GPUs alone, before prefill "
        "and at full use (~14.8k output tok/s per 8-GPU node), are about $0.30, "
        "about 7×. DeepSeek reported a 545% theoretical cost-profit margin.",
        "Credits: SemiAnalysis InferenceX (runs linked per point), Artificial "
        "Analysis (prices, specs), DeepSeek open-infra-index (CC0).",
    ],
    'economics_compute': [
        "Owned $/GPU-hr = [capex × capital-recovery factor + IT kW × PUE × $/kWh × "
        "8,760 + colocation $/kW-month × kW × 12] ÷ 8,760 h ÷ USE, where the "
        "capital-recovery factor spreads capex over the useful life at the cost of "
        "capital, r(1+r)^n ÷ ((1+r)^n − 1). The stack's three shares sum to 100% and "
        "are the units check: SemiAnalysis says colocation and electricity are "
        "typically under 20% of TCO.",
        "Every default carries its own source, date and status in the box's "
        "tooltip, and the GPU, preset and boxes override it: capex and kW are per "
        "GPU (an 8-GPU server ÷ 8, an NVL72 rack ÷ 72), and a typed capex or kW "
        "changes only the selected GPU.",
        "The cost of capital is the latest NY Fed SOFR plus a spread from a filing "
        "(CoreWeave's GPU-backed DDTL 5.5, SOFR + 5.50%), or a disclosed fixed rate "
        "(Lambda's 6.78%); useful lives are filed (CoreWeave 6 years, Nebius 4) or "
        "argued (SemiAnalysis 4, Burry 3, labelled as an argument). B300 and the AMD "
        "parts have no public all-in system price, so they stay unpriced until you "
        "type one. PUE 1.3 is an estimate (Uptime's global average is above 1.5; "
        "hyperscale sites run 1.1-1.15); the colocation rate is CBRE's US average, "
        "which excludes power, so metered power is added on top.",
        "The capex defaults are not on one price basis, and each bar names its own: "
        "H100 and GB200 are SemiAnalysis's 2025 hyperscaler-volume all-in "
        "estimates; H200 and B200 are 2026 street or appraisal prices for the bare "
        "server scaled by H100's all-in ratio (250/190), so they read dearer; GB300 "
        "is a 2026 rack deployment price that excludes core switching and storage.",
        "USE follows Serve's rule, owned or rented alike: cost per busy hour = cost "
        "per paid hour ÷ USE. The stack shows the owned cost per busy hour. The "
        "ladder and the weekly chart stay per paid GPU-hour, busy or idle, the basis "
        "every rental, list price and InferenceX mark is quoted on, so the owned "
        "mark sits there at full use; dividing every mark by USE would move them all "
        "alike and leave every ratio unchanged.",
        "The ladder runs from the owned cost through InferenceX's own and rent "
        "marks, SemiAnalysis's 1-year H100 contract index (two quoted points; its "
        "spot-contract composite is a different series and is not shown), "
        "GetDeploying's latest weekly marketplace medians, neocloud list prices "
        "(each dated in its hover), AWS capacity blocks and hyperscaler on-demand "
        "list prices, which the bot fetches weekly. No two series are ever joined or "
        "averaged. GetDeploying's medians are taken over offerings (one provider can "
        "list several), and each hover gives both counts. GetDeploying's "
        "'nvidia-h100' slug pools SXM, PCIe and NVL listings.",
        "Credits: rental medians GetDeploying (CC BY 4.0, "
        "getdeploying.com/dataset/gpu-prices); chip TDP Epoch AI, 'Data on Machine "
        "Learning Hardware' (CC BY 4.0); SOFR Federal Reserve Bank of New York; "
        "list prices Azure Retail Prices API and AWS Price List; TCO marks "
        "SemiAnalysis InferenceX™ (inferencex.semianalysis.com/chips/<chip>); "
        "financing SEC EDGAR filings and the press reports linked in each tooltip.",
    ],
    'economics_labs': [
        "BADGES: DISCLOSED = a company filing or an on-record company statement "
        "(filled marker); REPORTED = press citing people or documents (open marker); "
        "ESTIMATED = an analyst's model or Epoch AI's interpolation of disclosed "
        "period revenue (open diamond). Audit status is shown separately in the "
        "ledger: HKEX interim results and the S-1's half-year columns are "
        "unaudited, full-year results audited.",
        "RUN-RATE: Epoch AI's annualised revenue per lab, as reported; not like for "
        "like. Anthropic's S-1 (filed 2026-10-26) defines run-rate as "
        "trailing-28-day consumption fees ×13 plus monthly subscriptions ×12, and "
        "books cloud-marketplace sales gross, as principal (47% of FY2025 and 48% of "
        "H1 2026 revenue); OpenAI's ~$50B (Sep 2026) excludes cloud partners' "
        "resale, per Epoch's note. Epoch's period interpolations (booked half-year "
        "or quarterly revenue × 2 or × 4, backward-looking) sit on a dotted line of "
        "their own, never joined to a run-rate or ARR snapshot.",
        "MARGINS: each row is ONE definition and rows are not comparable to each "
        "other: Z AI's (Zhipu) and MiniMax's IFRS gross margins (HKEX filings), Z "
        "AI's cloud/API segment margin, Anthropic's gross margin on the cost of "
        "revenue in S-1 Note 8 (the statement of operations presents expenses by "
        "nature; Note 8 gives them by function, and substantially all cost of "
        "revenue is compute for customer-serving workloads), and separately its "
        "revenue less ALL compute and infrastructure including training (derived, "
        "not a gross margin), OpenAI's reported adjusted gross margin, DeepSeek's "
        "reported API margin and SemiAnalysis's estimate of Anthropic's "
        "inference-only margin. The axis stops at −120%; a point beyond it is marked "
        "at the edge with its value.",
        "COMMITMENTS: Anthropic's S-1 Note 13 purchase commitments (non-cancellable "
        "with over a year left, $54.6B at 2025-12-31, audited; fixed or minimum "
        "guaranteed, ~$265.6B at 2026-06-30, unaudited) over its latest Epoch "
        "run-rate dated at most 92 days earlier; CoreWeave's $103.7B of remaining "
        "performance obligations (10-Q, 2026-06-30) is the sell side, over its Q2 "
        "revenue ×4. Excluded and never charted: the $518B from a leaked draft "
        "prospectus (the public S-1 leaves the aggregate blank; EDGAR has no "
        "amendment as of 2026-10-27) and OpenAI's $665B, which is projected compute "
        "spend through 2030, not commitments.",
        "TOKENS: per day, using Epoch's convention (monthly ÷ 30, weekly ÷ 7, "
        "per-minute × 1,440); the symbol says what is counted (all products, API "
        "only, one product or model family) and a line never joins two scopes. A "
        "router's tokens (OpenRouter) are also counted by the labs it routes to. For "
        "context, quoted, not computed: the a16z × OpenRouter State of AI study (Dec "
        "2025) finds that across models a 10% lower price goes with only about "
        "0.5–0.7% more usage, a cross-sectional pattern, not a measured elasticity "
        "of demand (openrouter.ai/state-of-ai).",
        "STACK: each layer's dollars per year on one log axis, never added (one "
        "layer's revenue is the next one's cost): Nvidia's data-center revenue for "
        "the quarter to 26 Jul 2026 (10-Q, $89.0B) × 4; the four hyperscalers' 2026 "
        "capex guidance, which is spend, not revenue (Alphabet $195–205B and "
        "Microsoft ~$175B for calendar 2026 from their CFOs, Meta $135–145B from its "
        "Q3 release, Amazon ~$220B company-wide cash capex from its CEO); the sum of "
        "each lab's latest Epoch run-rate or ARR dated within a year of the newest "
        "(gross and net mixed, as above); and Cursor's reported >$4B ARR (early June "
        "2026), one app, so a floor. The dashed line reruns Sequoia's \"AI's $600B "
        "Question\" (David Cahn, June 2024: Nvidia's run-rate × 2 for the rest of the "
        "data centre × 2 for a 50% end-user gross margin) on the filed quarter, our "
        "arithmetic; the dotted line is Sequoia's 2024 figure.",
        "Ledger only, as it is capital raised, not a yearly flow: Dealroom's 29 "
        "neolabs, $23B raised, fewer than 35% with a commercial product (Oct 2026). "
        "↓CSV exports every number with its source.",
        "Data: Epoch AI, \"Data on AI companies\" (You, Croxton, Somala, Edelman), "
        "CC BY 4.0; company filings on SEC EDGAR and HKEXnews; Sequoia Capital; "
        "Dealroom; earnings-call statements as linked in the ledger.",
    ],
    'economics_open_gap': [
        "All three use Artificial Analysis data this dashboard scrapes "
        "(artificialanalysis.ai). Open or closed is AA's isOpenWeights flag; a row "
        "without one (every daily snapshot before 2026-10-27) is classified by a "
        "name match to AA's open-weights list, kept from every revision since "
        "2026-05-06 (exact, else the same name without its variant tag, as in "
        "2026-02-25's \"GLM-5\" for \"GLM-5 (Reasoning)\"), and otherwise counted as "
        "closed.",
        "SCORE GAP: in each daily snapshot, the best closed model's AA Intelligence "
        "Index minus the best open model's. AA has rescaled the index — found in the "
        "snapshots on 2026-06-16, 2026-09-05, 2026-09-08 and 2026-10-29, the first "
        "snapshot on Intelligence Index v5.0 (launched 2026-10-28; nearly every "
        "model moved, by one factor or, on v5.0, by x0.4 to x1.1) — so the line "
        "breaks there and a gap is comparable only within one segment; the "
        "2026-03-07 snapshot, when AA briefly published the index renormalised to "
        "0–100, is left out. Early snapshots can still overstate the gap where an "
        "open model left every list this dashboard kept.",
        "RELEASE LAG: for each closed model that set a new closed high when "
        "released, the time until an open model scoring at least as much was "
        "released; one not matched yet is censored, not given a number, and the "
        "median is Kaplan–Meier over matched and censored records alike. The oldest "
        "closed model AA still lists is a record only because everything stronger "
        "from its day was dropped, so it is drawn but not counted. A model whose "
        "score came from a later checkpoint under a new name (Qwen3.8 Max (0902)) is "
        "dated by the first snapshot with that score, not the base model's release. "
        "Only models AA still lists are seen (it drops superseded ones), so this is "
        "a survivor's view, not a measured history.",
        "PRICE: for each open family, the price of the cheapest closed family "
        "scoring at least as well, divided by the open model's price, both at our "
        "1:3 in:out blend (the Economics tab does not follow MIX). Only scores AA "
        "measured are used, on both sides: AA estimates the index for models it has "
        "not fully run, and on 2026-10-27 one estimated closed score was the "
        "cheapest peer for most open models. AA lists one price per model, not the "
        "cheapest third-party host, and the spread runs both ways, so no single "
        "ratio is claimed.",
        "For comparison, quoted: Epoch AI finds open models trailing the closed "
        "frontier by about four months since January 2026 on its Capabilities "
        "Index, a different method from the release lag here "
        "(epoch.ai/data-insights/open-closed-eci-gap, 2026-05-29, CC BY 4.0); "
        "OpenRouter reports open models above 62% of its tokens while closed models "
        "took 73% of spend in September 2026 "
        "(openrouter.ai/data/newsletter/the-token-state-of-play, quoted, not "
        "scraped); Nous Research measured open reasoning models using about 1.5–4× "
        "the tokens of closed ones on the same tasks "
        "(nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark, "
        "August 2025, quoted), which a per-token price ratio does not see — the "
        "hover adds AA's cost per task where AA publishes one for both models.",
    ],
}
METHODS['overview_price_dyn'] = METHODS['overview_price']


def caption_text(key: str) -> str:
    """The caption and its disclosure as one string — what a reader can reach."""
    return " ".join([CAPTIONS[key], *METHODS.get(key, [])])


CAPTION_TEXT: dict[str, str] = {k: caption_text(k) for k in CAPTIONS}


# A bare source domain or path in method text ("inferencex.semianalysis.com/about")
# becomes a link. "<chip>"-style placeholders end the match, and trailing
# punctuation stays text.
_LINK = re.compile(
    r"\b((?:[a-z0-9-]+\.)+(?:ai|com|org|io|gov|net)(?:/[A-Za-z0-9._~/%#?=&-]*)?)"
)


def link_segments(text: str) -> list:
    """Split method text into plain strings and {"text", "href"} link segments.

    Computed once here and shipped in the manifest, so app.py and docs/app.js
    render the same links without two regexes to keep in step."""
    out: list = []
    pos = 0
    for m in _LINK.finditer(text):
        url = m.group(1).rstrip(".")
        if text[m.start() + len(url):m.start() + len(url) + 1] == "<":
            continue  # a URL template, not a page
        if m.start() > pos:
            out.append(text[pos:m.start()])
        out.append({"text": url, "href": "https://" + url})
        pos = m.start() + len(url)
    if pos < len(text):
        out.append(text[pos:])
    return out


def method_payload() -> dict[str, list[list]]:
    """METHODS as link-split paragraphs, for the manifest."""
    return {k: [link_segments(p) for p in paras]
            for k, paras in METHODS.items() if paras}
