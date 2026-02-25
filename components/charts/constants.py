"""
Shared chart constants — imported by all chart modules.
Single source of truth for colors and styling.
"""

PROVIDER_COLORS: dict[str, str] = {
    "Anthropic":              "#c084fc",  # purple
    "OpenAI":                 "#34d399",  # emerald
    "Google":                 "#60a5fa",  # blue
    "Meta":                   "#fb923c",  # orange
    "DeepSeek":               "#f472b6",  # pink
    "Mistral":                "#facc15",  # yellow
    "xAI":                    "#a3e635",  # lime
    "Alibaba":                "#38bdf8",  # sky
    "Amazon":                 "#ff9900",  # aws orange
    "NVIDIA":                 "#22d3ee",  # cyan
    "Microsoft Azure":        "#818cf8",  # indigo
    "Cohere":                 "#f87171",  # red
    "Kimi":                   "#d4a1f5",  # lavender
    "Z AI":                   "#7dd3fc",  # light blue
    "MiniMax":                "#86efac",  # light green
    "InclusionAI":            "#fca5a5",  # light red
    "Xiaomi":                 "#6ee7b7",  # teal
    "Baidu":                  "#fde68a",  # amber
    "IBM":                    "#93c5fd",  # periwinkle
    "LG AI Research":         "#c4b5fd",  # violet
    "Nous Research":          "#f9a8d4",  # rose
    "Reka AI":                "#a78bfa",  # purple-blue
    "AI21 Labs":              "#34d399",  # green
    "Allen Institute for AI": "#67e8f9",  # light cyan
    "Inception":              "#fb7185",  # hot pink
    "Upstage":                "#fbbf24",  # amber
    "Perplexity":             "#a3a3a3",  # neutral
    "KwaiKAT":                "#f97316",  # amber-orange
    "Deep Cogito":            "#a8a29e",  # stone
}

DEFAULT_COLOR = "#6b7280"

# Shared chart theme tokens
BG    = "#111111"
GRID  = "rgba(255,255,255,0.04)"
TICK  = "#444444"
AXIS  = "#444444"
FONT  = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
