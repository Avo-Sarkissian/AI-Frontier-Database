"""
Shared chart constants — imported by all chart modules.
Single source of truth for colors and styling.
"""
import re as _re

# Ordered from most-specific to least-specific so longer matches win
_NAME_SUBS = [
    (_re.compile(r'\(\s*Adaptive\s+Reasoning\s*\)', _re.I), '(AR)'),
    (_re.compile(r'\(\s*Non-reasoning\s*\)',          _re.I), '(NR)'),
    (_re.compile(r'\(\s*Reasoning\s*\)',              _re.I), '(R)'),
    (_re.compile(r'\(\s*xhigh\s*\)',                 _re.I), '(XH)'),
    (_re.compile(r'\(\s*minimal\s*\)',               _re.I), '(min)'),
    (_re.compile(r'\(\s*high\s*\)',                  _re.I), '(H)'),
    (_re.compile(r'\(\s*medium\s*\)',                _re.I), '(M)'),
    (_re.compile(r'\(\s*low\s*\)',                   _re.I), '(L)'),
]


def clean_model_name(name: str, max_len: int = 32) -> str:
    """Abbreviate verbose model name suffixes for display."""
    for pattern, repl in _NAME_SUBS:
        name = pattern.sub(repl, name)
    if len(name) > max_len:
        name = name[:max_len - 1] + '…'
    return name

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

# Marker shapes per provider — encodes provider identity via shape in addition to
# color so colorblind users can distinguish the major labs.
# Cycles through 6 symbols for the most common providers; others get 'circle'.
PROVIDER_SHAPES: dict[str, str] = {
    "Anthropic":       "square",
    "OpenAI":          "diamond",
    "Google":          "triangle-up",
    "Meta":            "cross",
    "DeepSeek":        "star",
    "Mistral":         "pentagon",
    "xAI":             "hexagon",
    "Alibaba":         "triangle-down",
    "Amazon":          "diamond-open",
    "Microsoft Azure": "square-open",
    "NVIDIA":          "star-triangle-up",
    "Cohere":          "circle-open",
}
DEFAULT_SHAPE = "circle"

# Shared chart theme tokens
BG    = "#111111"
GRID  = "rgba(255,255,255,0.04)"
TICK  = "#888888"
AXIS  = "#999999"
FONT  = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
