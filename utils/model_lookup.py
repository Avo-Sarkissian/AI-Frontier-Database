"""
Model name normalization lookup table.
Different sources (Artificial Analysis, Epoch AI, LMArena, LiveBench)
use different spellings/versions for the same model.
"""

# canonical_name → list of aliases across sources
LOOKUP: dict[str, list[str]] = {
    # OpenAI
    "GPT-4o":              ["gpt-4o", "gpt4o", "GPT-4o", "GPT-4O", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"],
    "GPT-4o mini":         ["gpt-4o-mini", "GPT-4o mini", "gpt-4o-mini-2024-07-18"],
    "GPT-4 Turbo":         ["gpt-4-turbo", "GPT-4 Turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview"],
    "o1":                  ["o1", "o1-preview", "o1-2024-12-17"],
    "o1 mini":             ["o1-mini", "o1-mini-2024-09-12"],
    "o3":                  ["o3", "o3-2025-04-16"],
    # Anthropic
    "Claude 3.5 Sonnet":   ["claude-3-5-sonnet", "Claude 3.5 Sonnet", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
    "Claude 3.5 Haiku":    ["claude-3-5-haiku", "Claude 3.5 Haiku", "claude-3-5-haiku-20241022"],
    "Claude 3 Opus":       ["claude-3-opus", "Claude 3 Opus", "claude-3-opus-20240229"],
    "Claude 3 Haiku":      ["claude-3-haiku", "Claude 3 Haiku", "claude-3-haiku-20240307"],
    "Claude Sonnet 4.6":   ["Claude Sonnet 4.6", "claude-sonnet-4-6"],
    "Claude Opus 4.6":     ["Claude Opus 4.6", "claude-opus-4-6"],
    # Google
    "Gemini 1.5 Pro":      ["gemini-1.5-pro", "Gemini 1.5 Pro", "gemini-1.5-pro-002"],
    "Gemini 1.5 Flash":    ["gemini-1.5-flash", "Gemini 1.5 Flash", "gemini-1.5-flash-002"],
    "Gemini 2.5 Pro":      ["gemini-2.5-pro", "Gemini 2.5 Pro", "gemini-2.5-pro-preview"],
    "Gemini 2.5 Flash":    ["gemini-2.5-flash", "Gemini 2.5 Flash"],
    # Meta
    "Llama 3.1 405B":      ["llama-3.1-405b", "Llama 3.1 405B", "meta-llama/Meta-Llama-3.1-405B-Instruct"],
    "Llama 3.1 70B":       ["llama-3.1-70b", "Llama 3.1 70B", "meta-llama/Meta-Llama-3.1-70B-Instruct"],
    "Llama 3.3 70B":       ["llama-3.3-70b", "Llama 3.3 70B", "meta-llama/Llama-3.3-70B-Instruct"],
    "Llama 4 Maverick":    ["Llama 4 Maverick", "llama-4-maverick"],
    "Llama 4 Scout":       ["Llama 4 Scout", "llama-4-scout"],
    # DeepSeek
    "DeepSeek V3":         ["deepseek-v3", "DeepSeek V3", "DeepSeek-V3"],
    "DeepSeek V3.2":       ["DeepSeek V3.2", "deepseek-v3.2"],
    "DeepSeek R1":         ["deepseek-r1", "DeepSeek R1", "DeepSeek-R1"],
    # Mistral
    "Mistral Large":       ["mistral-large", "Mistral Large", "mistral-large-2411"],
    "Mistral Large 3":     ["Mistral Large 3", "mistral-large-3"],
    "Mistral Small":       ["mistral-small", "Mistral Small", "mistral-small-3.1"],
    # Alibaba
    "Qwen3 Max":           ["Qwen3 Max", "qwen3-max"],
    "Qwen3 235B":          ["Qwen3 235B A22B 2507", "qwen3-235b"],
    # xAI
    "Grok 3":              ["grok-3", "Grok 3"],
    "Grok 4":              ["Grok 4", "grok-4"],
}

# Reverse map: alias → canonical
_REVERSE: dict[str, str] = {}
for canonical, aliases in LOOKUP.items():
    for alias in aliases:
        _REVERSE[alias.lower()] = canonical
    _REVERSE[canonical.lower()] = canonical


def normalize(name: str) -> str:
    """Return the canonical model name, or the original if not found."""
    return _REVERSE.get(name.lower().strip(), name)


def provider_from_model(name: str) -> str:
    """Best-effort provider extraction from model name."""
    name_l = name.lower()
    if "gpt" in name_l or "o1" in name_l or "o3" in name_l or "openai" in name_l:
        return "OpenAI"
    if "claude" in name_l:
        return "Anthropic"
    if "gemini" in name_l:
        return "Google"
    if "llama" in name_l:
        return "Meta"
    if "deepseek" in name_l:
        return "DeepSeek"
    if "mistral" in name_l or "mixtral" in name_l:
        return "Mistral"
    if "grok" in name_l:
        return "xAI"
    if "qwen" in name_l:
        return "Alibaba"
    if "command" in name_l:
        return "Cohere"
    return "Other"
