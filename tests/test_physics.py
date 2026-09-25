"""Run Local physics: chunked/hybrid KV geometry, KV aliases, and the usable
memory of unified-memory presets.

Every expected KV size here is worked by hand from the published config, not
read back from the code under test:

    GQA bytes per token per layer = 2 (K and V) x n_kv_heads x head_dim x 2 (FP16)
"""

import pytest

from data.arch_scraper import _geometry
from data.local_models import (
    GPU_BY_NAME, GPUS, DEFAULT_CONTEXT_TOKENS, get_local_df, kv_cache_bytes,
    resolve_attention, usable_memory_gb, vram_breakdown,
)

GIB = 2 ** 30
CTX_128K = 131072


# ── Llama 4 chunked attention ────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["Llama 4 Scout", "Llama 4 Maverick"])
def test_llama4_kv_at_128k_counts_only_12_global_layers(name):
    """48 layers, 8 KV heads x 128, attention_chunk_size 8192, NoPE (global)
    on every 4th layer -> 12 global + 36 chunked.

    width   = 2 x 8 x 128 x 2 B            = 4096 B per token per layer
    global  = 12 x 131072                  = 1,572,864 token-layers
    chunked = 36 x 8192                    =   294,912 token-layers
    total   = 4096 x 1,867,776             = 7,650,410,496 B = 7.125 GiB
    (full attention on all 48 would be 4096 x 48 x 131072 = 24.0 GiB)
    """
    arch, src = resolve_attention(name, 109.0, 17.0, is_moe=True)
    assert src == "config"
    got = kv_cache_bytes(arch, CTX_128K, "FP16")
    assert got == 7_650_410_496
    assert got / GIB == pytest.approx(7.125)


def test_llama4_chunked_equals_full_at_or_below_one_chunk():
    """Below the chunk size every layer holds the whole sequence, so the
    default 8k context must price exactly as plain GQA did."""
    arch, _ = resolve_attention("Llama 4 Scout", 109.0, 17.0, is_moe=True)
    assert kv_cache_bytes(arch, 8192, "FP16") == 4096 * 48 * 8192


def _catalogue_row(name, **kw):
    """The live open-weight catalogue's row for `name`, or a skip. The
    catalogue is scraped: when AA flags a model deprecated the scraper drops it
    on a data-only bot commit, and indexing it by a hard-coded name turned that
    into a KeyError blamed on the refresh. The physics is asserted without the
    catalogue; this only adds the end-to-end check while the row exists."""
    df = get_local_df(**kw)
    hit = df[df["name"] == name]
    if hit.empty:
        pytest.skip(f"{name!r} is not in the scraped catalogue today")
    return hit.iloc[0]


def test_scout_q4_at_128k_fits_an_h100_now_that_chunking_is_priced():
    # Meta's published size: 109B total, 17B active. Not the scraped row.
    h100 = GPU_BY_NAME["NVIDIA H100 SXM"]
    vb = vram_breakdown(109, "Q4", CTX_128K, name="Llama 4 Scout", active_b=17,
                        is_moe=True, max_context_tokens=10_000_000)
    assert vb.kv_gib == pytest.approx(7.125, abs=0.01)
    assert vb.total_gib < h100["vram_gb"]


def test_scout_catalogue_row_prices_chunking_too():
    h100 = GPU_BY_NAME["NVIDIA H100 SXM"]
    row = _catalogue_row("Llama 4 Scout", quant="Q4", vram_gb=h100["vram_gb"],
                         bandwidth_gbps=h100["bandwidth_gbps"], hw_type="nvidia",
                         ctx_tokens=CTX_128K)
    assert row["kv_gb"] == pytest.approx(7.125, abs=0.01)
    assert row["vram_req_gb"] < h100["vram_gb"]


def _llama4_cfg(no_rope_layers):
    return {"text_config": {
        "num_hidden_layers": 48, "num_key_value_heads": 8, "head_dim": 128,
        "num_attention_heads": 40, "hidden_size": 5120,
        "attention_chunk_size": 8192, "no_rope_layers": no_rope_layers,
    }}


def test_arch_scraper_reads_attention_chunk_size_from_scout_config():
    """Scout's published config spells no_rope_layers out as [1,1,1,0] x 12."""
    geo = _geometry(_llama4_cfg([1, 1, 1, 0] * 12))
    assert geo["global_layers"] == 12
    assert geo["sliding_window"] == 8192
    assert geo["local_kind"] == "sliding"


def test_arch_scraper_expands_maverick_empty_no_rope_layers():
    """Maverick ships no_rope_layers: [] — transformers expands that with
    no_rope_layer_interval=4, so it is 12 global layers, not zero."""
    geo = _geometry(_llama4_cfg([]))
    assert geo["global_layers"] == 12
    assert geo["sliding_window"] == 8192


# ── KV aliases: R1 1776 and Jamba ────────────────────────────────────────────

def test_r1_1776_is_priced_as_deepseek_mla():
    """61 layers x (512 + 64) latent x 2 B = 70,272 B/token, one latent only."""
    arch, src = resolve_attention("R1 1776", 671.0, 37.0, is_moe=True)
    assert src == "config"
    assert kv_cache_bytes(arch, 1, "FP16") == 70_272


@pytest.mark.parametrize("name,expected", [
    # 9 attention layers x 2 x 8 x 128 x 2 B
    ("Jamba 1.7 Large", 36_864),
    # 4 attention layers x 2 x 8 x 128 x 2 B
    ("Jamba 1.7 Mini", 16_384),
    # 2 attention layers x 2 x 1 x 128 x 2 B
    ("Jamba Reasoning 3B", 1_024),
])
def test_jamba_only_attention_layers_carry_kv(name, expected):
    arch, src = resolve_attention(name, 52.0, 12.0, is_moe=True)
    assert src == "config"
    assert kv_cache_bytes(arch, 1, "FP16") == expected
    # Mamba layers hold a fixed state: the cache is exactly linear in context.
    assert kv_cache_bytes(arch, CTX_128K, "FP16") == expected * CTX_128K


def test_arch_scraper_reads_jamba_attn_layer_period():
    """Jamba Reasoning 3B's config.json: 28 layers, attn_layer_period 14,
    attn_layer_offset 7 -> attention at layers 7 and 21 only."""
    geo = _geometry({
        "num_hidden_layers": 28, "num_key_value_heads": 1,
        "num_attention_heads": 20, "hidden_size": 2560,
        "attn_layer_period": 14, "attn_layer_offset": 7, "sliding_window": None,
    })
    assert geo["global_layers"] == 2
    assert geo["local_kind"] == "linear"
    assert geo["head_dim"] == 128
    assert geo["sliding_window"] == ""


def test_arch_scraper_reads_jamba_layers_block_type():
    geo = _geometry({
        "num_hidden_layers": 8, "num_key_value_heads": 8,
        "num_attention_heads": 32, "hidden_size": 4096,
        "layers_block_type": ["mamba"] * 4 + ["attention"] + ["mamba"] * 3,
    })
    assert geo["global_layers"] == 1
    assert geo["local_kind"] == "linear"


# ── Unified-memory presets ───────────────────────────────────────────────────

@pytest.mark.parametrize("preset,usable", [
    ("Apple M1 (8 GB)", 5),              # 8 x 2/3 = 5.33, rounded down
    ("Apple M3 Pro (36 GB)", 24),        # 36 x 2/3 (<= 36 GB band)
    ("Apple M4 Pro (48 GB)", 36),        # 48 x 3/4
    ("Apple M4 Max (128 GB)", 96),       # 128 x 3/4
    ("Apple M2 Ultra (192 GB)", 144),    # 192 x 3/4
    ("A17 Pro (iPhone 15 Pro)", 6),      # 8 - 2 GB iOS reserve (unchanged)
    ("A19 Pro (iPhone 17 Pro)", 10),     # 12 - 2 (unchanged)
    ("Snapdragon X Elite (32 GB)", 28),  # 32 - 4 GB Windows reserve
    ("CPU only — DDR5 laptop", 12),      # 16 - 4
    ("NVIDIA RTX 5090", 32),             # discrete: untouched
])
def test_preset_usable_memory(preset, usable):
    assert GPU_BY_NAME[preset]["vram_gb"] == usable
    assert usable_memory_gb(GPU_BY_NAME[preset]) == usable


def test_every_unified_memory_preset_leaves_room_for_the_os():
    for g in GPUS:
        if g["hw_type"] in ("apple", "qualcomm", "cpu"):
            assert "ram_gb" in g, g["name"]
            assert 0 < g["vram_gb"] < g["ram_gb"], g["name"]
        else:
            assert "ram_gb" not in g, g["name"]


def test_command_a_plus_does_not_fit_m4_max_128gb():
    """Command A+ needs ~126.5 GiB at Q4; 128 GB of RAM fit it with 1.5 GiB for
    macOS. Metal's default wired limit on that machine is 96 GiB."""
    g = GPU_BY_NAME["Apple M4 Max (128 GB)"]
    # Command A+ as AA lists it: 218B total, 25B active, 192k context. Built
    # here rather than read from the scraped catalogue, which can drop it.
    vb = vram_breakdown(218, "Q4", DEFAULT_CONTEXT_TOKENS, name="Command A+",
                        active_b=25, is_moe=True, max_context_tokens=192_000)
    # The defect: it would squeeze into the raw 128 GB, not into what Metal wires.
    assert g["vram_gb"] < vb.total_gib < g["ram_gb"]
    row = _catalogue_row("Command A+", quant="Q4", vram_gb=g["vram_gb"],
                         bandwidth_gbps=g["bandwidth_gbps"], hw_type=g["hw_type"],
                         ctx_tokens=DEFAULT_CONTEXT_TOKENS)
    assert row["vram_req_gb"] > g["vram_gb"]
    assert row["fits"] == "no"
