#!/usr/bin/env python3
"""Single-step BF16 → Marlin INT4 quantization for Voxtral Realtime 4B.

Produces a single consolidated.safetensors with:
  - Encoder + adapter + tok_embeddings + norms: BF16 (copied as-is)
  - Decoder linear weights: Marlin-packed INT4 (group_size=128)

The decoder linears are RTN-quantized (round-to-nearest, symmetric, per-group)
and packed directly into Marlin's tiled INT4 format in one step — no intermediate
GPTQ format, no multiple requantization cycles.

Why RTN over GPTQ: GPTQ's Hessian optimization destroys the critical SPAD-to-text
transition boundary in Voxtral's streaming architecture because calibration runs
through MistralForCausalLM (without ada_rms_norm_t_cond). RTN preserves it.

Marlin pack logic from IST-DASLab/marlin (Apache 2.0):
  https://github.com/IST-DASLab/marlin

Usage:
    # From original HuggingFace BF16 model:
    python3 quantize_marlin.py --model-dir path/to/Voxtral-Mini-4B-Realtime-2602

    # Output (default: ./output/consolidated.safetensors):
    python3 quantize_marlin.py --model-dir path/to/model --output-dir ./my-output

Requires: torch, numpy, safetensors
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ─── Model constants ─────────────────────────────────────────────────────────

N_LAYERS = 26
N_HEADS = 32
N_KV_HEADS = 8
DIM = 3072
HEAD_DIM = 128

# ─── Quantization constants ──────────────────────────────────────────────────

BITS = 4
GROUP_SIZE = 128
PACK_FACTOR = 32 // BITS   # 8 int4 values per int32
BIAS = 1 << (BITS - 1)     # 8 (uint4b8 encoding: stored = value + 8)
MAXQ = (1 << BITS) - 1     # 15

# ─── Mistral → HF naming for decoder linears ─────────────────────────────────

DECODER_LINEARS = {
    "attention.wq": ("self_attn.q_proj", True,  N_HEADS),     # needs Q/K permute
    "attention.wk": ("self_attn.k_proj", True,  N_KV_HEADS),  # needs Q/K permute
    "attention.wv": ("self_attn.v_proj", False, None),
    "attention.wo": ("self_attn.o_proj", False, None),
    "feed_forward.w1": ("mlp.gate_proj", False, None),
    "feed_forward.w2": ("mlp.down_proj", False, None),
    "feed_forward.w3": ("mlp.up_proj",   False, None),
}


# ─── Marlin permutation tables (from IST-DASLab/marlin, Apache 2.0) ─────────

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)

    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    return perm, scale_perm


_perm, _scale_perm = _get_perms()


# ─── Q/K head permutation (Mistral → HF interleaving) ────────────────────────

def permute_qk(w, n_heads, hidden_size):
    """Apply Mistral→HF head dimension interleaving for Q/K weights."""
    head_dim = w.shape[0] // n_heads
    return (
        w.view(n_heads, head_dim // 2, 2, hidden_size)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, hidden_size)
    )


# ─── Single-step RTN quantize + Marlin pack ──────────────────────────────────

def quantize_and_pack_marlin(w_bf16, group_size=GROUP_SIZE):
    """RTN-quantize a BF16 weight and pack into Marlin format in one step.

    Args:
        w_bf16: [N_out, K] BF16/FP16 weight tensor

    Returns:
        B: [K//16, 2*N_out] int32 (Marlin-packed weights)
        s: [K//group_size, N_out] fp16 (Marlin-permuted scales)
    """
    N_out, K = w_bf16.shape
    n_groups = K // group_size
    tile = 16

    # ── Step 1: Compute per-group RTN scales ──
    # Work in [K, N] layout for Marlin packing
    w = w_bf16.t().float().contiguous()  # [K, N]
    w_grouped = w.reshape(n_groups, group_size, N_out)
    max_val = w_grouped.abs().amax(dim=1).clamp(min=1e-10)  # [n_groups, N]
    scales = (max_val / BIAS).half()  # [n_groups, N] — scale = max_abs / 8

    # ── Step 2: Quantize to uint4 ──
    s_expanded = scales.float().unsqueeze(1).expand_as(w_grouped)  # [n_groups, gs, N]
    w_int = torch.round(w_grouped / s_expanded).clamp(-BIAS, BIAS - 1).int()
    w_uint = (w_int + BIAS).clamp(0, MAXQ)  # uint4b8: [-8,7] → [0,15]
    w_uint = w_uint.reshape(K, N_out)  # [K, N]

    # ── Step 3: Permute scales for Marlin ──
    s = scales.clone()  # [n_groups, N]
    s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    s = s.reshape((-1, N_out)).contiguous()

    # ── Step 4: Tile into 16×16 blocks ──
    w_tiled = w_uint.reshape(K // tile, tile, N_out // tile, tile)
    w_tiled = w_tiled.permute(0, 2, 1, 3)
    w_tiled = w_tiled.reshape(K // tile, N_out * tile)

    # ── Step 5: Apply Marlin permutation ──
    res = w_tiled.reshape((-1, _perm.numel()))[:, _perm].reshape(w_tiled.shape)

    # ── Step 6: Pack 8 int4 values into each int32 ──
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res_np = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res_np[:, i::8] << (4 * i)
    B = torch.from_numpy(q.astype(np.int32))

    return B, s.half()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral BF16 → single-file Marlin INT4")
    parser.add_argument("--model-dir", required=True,
                        help="Directory with consolidated.safetensors (BF16, Mistral format)")
    parser.add_argument("--output-dir", default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    sf_path = os.path.join(args.model_dir, "consolidated.safetensors")
    if not os.path.exists(sf_path):
        print(f"Error: {sf_path} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "consolidated.safetensors")

    print(f"Input:  {sf_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: RTN {BITS}-bit, group_size={GROUP_SIZE}, uint4b8 Marlin")
    print()

    sf = safe_open(sf_path, framework="pt", device="cpu")
    all_keys = list(sf.keys())
    tensors = {}
    t0 = time.time()

    # ── Pass 1: Copy non-decoder-linear tensors as-is ──
    # These are encoder, adapter, tok_embeddings, norms, ada_rms_norm, final norm
    decoder_linear_keys = set()
    for layer_idx in range(N_LAYERS):
        for mistral_name in DECODER_LINEARS:
            decoder_linear_keys.add(f"layers.{layer_idx}.{mistral_name}.weight")

    n_copied = 0
    for key in all_keys:
        if key in decoder_linear_keys:
            continue
        tensors[key] = sf.get_tensor(key)
        n_copied += 1

    print(f"Copied {n_copied} non-linear tensors (encoder, norms, embeddings, etc.)")

    # ── Pass 2: Quantize decoder linears → Marlin ──
    n_quantized = 0
    for layer_idx in range(N_LAYERS):
        for mistral_name, (hf_name, needs_permute, n_heads) in DECODER_LINEARS.items():
            src_key = f"layers.{layer_idx}.{mistral_name}.weight"
            w = sf.get_tensor(src_key).half()  # bf16 → fp16 for torch ops

            # Apply Q/K head permutation if needed
            if needs_permute:
                w = permute_qk(w, n_heads, DIM)

            # Single-step quantize + Marlin pack
            B, s = quantize_and_pack_marlin(w)
            del w

            out_prefix = f"layers.{layer_idx}.{hf_name}"
            tensors[f"{out_prefix}.B"] = B
            tensors[f"{out_prefix}.s"] = s
            n_quantized += 1

        gc.collect()
        elapsed = time.time() - t0
        print(f"  Layer {layer_idx + 1}/{N_LAYERS} quantized ({elapsed:.1f}s)")

    print(f"\nQuantized {n_quantized} decoder linear weights to Marlin INT4")
    print(f"Total tensors in output: {len(tensors)}")

    # ── Save ──
    print(f"\nSaving to {output_path}...")
    save_file(tensors, output_path)
    file_size = os.path.getsize(output_path)
    print(f"Output: {file_size / (1024**3):.2f} GB ({len(tensors)} tensors)")

    # ── Copy auxiliary files ──
    for aux in ["params.json", "tekken.json"]:
        src = os.path.join(args.model_dir, aux)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, aux))
            print(f"Copied {aux}")

    print(f"\nDone in {time.time() - t0:.1f}s")

    # ── Verify tensor names ──
    print(f"\nSample Marlin tensor names:")
    marlin_keys = sorted(k for k in tensors if k.endswith(".B"))[:5]
    for k in marlin_keys:
        print(f"  {k}: {list(tensors[k].shape)} {tensors[k].dtype}")
        sk = k[:-2] + ".s"
        print(f"  {sk}: {list(tensors[sk].shape)} {tensors[sk].dtype}")


if __name__ == "__main__":
    main()