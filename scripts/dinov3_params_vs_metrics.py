#!/usr/bin/env python3
"""
DINOv3: Parameter count & estimated inference time vs. accuracy scatter plots.

Computes the number of parameters and theoretical GFLOPs (proxy for inference
time) for each listed DINOv3 variant and plots them against average accuracy
metrics extracted from DINOv3.md, helping identify the best version in the
params/speed/metrics trade-off space.

Usage:
    python scripts/dinov3_params_vs_metrics.py [--output plot.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Model architecture definitions (mirrors dinov3_src/hub/backbones.py)
# ---------------------------------------------------------------------------

VIT_CONFIGS: dict[str, dict] = {
    # LVD-1689M ViT models
    "ViT-S/16": dict(
        embed_dim=384, depth=12, num_heads=6, ffn_ratio=4,
        ffn_layer="mlp", qkv_bias=True,
    ),
    "ViT-S+/16": dict(
        embed_dim=384, depth=12, num_heads=6, ffn_ratio=6,
        ffn_layer="swiglu", qkv_bias=True,
    ),
    "ViT-B/16": dict(
        embed_dim=768, depth=12, num_heads=12, ffn_ratio=4,
        ffn_layer="mlp", qkv_bias=True,
    ),
    "ViT-L/16": dict(
        embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4,
        ffn_layer="mlp", qkv_bias=True,
    ),
    "ViT-H+/16": dict(
        embed_dim=1280, depth=32, num_heads=20, ffn_ratio=6,
        ffn_layer="swiglu", qkv_bias=True,
    ),
    "ViT-7B/16": dict(
        embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3,
        ffn_layer="swiglu64", qkv_bias=False,
    ),
}

# Keys: model name -> (depths, dims) for ConvNeXt
CONVNEXT_CONFIGS: dict[str, tuple[list[int], list[int]]] = {
    "ConvNeXt Tiny":   ([3, 3, 9, 3],  [96, 192, 384, 768]),
    "ConvNeXt Small":  ([3, 3, 27, 3], [96, 192, 384, 768]),
    "ConvNeXt Base":   ([3, 3, 27, 3], [128, 256, 512, 1024]),
    "ConvNeXt Large":  ([3, 3, 27, 3], [192, 384, 768, 1536]),
}


# ---------------------------------------------------------------------------
# 2. Accuracy metrics from DINOv3.md (LVD-1689M table, ViT backbones)
# ---------------------------------------------------------------------------

# Global tasks: IN-ReaL, IN-R, Obj.Net, Ox.-H
# Dense tasks:  ADE20k, NYU↓, DAVIS, NAVI, SPair
VIT_METRICS: dict[str, dict[str, float]] = {
    "ViT-S/16":   {"IN-ReaL": 87.0, "IN-R": 60.4, "Obj.Net": 50.9, "Ox.-H": 49.5,
                    "ADE20k": 47.0, "NYU": 0.403, "DAVIS": 72.7, "NAVI": 56.3, "SPair": 50.4},
    "ViT-S+/16":  {"IN-ReaL": 88.0, "IN-R": 68.8, "Obj.Net": 54.6, "Ox.-H": 50.0,
                    "ADE20k": 48.8, "NYU": 0.399, "DAVIS": 75.5, "NAVI": 57.1, "SPair": 55.2},
    "ViT-B/16":   {"IN-ReaL": 89.3, "IN-R": 76.7, "Obj.Net": 64.1, "Ox.-H": 58.5,
                    "ADE20k": 51.8, "NYU": 0.373, "DAVIS": 77.2, "NAVI": 58.8, "SPair": 57.2},
    "ViT-L/16":   {"IN-ReaL": 90.2, "IN-R": 88.1, "Obj.Net": 74.8, "Ox.-H": 63.1,
                    "ADE20k": 54.9, "NYU": 0.352, "DAVIS": 79.9, "NAVI": 62.3, "SPair": 61.3},
    "ViT-H+/16":  {"IN-ReaL": 90.3, "IN-R": 90.0, "Obj.Net": 78.6, "Ox.-H": 64.5,
                    "ADE20k": 54.8, "NYU": 0.352, "DAVIS": 79.3, "NAVI": 63.3, "SPair": 56.3},
    "ViT-7B/16":  {"IN-ReaL": 90.4, "IN-R": 91.1, "Obj.Net": 91.1, "Ox.-H": 72.8,
                    "ADE20k": 55.9, "NYU": 0.309, "DAVIS": 79.7, "NAVI": 64.4, "SPair": 58.7},
}

# ConvNeXt LVD-1689M: only columns available across all sizes — use @512px where available
# IN-ReaL@512, IN-R@512, Obj.Net@512, ADE20k, NYU↓
CONVNEXT_METRICS: dict[str, dict[str, float]] = {
    "ConvNeXt Tiny":  {"IN-ReaL": 87.7, "IN-R": 74.1, "Obj.Net": 58.7, "ADE20k": 42.7, "NYU": 0.448},
    "ConvNeXt Small": {"IN-ReaL": 88.7, "IN-R": 74.1, "Obj.Net": 58.7, "ADE20k": 44.8, "NYU": 0.432},
    "ConvNeXt Base":  {"IN-ReaL": 89.2, "IN-R": 78.2, "Obj.Net": 61.3, "ADE20k": 46.3, "NYU": 0.420},
    "ConvNeXt Large": {"IN-ReaL": 89.4, "IN-R": 82.4, "Obj.Net": 65.2, "ADE20k": 47.8, "NYU": 0.403},
}


# ---------------------------------------------------------------------------
# 3. GFLOPs computation (theoretical MACs, proxy for inference time)
# ---------------------------------------------------------------------------

# Input resolution used throughout the DINOv3 paper for these metrics.
IMG_SIZE = 224
PATCH_SIZE = 16


def _swiglu_hidden(embed_dim: int, ffn_ratio: float, align_to: int) -> int:
    hidden = int(embed_dim * ffn_ratio)
    d = int(hidden * 2 / 3)
    return d + (-d % align_to)


def _vit_gflops(cfg: dict) -> float:
    """Theoretical MACs for one 224×224 forward pass (in giga-MACs)."""
    embed_dim = cfg["embed_dim"]
    depth = cfg["depth"]
    ffn_ratio = cfg["ffn_ratio"]
    ffn_layer = cfg["ffn_layer"]

    n_storage_tokens = 4
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
    N = num_patches + 1 + n_storage_tokens       # 201 tokens

    macs = 0.0

    # PatchEmbed: Conv2d(3, embed_dim, 16, 16) -> 14×14 output
    macs += 3 * embed_dim * PATCH_SIZE * PATCH_SIZE * num_patches

    # Per transformer block
    block_macs = 0.0
    # QKV projection: Linear(embed_dim -> 3*embed_dim)
    block_macs += N * embed_dim * 3 * embed_dim
    # Attention: Q*K^T + A*V
    block_macs += 2 * N * N * embed_dim
    # Output projection: Linear(embed_dim -> embed_dim)
    block_macs += N * embed_dim * embed_dim

    # FFN
    if ffn_layer == "mlp":
        hidden = int(embed_dim * ffn_ratio)
        block_macs += N * embed_dim * hidden   # fc1
        block_macs += N * hidden * embed_dim   # fc2
    elif ffn_layer.startswith("swiglu"):
        align_to = {"swiglu": 8, "swiglu32": 32, "swiglu64": 64, "swiglu128": 128}[ffn_layer]
        sh = _swiglu_hidden(embed_dim, ffn_ratio, align_to)
        block_macs += N * embed_dim * sh * 2   # w1 + w2
        block_macs += N * sh * embed_dim        # w3

    macs += depth * block_macs

    # Final norms (negligible, ~4*N*embed_dim*depth)
    macs += 4 * N * embed_dim * depth

    return macs / 1e9


def _convnext_gflops(depths: list[int], dims: list[int]) -> float:
    """Theoretical MACs for one ConvNeXt forward pass at 224×224 (in giga-MACs)."""
    macs = 0.0
    H, W = IMG_SIZE, IMG_SIZE

    # Stem: Conv2d(3, dims[0], 4, stride=4)
    H, W = H // 4, W // 4
    macs += 3 * dims[0] * 4 * 4 * H * W

    for stage_idx in range(4):
        d = dims[stage_idx]
        n_blocks = depths[stage_idx]

        # Blocks at current spatial resolution
        for _ in range(n_blocks):
            # dwconv: Conv2d(d, d, 7, groups=d)
            macs += d * 7 * 7 * H * W
            # pwconv1: Linear(d -> 4d)
            macs += H * W * d * 4 * d
            # pwconv2: Linear(4d -> d)
            macs += H * W * 4 * d * d

        # Downsample (except last stage)
        if stage_idx < 3:
            d_next = dims[stage_idx + 1]
            H, W = H // 2, W // 2
            # Conv2d(d, d_next, 2, stride=2)
            macs += d * d_next * 2 * 2 * H * W

    return macs / 1e9


# ---------------------------------------------------------------------------
# 4. Parameter-count computation (manual, no torch import required)
# ---------------------------------------------------------------------------

def _vit_params(cfg: dict) -> int:
    """Compute parameter count for a ViT variant from its config dict."""
    embed_dim = cfg["embed_dim"]
    depth = cfg["depth"]
    ffn_ratio = cfg["ffn_ratio"]
    ffn_layer = cfg["ffn_layer"]
    qkv_bias = cfg["qkv_bias"]

    # Common settings across all listed ViT variants
    img_size = 224
    patch_size = 16
    in_chans = 3
    proj_bias = True
    ffn_bias = True
    layerscale_init = 1e-5  # >0 → LayerScale present
    n_storage_tokens = 4
    norm_eps = 1e-5 if cfg.get("norm_eps_bf16", True) else 1e-6
    # All use "layernormbf16" except maybe 7B, but it's the same params
    untie_global_and_local_cls_norm = cfg.get("untie_global_and_local_cls_norm", False)

    params = 0

    # --- PatchEmbed ---
    # Conv2d(in_chans, embed_dim, patch_size, patch_size, stride=patch_size)
    params += embed_dim * in_chans * patch_size * patch_size  # weight
    params += embed_dim  # bias
    # LayerNorm after conv (norm_layer is always present in PatchEmbed for dinov3)
    params += 2 * embed_dim  # weight + bias

    # --- CLS token, storage tokens, mask token ---
    params += embed_dim  # cls_token
    params += n_storage_tokens * embed_dim  # storage_tokens
    params += embed_dim  # mask_token

    # --- Transformer blocks ---
    for _ in range(depth):
        # norm1: LayerNorm
        params += 2 * embed_dim

        # Self-attention
        # qkv projection: Linear(embed_dim, 3*embed_dim)
        params += 3 * embed_dim * embed_dim
        if qkv_bias:
            params += 3 * embed_dim
        # proj: Linear(embed_dim, embed_dim)
        params += embed_dim * embed_dim
        if proj_bias:
            params += embed_dim

        # LayerScale after attention
        params += embed_dim

        # norm2: LayerNorm
        params += 2 * embed_dim

        # FFN
        if ffn_layer == "mlp":
            hidden = int(embed_dim * ffn_ratio)
            # fc1: Linear(embed_dim, hidden)
            params += embed_dim * hidden
            if ffn_bias:
                params += hidden
            # fc2: Linear(hidden, embed_dim)
            params += hidden * embed_dim
            if ffn_bias:
                params += embed_dim
        elif ffn_layer.startswith("swiglu"):
            # Determine alignment
            if ffn_layer == "swiglu":
                align_to = 8
            elif ffn_layer == "swiglu32":
                align_to = 32
            elif ffn_layer == "swiglu64":
                align_to = 64
            elif ffn_layer == "swiglu128":
                align_to = 128
            else:
                raise ValueError(f"Unknown ffn_layer: {ffn_layer}")

            hidden = int(embed_dim * ffn_ratio)
            d = int(hidden * 2 / 3)
            swiglu_hidden = d + (-d % align_to)

            # w1, w2: Linear(embed_dim, swiglu_hidden)
            params += 2 * embed_dim * swiglu_hidden
            if ffn_bias:
                params += 2 * swiglu_hidden
            # w3: Linear(swiglu_hidden, embed_dim)
            params += swiglu_hidden * embed_dim
            if ffn_bias:
                params += embed_dim

        # LayerScale after FFN
        params += embed_dim

    # --- Final norms ---
    params += 2 * embed_dim  # self.norm
    if untie_global_and_local_cls_norm:
        params += 2 * embed_dim  # self.local_cls_norm

    # --- RopePositionEmbedding (no trainable params) ---
    # The rope embed has no parameters — it's purely a function.

    return params


def _convnext_params(depths: list[int], dims: list[int]) -> int:
    """Compute parameter count for a ConvNeXt variant."""
    params = 0
    in_chans = 3

    # Stem: Conv2d(in_chans, dims[0], 4, stride=4) + LayerNorm(dims[0])
    params += dims[0] * in_chans * 4 * 4  # Conv2d weight
    params += dims[0]  # bias
    params += 2 * dims[0]  # LayerNorm (weight + bias)

    # Downsample layers (3 stages, i=0,1,2)
    for i in range(3):
        # LayerNorm(dims[i])
        params += 2 * dims[i]
        # Conv2d(dims[i], dims[i+1], 2, stride=2)
        params += dims[i + 1] * dims[i] * 2 * 2  # weight
        params += dims[i + 1]  # bias

    # Stages (4 stages)
    for stage_idx in range(4):
        d = dims[stage_idx]
        n_blocks = depths[stage_idx]
        for _ in range(n_blocks):
            # dwconv: Conv2d(d, d, 7, padding=3, groups=d)
            params += d * 1 * 7 * 7  # weight (in_channels/groups = 1)
            params += d  # bias
            # LayerNorm(d)
            params += 2 * d
            # pwconv1: Linear(d, 4*d)
            params += d * 4 * d
            params += 4 * d  # bias
            # pwconv2: Linear(4*d, d)
            params += 4 * d * d
            params += d  # bias
            # gamma (LayerScale, always >0 since init_value=1e-6)
            params += d

    # Final LayerNorm
    params += 2 * dims[-1]

    return params


def _try_torch_params() -> dict[str, int] | None:
    """Attempt to compute params by instantiating real models (no weight download)."""
    try:
        import torch
        from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
            DinoVisionTransformer,
        )
        from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
    except ImportError:
        return None

    results: dict[str, int] = {}

    # ViT models via builder functions that skip pretrained loading
    from lightly_train._models.dinov3.dinov3_src.hub.backbones import (
        dinov3_vits16,
        dinov3_vits16plus,
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vith16plus,
        dinov3_vit7b16,
        dinov3_convnext_tiny,
        dinov3_convnext_small,
        dinov3_convnext_base,
        dinov3_convnext_large,
    )

    vit_builders = {
        "ViT-S/16": dinov3_vits16,
        "ViT-S+/16": dinov3_vits16plus,
        "ViT-B/16": dinov3_vitb16,
        "ViT-L/16": dinov3_vitl16,
        "ViT-H+/16": dinov3_vith16plus,
        "ViT-7B/16": dinov3_vit7b16,
    }
    convnext_builders = {
        "ConvNeXt Tiny": dinov3_convnext_tiny,
        "ConvNeXt Small": dinov3_convnext_small,
        "ConvNeXt Base": dinov3_convnext_base,
        "ConvNeXt Large": dinov3_convnext_large,
    }

    for name, builder in vit_builders.items():
        model = builder(pretrained=False, in_chans=3)
        params = sum(p.numel() for p in model.parameters())
        results[name] = params

    for name, builder in convnext_builders.items():
        model = builder(pretrained=False, in_chans=3)
        params = sum(p.numel() for p in model.parameters())
        results[name] = params

    return results


# ---------------------------------------------------------------------------
# 5. Aggregation helpers
# ---------------------------------------------------------------------------

def _avg_metric(metrics: dict[str, float]) -> float:
    """Compute average accuracy.  NYU is ↓ (lower is better), invert via 1-val."""
    vals = []
    for k, v in metrics.items():
        if k == "NYU":
            # Invert so higher = better (1 - error)
            vals.append(1.0 - v)
        else:
            vals.append(v)
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot(params_dict: dict[str, int],
         gflops_dict: dict[str, float],
         vit_metrics: dict[str, dict[str, float]],
         convnext_metrics: dict[str, dict[str, float]],
         output_path: str = "dinov3_params_vs_metrics.png") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect data ----------------------------------------------------------
    vit_names: list[str] = []
    vit_params, vit_gflops, vit_avg = [], [], []
    for name in VIT_CONFIGS:
        if name in params_dict and name in vit_metrics:
            vit_names.append(name)
            vit_params.append(params_dict[name])
            vit_gflops.append(gflops_dict[name])
            vit_avg.append(_avg_metric(vit_metrics[name]))

    cn_names: list[str] = []
    cn_params, cn_gflops, cn_avg = [], [], []
    for name in CONVNEXT_CONFIGS:
        if name in params_dict and name in convnext_metrics:
            cn_names.append(name)
            cn_params.append(params_dict[name])
            cn_gflops.append(gflops_dict[name])
            cn_avg.append(_avg_metric(convnext_metrics[name]))

    n_total = len(vit_names) + len(cn_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_total))

    # Two subplots side by side ---------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Params vs Accuracy ---
    for i, name in enumerate(vit_names):
        ax1.scatter(vit_params[i], vit_avg[i], c=[colors[i]], marker="o", s=100,
                    edgecolors="black", linewidth=0.5, zorder=5)
        ax1.annotate(name, (vit_params[i], vit_avg[i]),
                     textcoords="offset points", xytext=(8, 6), fontsize=7, alpha=0.9)
    for i, name in enumerate(cn_names):
        ax1.scatter(cn_params[i], cn_avg[i], c=[colors[i + len(vit_names)]], marker="s",
                    s=100, edgecolors="black", linewidth=0.5, zorder=5)
        ax1.annotate(name, (cn_params[i], cn_avg[i]),
                     textcoords="offset points", xytext=(8, -10), fontsize=7, alpha=0.9)

    _draw_pareto(vit_params + cn_params, vit_avg + cn_avg, ax1)
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Average Accuracy (NYU inverted)")
    ax1.set_title("Params vs. Accuracy")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    from matplotlib.ticker import FuncFormatter
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- Right: GFLOPs vs Accuracy ---
    for i, name in enumerate(vit_names):
        ax2.scatter(vit_gflops[i], vit_avg[i], c=[colors[i]], marker="o", s=100,
                    edgecolors="black", linewidth=0.5, zorder=5)
        ax2.annotate(name, (vit_gflops[i], vit_avg[i]),
                     textcoords="offset points", xytext=(8, 6), fontsize=7, alpha=0.9)
    for i, name in enumerate(cn_names):
        ax2.scatter(cn_gflops[i], cn_avg[i], c=[colors[i + len(vit_names)]], marker="s",
                    s=100, edgecolors="black", linewidth=0.5, zorder=5)
        ax2.annotate(name, (cn_gflops[i], cn_avg[i]),
                     textcoords="offset points", xytext=(8, -10), fontsize=7, alpha=0.9)

    _draw_pareto(vit_gflops + cn_gflops, vit_avg + cn_avg, ax2)
    ax2.set_xlabel("Theoretical GFLOPs (proxy for inference time)")
    ax2.set_ylabel("Average Accuracy (NYU inverted)")
    ax2.set_title("Estimated Inference Cost vs. Accuracy")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.suptitle("DINOv3: Model Selection — Params / Speed / Accuracy (LVD-1689M)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def _draw_pareto(x_vals: list[float], y_vals: list[float], ax) -> None:
    """Draw Pareto frontier on an axis."""
    points = sorted(zip(x_vals, y_vals))
    px, py = [], []
    best = -1.0
    for x, y in points:
        if y > best:
            px.append(x)
            py.append(y)
            best = y
    if len(px) > 1:
        ax.plot(px, py, "r--", alpha=0.35, linewidth=1.5, label="Pareto frontier")
        ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DINOv3 params / speed vs metrics plot")
    parser.add_argument("--output", default="dinov3_params_vs_metrics.png",
                        help="Output image path")
    parser.add_argument("--no-torch", action="store_true",
                        help="Use manual param count instead of torch model instantiation")
    args = parser.parse_args()

    # Compute params
    if not args.no_torch:
        print("Attempting torch-based param count...")
        params_dict = _try_torch_params()
        if params_dict is None:
            print("Torch not available, falling back to manual calculation.")
            params_dict = {}
        else:
            print("  OK — using real model parameter counts.")
    else:
        params_dict = {}

    # Fall back to manual for any missing
    manual_params: dict[str, int] = {}
    for name, cfg in VIT_CONFIGS.items():
        manual_params[name] = _vit_params(cfg)
    for name, (depths, dims) in CONVNEXT_CONFIGS.items():
        manual_params[name] = _convnext_params(depths, dims)

    # Merge: prefer torch if available
    final_params: dict[str, int] = {}
    for name in list(VIT_CONFIGS) + list(CONVNEXT_CONFIGS):
        final_params[name] = params_dict.get(name, manual_params[name])

    # Compute GFLOPs
    gflops: dict[str, float] = {}
    for name, cfg in VIT_CONFIGS.items():
        gflops[name] = _vit_gflops(cfg)
    for name, (depths, dims) in CONVNEXT_CONFIGS.items():
        gflops[name] = _convnext_gflops(depths, dims)

    # Print table
    print("\n{:20s} {:>14s} {:>12s} {:>10s}".format(
        "Model", "Params", "GFLOPs", "Avg Metric"))
    print("-" * 60)
    for name in list(VIT_CONFIGS) + list(CONVNEXT_CONFIGS):
        metrics = VIT_METRICS.get(name) or CONVNEXT_METRICS.get(name) or {}
        avg = _avg_metric(metrics)
        print(f"{name:20s} {final_params[name]:>14,d} {gflops[name]:>12.2f} {avg:>10.2f}")

    # Plot
    plot(final_params, gflops, VIT_METRICS, CONVNEXT_METRICS, output_path=args.output)


if __name__ == "__main__":
    main()
