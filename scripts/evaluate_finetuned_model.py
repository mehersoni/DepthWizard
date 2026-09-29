"""
Evaluation Script: Benchmark Fine-Tuned Depth Anything V2 vs Frozen Baseline
Evaluates on both ISPRS Potsdam (Held-out Tile 02_12) and GAMUS (Held-out Tile DC_03_26).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import rasterio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from depth.depth_model import load_model, estimate_depth
from evaluation.metrics import calculate_metrics
from scripts.gamus_dataset import GAMUSDataset
from train.losses import align_scale_shift


def evaluate_on_potsdam(model, processor, device, tile_id="2_12", crop_size=1024):
    rgb_path = f"data/potsdam/2_Ortho_RGB/top_potsdam_{tile_id}_RGB.tif"
    dsm_path = f"data/potsdam/1_DSM/dsm_potsdam_02_12.tif"

    if not os.path.exists(rgb_path) or not os.path.exists(dsm_path):
        return None

    with rasterio.open(rgb_path) as src_rgb, rasterio.open(dsm_path) as src_dsm:
        # Center crop 1024x1024 for high-resolution evaluation
        w_off = (src_rgb.width - crop_size) // 2
        h_off = (src_rgb.height - crop_size) // 2
        win = rasterio.windows.Window(w_off, h_off, crop_size, crop_size)

        rgb = src_rgb.read([1, 2, 3], window=win).transpose(1, 2, 0)
        dsm = src_dsm.read(1, window=win).astype(np.float32)

    valid = np.isfinite(dsm) & (dsm > -100.0)
    ground_z = np.percentile(dsm[valid], 5)
    agl_gt = np.clip(dsm - ground_z, 0.0, 150.0)

    # Inference
    t0 = time.time()
    pred_rel = estimate_depth(rgb, model=model, processor=processor, device=device)
    dur = time.time() - t0

    # Least squares affine alignment
    pred_t = torch.from_numpy(pred_rel).unsqueeze(0)
    gt_t = torch.from_numpy(agl_gt).unsqueeze(0)
    mask_t = torch.from_numpy(valid).unsqueeze(0)

    aligned = align_scale_shift(pred_t, gt_t, mask_t)
    if aligned.ndim == 4:
        aligned = aligned.squeeze(1)
    pred_aligned = aligned[0].numpy()

    metrics = calculate_metrics(pred_aligned, agl_gt, valid_mask=valid)
    metrics["latency_ms"] = float(dur * 1000.0)
    return metrics


def evaluate_on_gamus(model, processor, device, tile_id="DC_03_26", crop_size=1024):
    ds = GAMUSDataset()
    tile = ds.load_tile(tile_id, split="test", include_class=False)
    rgb = tile["rgb"]
    height_gt = tile["height"]

    if crop_size and (rgb.shape[0] > crop_size or rgb.shape[1] > crop_size):
        h, w = height_gt.shape
        y = (h - crop_size) // 2
        x = (w - crop_size) // 2
        rgb = rgb[y:y+crop_size, x:x+crop_size]
        height_gt = height_gt[y:y+crop_size, x:x+crop_size]

    valid = np.isfinite(height_gt) & (height_gt >= 0.0)

    t0 = time.time()
    pred_rel = estimate_depth(rgb, model=model, processor=processor, device=device)
    dur = time.time() - t0

    pred_t = torch.from_numpy(pred_rel).unsqueeze(0)
    gt_t = torch.from_numpy(height_gt).unsqueeze(0)
    mask_t = torch.from_numpy(valid).unsqueeze(0)

    aligned = align_scale_shift(pred_t, gt_t, mask_t)
    if aligned.ndim == 4:
        aligned = aligned.squeeze(1)
    pred_aligned = aligned[0].numpy()

    metrics = calculate_metrics(pred_aligned, height_gt, valid_mask=valid)
    metrics["latency_ms"] = float(dur * 1000.0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Depth Anything V2")
    parser.add_argument("--finetuned_path", type=str, default="checkpoints/depth_anything_v2_aerial",
                        help="Path to fine-tuned checkpoint folder")
    parser.add_argument("--base_model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace model ID of base frozen model")
    parser.add_argument("--crop_size", type=int, default=1024,
                        help="Evaluation crop size")
    parser.add_argument("--output_file", type=str, default="output/finetuning_benchmark_comparison.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("DepthWizard - Model Evaluation Benchmark")
    print(f"Base Model:       {args.base_model}")
    print(f"Fine-Tuned Model: {args.finetuned_path}")
    print(f"Crop Size:        {args.crop_size}x{args.crop_size}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # 1. Base Model Evaluation
    print("\n[1/2] Evaluating Base Frozen Model...")
    base_model, base_proc, _ = load_model(checkpoint_path=args.base_model, device=device)

    base_potsdam = evaluate_on_potsdam(base_model, base_proc, device, crop_size=args.crop_size)
    base_gamus = evaluate_on_gamus(base_model, base_proc, device, crop_size=args.crop_size)

    results["base"] = {
        "potsdam": base_potsdam,
        "gamus": base_gamus
    }

    # 2. Fine-Tuned Model Evaluation
    print("\n[2/2] Evaluating Fine-Tuned Aerial Model...")
    if not os.path.exists(args.finetuned_path):
        print(f"Warning: Fine-tuned checkpoint '{args.finetuned_path}' not found yet. Using test_run if available.")
        if os.path.exists("checkpoints/test_run"):
            args.finetuned_path = "checkpoints/test_run"

    ft_model, ft_proc, _ = load_model(checkpoint_path=args.finetuned_path, device=device)

    ft_potsdam = evaluate_on_potsdam(ft_model, ft_proc, device, crop_size=args.crop_size)
    ft_gamus = evaluate_on_gamus(ft_model, ft_proc, device, crop_size=args.crop_size)

    results["finetuned"] = {
        "potsdam": ft_potsdam,
        "gamus": ft_gamus
    }

    # Summary Table
    print("\n" + "=" * 75)
    print("DEPTH ANYTHING V2 FINE-TUNING BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Dataset / Split':<24} | {'Metric':<14} | {'Base (Zero-Shot)':<16} | {'Fine-Tuned (Ours)':<16} | {'Delta':<10}")
    print("-" * 75)

    comparisons = [
        ("ISPRS Potsdam (Held-out)", "correlation", base_potsdam, ft_potsdam, "+"),
        ("ISPRS Potsdam (Held-out)", "mae (m)", base_potsdam, ft_potsdam, "-", "mae"),
        ("ISPRS Potsdam (Held-out)", "rmse (m)", base_potsdam, ft_potsdam, "-", "rmse"),
        ("GAMUS AGL (Held-out)", "correlation", base_gamus, ft_gamus, "+"),
        ("GAMUS AGL (Held-out)", "mae (m)", base_gamus, ft_gamus, "-", "mae"),
        ("GAMUS AGL (Held-out)", "rmse (m)", base_gamus, ft_gamus, "-", "rmse"),
    ]

    for ds_name, metric_label, b_dict, f_dict, direction, *key in comparisons:
        k = key[0] if key else metric_label
        b_val = b_dict.get(k, 0.0) if b_dict else 0.0
        f_val = f_dict.get(k, 0.0) if f_dict else 0.0
        diff = f_val - b_val
        diff_str = f"{diff:+.4f}" if "corr" in metric_label else f"{diff:+.2f} m"
        b_str = f"{b_val:.4f}" if "corr" in metric_label else f"{b_val:.2f} m"
        f_str = f"{f_val:.4f}" if "corr" in metric_label else f"{f_val:.2f} m"
        print(f"{ds_name:<24} | {metric_label:<14} | {b_str:<16} | {f_str:<16} | {diff_str:<10}")

    print("=" * 75)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark comparison saved to '{args.output_file}'.")


if __name__ == "__main__":
    main()
