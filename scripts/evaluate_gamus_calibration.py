"""
DepthWizard - GAMUS Calibration Benchmark
Evaluates M1 Depth Estimation & M2 Metric Calibration against LiDAR Ground Truth from GAMUS.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.gamus_dataset import GAMUSDataset
from depth.models import DepthEstimator
from calibration.gcp_calibration import (
    sample_random_gcps,
    sample_grid_stratified_gcps,
    sample_terrain_structure_gcps,
    fit_gcp_calibration,
    apply_gcp_calibration,
    fit_gcp_calibration_nonlinear,
    apply_gcp_calibration_nonlinear
)
from evaluation.metrics import calculate_metrics
from evaluate_m1 import calculate_pairwise_accuracy


def run_gamus_calibration_benchmark(
    tile_id: str = "DC_03_26",
    split: str = "test",
    model_size: str = "Small",
    gcp_counts: list = [4, 10, 25, 50, 100],
    seed: int = 42
):
    print("=" * 72)
    print(f" DepthWizard M2 Calibration Evaluation on GAMUS Benchmark")
    print(f" Tile: {tile_id} ({split} split) | Backbone: Depth-Anything-V2-{model_size}")
    print("=" * 72)

    # 1. Load GAMUS Data
    print("\n[1/4] Loading GAMUS tile from Hugging Face...")
    ds = GAMUSDataset()
    tile = ds.load_tile(tile_id, split=split)
    rgb = tile["rgb"]
    gt_height = tile["height"]
    class_mask = tile.get("class_mask")

    h, w, c = rgb.shape
    valid_mask = np.isfinite(gt_height) & (gt_height > -10.0) & (gt_height < 300.0)
    num_valid = int(np.sum(valid_mask))
    print(f"  Raster size: {h}x{w}x{c}")
    print(f"  Valid GT pixels: {num_valid:,} / {h*w:,} ({num_valid/(h*w)*100:.1f}%)")
    print(f"  GT Elevation (AGL): min={float(np.min(gt_height[valid_mask])):.2f}m, "
          f"max={float(np.max(gt_height[valid_mask])):.2f}m, "
          f"mean={float(np.mean(gt_height[valid_mask])):.2f}m")

    # 2. Run Monocular Disparity Inference
    print(f"\n[2/4] Running Depth Anything V2 ({model_size}) Monocular Inference...")
    t0 = time.time()
    estimator = DepthEstimator()
    if not estimator.initialize(model_size=model_size):
        raise RuntimeError("Failed to initialize DepthEstimator.")

    pil_img = Image.fromarray(rgb)
    # Fast 1-view for evaluation speed
    pred_dict = estimator.predict_with_confidence(pil_img, tta_mode="none")
    if pred_dict is None:
        raise RuntimeError("Depth inference failed.")
    pred_disparity = pred_dict["relative_depth"]
    pred_rel_height = pred_dict.get("relative_height", pred_disparity)
    confidence = pred_dict["quality_map"]
    inf_time = time.time() - t0
    print(f"  Inference finished in {inf_time:.2f}s.")
    print(f"  Disparity range: [{pred_disparity.min():.2f}, {pred_disparity.max():.2f}]")
    print(f"  Rel Height (nDSM) range: [{pred_rel_height.min():.2f}, {pred_rel_height.max():.2f}]")

    # Raw disparity ordering accuracy
    pairwise_acc = calculate_pairwise_accuracy(gt_height, pred_disparity, num_samples=10000)
    corr_raw = float(np.corrcoef(pred_disparity[valid_mask], gt_height[valid_mask])[0, 1])
    print(f"  Raw Disparity Correlation: {corr_raw:+.4f}")
    print(f"  Pairwise Ordering Accuracy: {pairwise_acc:.2f}% (P(order correct))")

    # 3. Calibration Evaluation across GCP Regimes
    print("\n[3/4] Evaluating Metric Calibration (H = a * D + b) on Held-Out Pixels...")
    print("-" * 72)
    print(f"{'Sampling':<16} | {'K GCPs':<6} | {'Scale a':<8} | {'Offset b':<9} | {'MAE (m)':<8} | {'RMSE (m)':<8} | {'R^2':<7}")
    print("-" * 72)

    results = []
    strategies = [
        ("Random", sample_random_gcps),
        ("Grid-Stratified", sample_grid_stratified_gcps),
        ("Relief-Stratified", lambda vm, k, s: sample_terrain_structure_gcps(pred_disparity, vm, k, seed=s))
    ]

    for strat_name, sampler in strategies:
        for k in gcp_counts:
            try:
                # Sample K calibration GCPs
                rows, cols = sampler(valid_mask, k, seed)
                gcp_depth = pred_disparity[rows, cols]
                gcp_ref = gt_height[rows, cols]

                # Fit calibration: H = a * D + b
                a, b = fit_gcp_calibration(gcp_depth, gcp_ref)

                # Predict full metric DSM
                pred_metric_dsm = apply_gcp_calibration(pred_disparity, a, b)

                # Held-out mask: exclude the K training GCP points
                held_out_mask = valid_mask.copy()
                held_out_mask[rows, cols] = False

                # Compute metrics on held-out test pixels
                m = calculate_metrics(pred_metric_dsm, gt_height, valid_mask=held_out_mask)

                print(f"{strat_name:<16} | {k:<6} | {a:8.4f} | {b:9.2f} | {m['mae']:8.3f} | {m['rmse']:8.3f} | {m['r2']:7.3f}")
                results.append({
                    "strategy": strat_name,
                    "k": k,
                    "a": a,
                    "b": b,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "corr": m["correlation"]
                })
            except Exception as e:
                print(f"{strat_name:<16} | {k:<6} | FAILED: {str(e)}")

    print("-" * 72)

    # 4. Non-Linear Calibration Benchmark (Polynomial Deg 2 & 3)
    print("\n[4/4] Evaluating Non-Linear Calibration (K=50)...")
    print("-" * 72)
    k_nl = 50
    rows, cols = sample_grid_stratified_gcps(valid_mask, k_nl, seed=seed)
    gcp_depth = pred_disparity[rows, cols]
    gcp_ref = gt_height[rows, cols]
    held_out_mask = valid_mask.copy()
    held_out_mask[rows, cols] = False

    for method in ["linear", "polynomial_deg2", "polynomial_deg3"]:
        if method == "linear":
            a, b = fit_gcp_calibration(gcp_depth, gcp_ref)
            pred_h = apply_gcp_calibration(pred_disparity, a, b)
        else:
            model, _ = fit_gcp_calibration_nonlinear(gcp_depth, gcp_ref, method=method)
            pred_h = apply_gcp_calibration_nonlinear(pred_disparity, model, method=method)

        m = calculate_metrics(pred_h, gt_height, valid_mask=held_out_mask)
        print(f"  Method: {method:<16} | MAE: {m['mae']:.3f} m | RMSE: {m['rmse']:.3f} m | R^2: {m['r2']:.3f} | Pearson: {m['correlation']:.3f}")

    # 5. Class-Stratified Metric Breakdown (using best calibration)
    if class_mask is not None:
        print("\n[5/5] Semantic Class-Stratified Accuracy (K=50 Grid-Stratified)...")
        print("-" * 72)
        print(f"{'Class ID':<10} | {'Pixel Count':<12} | {'GT Mean (m)':<12} | {'MAE (m)':<10} | {'RMSE (m)':<10}")
        print("-" * 72)
        classes_present = np.unique(class_mask[valid_mask])
        for c_id in classes_present:
            c_mask = held_out_mask & (class_mask == c_id)
            c_count = int(np.sum(c_mask))
            if c_count > 100:
                c_diff = np.abs(pred_metric_dsm[c_mask] - gt_height[c_mask])
                c_mae = float(np.mean(c_diff))
                c_rmse = float(np.sqrt(np.mean(c_diff ** 2)))
                c_gt_mean = float(np.mean(gt_height[c_mask]))
                print(f"Class {int(c_id):<4} | {c_count:<12,} | {c_gt_mean:12.2f} | {c_mae:10.3f} | {c_rmse:10.3f}")
        print("-" * 72)

    print("\n GAMUS Calibration Benchmark Complete!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DepthWizard GAMUS Calibration Benchmark")
    parser.add_argument("--tile_id", type=str, default="DC_05_30", help="Tile ID in GAMUS (e.g., DC_05_30, DC_03_26)")
    parser.add_argument("--split", type=str, default="test", help="GAMUS split ('train', 'val', 'test')")
    parser.add_argument("--model_size", type=str, default="Small", choices=["Small", "Base", "Large"])
    args = parser.parse_args()

    run_gamus_calibration_benchmark(
        tile_id=args.tile_id,
        split=args.split,
        model_size=args.model_size
    )
