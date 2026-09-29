import os
import sys
import time
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth.depth_model import estimate_depth, load_model
from calibration.gcp_calibration import (
    sample_random_gcps,
    sample_grid_stratified_gcps,
    sample_terrain_structure_gcps,
    fit_gcp_calibration,
    apply_gcp_calibration
)
from evaluation.metrics import compute_error_map


def run_gcp_experiment(
    tile_id="2_10",
    k_values=[1, 3, 5, 10, 20, 50, 100],
    seeds=[42, 101, 2024, 777, 999],
    output_dir="outputs"
):
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    print("=" * 80, flush=True)
    print("DepthWizard M2 - Sparse GCP Elevation Calibration Experiment (Ultra-Fast)", flush=True)
    print("=" * 80, flush=True)

    rgb_path = f"data/potsdam/2_Ortho_RGB/top_potsdam_{tile_id}_RGB.tif"
    parts = tile_id.split("_")
    dsm_str = f"{int(parts[0]):02d}_{int(parts[1]):02d}"
    ref_dsm_path = f"data/potsdam/1_DSM/dsm_potsdam_{dsm_str}.tif"

    # [1/4] Load RGB & Reference DSM
    print(f"\n[1/4] Loading RGB ({rgb_path}) and Reference DSM ({ref_dsm_path})...", flush=True)
    with rasterio.open(rgb_path) as src_rgb:
        rgb = src_rgb.read().transpose(1, 2, 0)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

    with rasterio.open(ref_dsm_path) as src_ref:
        ref_dsm = src_ref.read(1).astype(np.float32)
        nodata = src_ref.nodata if src_ref.nodata is not None else -9999.0

    valid_mask = np.isfinite(ref_dsm) & (ref_dsm != nodata) & (np.abs(ref_dsm) < 1e5)
    r_valid = ref_dsm[valid_mask]
    ref_min = float(np.min(r_valid))
    ref_max = float(np.max(r_valid))
    ref_span = ref_max - ref_min
    print(f"      Reference DSM: [{ref_min:.2f}, {ref_max:.2f}] m (Span: {ref_span:.2f} m, Valid: {np.sum(valid_mask):,} pixels)", flush=True)

    # [2/4] Monocular Depth Inference
    print(f"\n[2/4] Running Depth Anything V2 Inference...", flush=True)
    model, proc, dev = load_model()
    t0 = time.time()
    depth = estimate_depth(rgb, model=model, processor=proc, device=dev)
    print(f"      Depth estimated in {time.time()-t0:.2f}s (Range: [{np.min(depth):.4f}, {np.max(depth):.4f}])", flush=True)

    # Pre-extract evaluation arrays
    eval_sub = 4  # 1500x1500 = 2,250,000 pixels
    depth_eval = depth[::eval_sub, ::eval_sub].ravel().astype(np.float64)
    ref_eval = ref_dsm[::eval_sub, ::eval_sub].ravel().astype(np.float64)
    mask_eval = valid_mask[::eval_sub, ::eval_sub].ravel()

    d_arr = depth_eval[mask_eval]
    h_arr = ref_eval[mask_eval]
    var_h = float(np.var(h_arr))

    # Pre-compute valid coordinate indices once (O(1) sampling afterwards!)
    print("      Pre-computing spatial index caches for O(1) sampling...", flush=True)
    valid_coords = np.argwhere(valid_mask)
    d_valid = depth[valid_mask]
    p30 = np.percentile(d_valid, 30.0)
    p70 = np.percentile(d_valid, 70.0)
    terrain_coords = np.argwhere(valid_mask & (depth <= p30))
    struct_coords = np.argwhere(valid_mask & (depth >= p70))
    print(f"      Caches ready: Valid={len(valid_coords):,}, Terrain={len(terrain_coords):,}, Structure={len(struct_coords):,}", flush=True)

    # [3/4] Run Sparse GCP Calibration Across Strategies & K Values
    strategies = ["random", "grid_stratified", "terrain_structure"]
    results_by_strategy = {s: {} for s in strategies}

    print(f"\n[3/4] Evaluating K={k_values} with {len(seeds)} random seeds across {len(strategies)} strategies...", flush=True)

    saved_exemplar_dsms = {}

    for strat in strategies:
        print(f"\n--- Strategy: {strat.upper()} ---", flush=True)
        results_by_strategy[strat] = {}

        for k in k_values:
            mae_list, rmse_list, r_list, r2_list = [], [], [], []
            a_list, b_list = [], []
            p_min_list, p_max_list, span_ratio_list = [], [], []

            for seed in seeds:
                if strat == "random":
                    rows, cols = sample_random_gcps(valid_mask, k=k, seed=seed, valid_indices=valid_coords)
                elif strat == "grid_stratified":
                    rows, cols = sample_grid_stratified_gcps(valid_mask, k=k, seed=seed)
                elif strat == "terrain_structure":
                    rows, cols = sample_terrain_structure_gcps(depth, valid_mask, k=k, seed=seed,
                                                              terrain_indices=terrain_coords,
                                                              structure_indices=struct_coords)

                gcp_d = depth[rows, cols]
                gcp_h = ref_dsm[rows, cols]

                # Fit calibration
                a, b = fit_gcp_calibration(gcp_d, gcp_h, scale_prior=6.9373)

                if seed == 42 and strat == "grid_stratified" and k in [1, 5, 20]:
                    pred_full = apply_gcp_calibration(depth, a, b)
                    saved_exemplar_dsms[k] = (pred_full, rows, cols)

                # Vectorized evaluation
                pred_eval = a * d_arr + b
                diff = pred_eval - h_arr
                mae = float(np.mean(np.abs(diff)))
                mse = float(np.mean(diff ** 2))
                rmse = float(np.sqrt(mse))
                r2 = float(1.0 - mse / var_h) if var_h > 0 else 0.0

                cov_dh = np.cov(d_arr, h_arr)[0, 1]
                std_d = np.std(d_arr)
                std_h = np.std(h_arr)
                corr = float(cov_dh / (std_d * std_h)) if (std_d * std_h) > 0 else 0.0

                p_min = float(np.min(pred_eval))
                p_max = float(np.max(pred_eval))
                s_ratio = (p_max - p_min) / ref_span

                mae_list.append(mae)
                rmse_list.append(rmse)
                r_list.append(corr)
                r2_list.append(r2)
                a_list.append(a)
                b_list.append(b)
                p_min_list.append(p_min)
                p_max_list.append(p_max)
                span_ratio_list.append(s_ratio)

            k_summary = {
                "k": k,
                "mae_mean": float(np.mean(mae_list)),
                "mae_std": float(np.std(mae_list)),
                "rmse_mean": float(np.mean(rmse_list)),
                "rmse_std": float(np.std(rmse_list)),
                "correlation_mean": float(np.mean(r_list)),
                "correlation_std": float(np.std(r_list)),
                "r2_mean": float(np.mean(r2_list)),
                "r2_std": float(np.std(r2_list)),
                "scale_a_mean": float(np.mean(a_list)),
                "scale_a_std": float(np.std(a_list)),
                "offset_b_mean": float(np.mean(b_list)),
                "offset_b_std": float(np.std(b_list)),
                "pred_min_mean": float(np.mean(p_min_list)),
                "pred_max_mean": float(np.mean(p_max_list)),
                "span_ratio_mean": float(np.mean(span_ratio_list)),
                "span_ratio_std": float(np.std(span_ratio_list))
            }
            results_by_strategy[strat][str(k)] = k_summary
            print(f"   K={k:<3d} | MAE: {k_summary['mae_mean']:.4f} ± {k_summary['mae_std']:.4f} m | RMSE: {k_summary['rmse_mean']:.4f} ± {k_summary['rmse_std']:.4f} m | R²: {k_summary['r2_mean']:.4f} ± {k_summary['r2_std']:.4f} | a={k_summary['scale_a_mean']:.2f}, b={k_summary['offset_b_mean']:.2f}m", flush=True)

    # [4/4] Save JSON and Figures
    json_record = {
        "tile_id": tile_id,
        "k_values": k_values,
        "seeds": seeds,
        "reference_dsm_range_m": [ref_min, ref_max],
        "reference_dsm_span_m": ref_span,
        "results_by_strategy": results_by_strategy
    }
    json_path = os.path.join(output_dir, "metrics", "gcp_calibration.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(json_record, fj, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}", flush=True)

    # Figure 1: GCP Calibration Curves
    print("Generating Figure 1: gcp_calibration_curve.png...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)

    strat_colors = {
        "random": "#e74c3c",
        "grid_stratified": "#2980b9",
        "terrain_structure": "#27ae60"
    }
    strat_labels = {
        "random": "Random Spatial",
        "grid_stratified": "Grid-Stratified Spatial",
        "terrain_structure": "Terrain + Structure Stratified"
    }

    # Plot MAE vs K
    for s_key in strategies:
        ks = [int(k) for k in k_values]
        maes = [results_by_strategy[s_key][str(k)]["mae_mean"] for k in ks]
        stds = [results_by_strategy[s_key][str(k)]["mae_std"] for k in ks]
        axes[0, 0].plot(ks, maes, marker="o", color=strat_colors[s_key], label=strat_labels[s_key], lw=2)
        axes[0, 0].fill_between(ks, np.array(maes)-np.array(stds), np.array(maes)+np.array(stds), color=strat_colors[s_key], alpha=0.15)
    axes[0, 0].set_title("A. Mean Absolute Error (MAE) vs Number of GCPs (K)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Number of GCPs (K)")
    axes[0, 0].set_ylabel("MAE (metres)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()

    # Plot RMSE vs K
    for s_key in strategies:
        ks = [int(k) for k in k_values]
        rmses = [results_by_strategy[s_key][str(k)]["rmse_mean"] for k in ks]
        stds = [results_by_strategy[s_key][str(k)]["rmse_std"] for k in ks]
        axes[0, 1].plot(ks, rmses, marker="s", color=strat_colors[s_key], label=strat_labels[s_key], lw=2)
        axes[0, 1].fill_between(ks, np.array(rmses)-np.array(stds), np.array(rmses)+np.array(stds), color=strat_colors[s_key], alpha=0.15)
    axes[0, 1].set_title("B. Root Mean Squared Error (RMSE) vs Number of GCPs (K)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Number of GCPs (K)")
    axes[0, 1].set_ylabel("RMSE (metres)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend()

    # Plot R² vs K
    for s_key in strategies:
        ks = [int(k) for k in k_values]
        r2s = [results_by_strategy[s_key][str(k)]["r2_mean"] for k in ks]
        stds = [results_by_strategy[s_key][str(k)]["r2_std"] for k in ks]
        axes[1, 0].plot(ks, r2s, marker="^", color=strat_colors[s_key], label=strat_labels[s_key], lw=2)
        axes[1, 0].fill_between(ks, np.array(r2s)-np.array(stds), np.array(r2s)+np.array(stds), color=strat_colors[s_key], alpha=0.15)
    axes[1, 0].set_title("C. Coefficient of Determination (R²) vs Number of GCPs (K)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Number of GCPs (K)")
    axes[1, 0].set_ylabel("R² Score")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    # Plot Scale Parameter 'a' vs K
    for s_key in strategies:
        ks = [int(k) for k in k_values]
        asc = [results_by_strategy[s_key][str(k)]["scale_a_mean"] for k in ks]
        stds = [results_by_strategy[s_key][str(k)]["scale_a_std"] for k in ks]
        axes[1, 1].plot(ks, asc, marker="d", color=strat_colors[s_key], label=strat_labels[s_key], lw=2)
        axes[1, 1].fill_between(ks, np.array(asc)-np.array(stds), np.array(asc)+np.array(stds), color=strat_colors[s_key], alpha=0.15)
    axes[1, 1].axhline(10.6501, color="black", linestyle=":", label="Full-Tile In-Sample Least-Squares (a=10.65)")
    axes[1, 1].set_title("D. Estimated Metric Scale (a) vs Number of GCPs (K)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Number of GCPs (K)")
    axes[1, 1].set_ylabel("Fitted Scale Parameter (a)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()

    plt.suptitle(f"DepthWizard M2: Sparse GCP Elevation Calibration Curves (Potsdam Tile {tile_id})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    curve_fig_path = os.path.join(output_dir, "figures", "gcp_calibration_curve.png")
    plt.savefig(curve_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved curve figure to: {curve_fig_path}", flush=True)

    # Figure 2: Comparison Maps (1, 5, 20 GCPs)
    print("Generating Figure 2: gcp_comparison_maps.png...", flush=True)
    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 11), dpi=150)

    sub = 6
    rgb_s = rgb[::sub, ::sub]
    ref_s = ref_dsm[::sub, ::sub]

    vmin_h, vmax_h = float(np.min(ref_s)), float(np.max(ref_s))

    # [0,0] RGB
    axes2[0, 0].imshow(rgb_s)
    axes2[0, 0].set_title(f"1. Potsdam RGB (Tile {tile_id})", fontsize=11, fontweight="bold")
    axes2[0, 0].axis("off")

    # [1,0] Reference DSM
    im_ref = axes2[1, 0].imshow(ref_s, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes2[1, 0].set_title("2. LiDAR Reference DSM Ground Truth", fontsize=11, fontweight="bold")
    axes2[1, 0].axis("off")
    cb = fig2.colorbar(im_ref, ax=axes2[1, 0], fraction=0.046, pad=0.04)
    cb.set_label("Elevation (m)")

    col_idx = 1
    for k_val in [1, 5, 20]:
        pred_full, r_pts, c_pts = saved_exemplar_dsms[k_val]
        pred_s = pred_full[::sub, ::sub]
        err_s = compute_error_map(pred_full, ref_dsm, valid_mask=valid_mask)[::sub, ::sub]

        diff_k = (pred_full[valid_mask] - ref_dsm[valid_mask]).astype(np.float64)
        mae_k = float(np.mean(np.abs(diff_k)))
        rmse_k = float(np.sqrt(np.mean(diff_k ** 2)))
        r2_k = float(1.0 - np.mean(diff_k ** 2) / np.var(ref_dsm[valid_mask]))
        vmax_err = float(np.nanpercentile(err_s, 95))

        # Predicted DSM
        im_p = axes2[0, col_idx].imshow(pred_s, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
        axes2[0, col_idx].scatter(c_pts / sub, r_pts / sub, c="red", s=45, marker="x", label=f"GCPs (K={k_val})")
        axes2[0, col_idx].set_title(f"Predicted DSM (K={k_val} GCPs)\nMAE: {mae_k:.2f}m, R²: {r2_k:.2f}", fontsize=11, fontweight="bold")
        axes2[0, col_idx].axis("off")
        axes2[0, col_idx].legend(loc="upper right", fontsize=9)
        cb_p = fig2.colorbar(im_p, ax=axes2[0, col_idx], fraction=0.046, pad=0.04)
        cb_p.set_label("Elevation (m)")

        # Absolute Error Map
        im_e = axes2[1, col_idx].imshow(err_s, cmap="magma", vmin=0, vmax=max(5.0, vmax_err))
        axes2[1, col_idx].set_title(f"Absolute Error Map (K={k_val})\nRMSE: {rmse_k:.2f}m", fontsize=11, fontweight="bold")
        axes2[1, col_idx].axis("off")
        cb_e = fig2.colorbar(im_e, ax=axes2[1, col_idx], fraction=0.046, pad=0.04)
        cb_e.set_label("Error (m)")

        col_idx += 1

    plt.suptitle("DepthWizard M2: Sparse Ground Control Point (GCP) Metric Calibration Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "figures", "gcp_comparison_maps.png")
    plt.savefig(comp_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to: {comp_fig_path}", flush=True)

    # Print Detailed Summary Table
    print("\n" + "=" * 105, flush=True)
    print(f"{'SPARSE GCP CALIBRATION SUMMARY TABLE (POTSDAM TILE ' + tile_id + ')':^105}", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Strategy':<20} | {'K':<4} | {'MAE (m)':<15} | {'RMSE (m)':<15} | {'Pearson r':<15} | {'R²':<15} | {'Span Ratio':<10}", flush=True)
    print("-" * 105, flush=True)
    for strat in strategies:
        for k in k_values:
            st = results_by_strategy[strat][str(k)]
            mae_str = f"{st['mae_mean']:.3f} ± {st['mae_std']:.3f}"
            rmse_str = f"{st['rmse_mean']:.3f} ± {st['rmse_std']:.3f}"
            r_str = f"{st['correlation_mean']:.3f} ± {st['correlation_std']:.3f}"
            r2_str = f"{st['r2_mean']:.3f} ± {st['r2_std']:.3f}"
            span_str = f"{st['span_ratio_mean']:.3f}"
            print(f"{strat:<20} | {k:<4d} | {mae_str:<15} | {rmse_str:<15} | {r_str:<15} | {r2_str:<15} | {span_str:<10}", flush=True)
        print("-" * 105, flush=True)


if __name__ == "__main__":
    run_gcp_experiment()
