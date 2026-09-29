import os
import sys
import time
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth.depth_model import estimate_depth, load_model
from calibration.gcp_calibration import (
    sample_grid_stratified_gcps,
    fit_gcp_calibration,
    apply_gcp_calibration
)
from shadow_detection import (
    compute_shadow_confidence,
    filter_shadow_mask,
    detect_building_shadow_pairs,
    apply_shadow_height_constraint
)
from evaluation.metrics import calculate_metrics, compute_error_map


def run_shadow_experiment(
    tile_id="2_10",
    output_dir="outputs"
):
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dsm"), exist_ok=True)

    print("=" * 85, flush=True)
    print("DepthWizard M2 - Shadow-Constrained Building Elevation Experiment", flush=True)
    print("=" * 85, flush=True)

    rgb_path = f"data/potsdam/2_Ortho_RGB/top_potsdam_{tile_id}_RGB.tif"
    parts = tile_id.split("_")
    dsm_str = f"{int(parts[0]):02d}_{int(parts[1]):02d}"
    ref_dsm_path = f"data/potsdam/1_DSM/dsm_potsdam_{dsm_str}.tif"

    # [1/6] Load RGB & Reference DSM
    print(f"\n[1/6] Loading RGB ({rgb_path}) and Reference DSM ({ref_dsm_path})...", flush=True)
    with rasterio.open(rgb_path) as src_rgb:
        rgb = src_rgb.read().transpose(1, 2, 0)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb_meta = src_rgb.meta
        rgb_tags = src_rgb.tags()

    with rasterio.open(ref_dsm_path) as src_ref:
        ref_dsm = src_ref.read(1).astype(np.float32)
        nodata = src_ref.nodata if src_ref.nodata is not None else -9999.0

    valid_mask = np.isfinite(ref_dsm) & (ref_dsm != nodata) & (np.abs(ref_dsm) < 1e5)
    r_valid = ref_dsm[valid_mask]
    ref_min = float(np.min(r_valid))
    ref_max = float(np.max(r_valid))
    ref_span = ref_max - ref_min
    ref_ground = float(np.percentile(r_valid, 10.0))
    print(f"      Reference DSM: [{ref_min:.2f}, {ref_max:.2f}] m (Span: {ref_span:.2f} m, Ground Datum: ~{ref_ground:.2f} m)", flush=True)

    # Check Solar Metadata Tags
    print(f"\n[2/6] Inspecting Solar Geometry Metadata in GeoTIFF tags...", flush=True)
    print(f"      GeoTIFF Tags: {rgb_tags}")
    solar_elevation = None
    solar_azimuth = None
    if "SOLAR_ELEVATION" in rgb_tags:
        solar_elevation = float(rgb_tags["SOLAR_ELEVATION"])
    if "SOLAR_AZIMUTH" in rgb_tags:
        solar_azimuth = float(rgb_tags["SOLAR_AZIMUTH"])

    if solar_elevation is None:
        print("      [RESULT] Solar elevation/azimuth tags NOT found in Potsdam orthophoto.", flush=True)
        print("      [DECISION] Operating strictly under MODE B (Structural / edge confidence only).", flush=True)
        print("      [SCIENTIFIC RULE] Solar geometry is NOT fabricated or invented.", flush=True)
    else:
        print(f"      [RESULT] Solar elevation found: {solar_elevation:.2f}°, Azimuth: {solar_azimuth:.2f}° (MODE A)", flush=True)

    # [3/6] Monocular Depth Inference & Initial GCP Calibration
    print(f"\n[3/6] Running Depth Anything V2 Inference...", flush=True)
    model, proc, dev = load_model()
    t0 = time.time()
    depth = estimate_depth(rgb, model=model, processor=proc, device=dev)
    print(f"      Depth estimated in {time.time()-t0:.2f}s (Range: [{np.min(depth):.4f}, {np.max(depth):.4f}])", flush=True)

    # Fit 5 GCP Calibration (Grid-Stratified, Seed 42)
    print(f"      Fitting 5 GCP Grid-Stratified Calibration...", flush=True)
    gcp_rows, gcp_cols = sample_grid_stratified_gcps(valid_mask, k=5, seed=42)
    gcp_d = depth[gcp_rows, gcp_cols]
    gcp_h = ref_dsm[gcp_rows, gcp_cols]
    scale_a, offset_b = fit_gcp_calibration(gcp_d, gcp_h, scale_prior=6.9373)
    initial_dsm = apply_gcp_calibration(depth, scale_a, offset_b)
    print(f"      Initial Metric DSM: H = {scale_a:.4f} * D + {offset_b:.4f} m (Range: [{np.min(initial_dsm):.2f}, {np.max(initial_dsm):.2f}] m)", flush=True)

    # [4/6] Shadow Detection & Shadow Geometry
    print(f"\n[4/6] Running Shadow Detection on RGB...", flush=True)
    t_sh = time.time()
    shadow_conf = compute_shadow_confidence(rgb)
    shadow_mask = filter_shadow_mask(shadow_conf, confidence_threshold=0.35, min_area_pixels=50)
    shadow_px = int(np.sum(shadow_mask))
    total_px = shadow_mask.size
    shadow_coverage_pct = (shadow_px / total_px) * 100.0
    print(f"      Shadow Detection completed in {time.time()-t_sh:.2f}s", flush=True)
    print(f"      Shadow Pixels: {shadow_px:,} ({shadow_coverage_pct:.2f}% coverage)", flush=True)

    # Detect building-shadow pairs
    print(f"      Extracting Building-Shadow Pairs...", flush=True)
    pair_data = detect_building_shadow_pairs(
        rgb=rgb,
        shadow_mask=shadow_mask,
        initial_dsm=initial_dsm,
        gsd_m=0.05,
        min_shadow_len_px=15,
        max_shadow_len_px=500
    )
    print(f"      Raw Shadow Candidates: {pair_data['num_shadow_candidates']:,}", flush=True)
    print(f"      Reliable Accepted Pairs: {pair_data['num_accepted_pairs']:,}", flush=True)
    print(f"      Mean Shadow Length: {pair_data['mean_shadow_length_m']:.2f} m (Median: {pair_data['median_shadow_length_m']:.2f} m)", flush=True)

    # Apply Shadow Height Constraint
    constrained_dsm, shadow_summary = apply_shadow_height_constraint(
        initial_dsm=initial_dsm,
        shadow_mask=shadow_mask,
        building_shadow_data=pair_data,
        solar_elevation_deg=solar_elevation,
        solar_azimuth_deg=solar_azimuth
    )

    # [5/6] Metrics Calculation & Comparison
    print(f"\n[5/6] Evaluating Metrics across Configurations...", flush=True)
    
    # Subsampled evaluation for fast accurate metrics
    sub = 4  # 1500x1500 evaluation grid
    mask_eval = valid_mask[::sub, ::sub]
    ref_eval = ref_dsm[::sub, ::sub]
    d_eval = depth[::sub, ::sub]
    
    # Config A: Depth Only (Prior Scale a=6.94, Offset b=37.60m)
    dsm_a = (6.9373 * d_eval + 37.60).astype(np.float32)
    metrics_a = calculate_metrics(dsm_a, ref_eval, valid_mask=mask_eval)

    # Config B: Depth + 5 GCPs
    dsm_b = initial_dsm[::sub, ::sub]
    metrics_b = calculate_metrics(dsm_b, ref_eval, valid_mask=mask_eval)

    # Config C: Depth + 5 GCPs + Shadow Constraint (Mode B)
    dsm_c = constrained_dsm[::sub, ::sub]
    metrics_c = calculate_metrics(dsm_c, ref_eval, valid_mask=mask_eval)

    # Building-Focused Metrics (Reference DSM >= 48.0m, elevated structures)
    bldg_mask_eval = mask_eval & (ref_eval >= 48.0)
    bldg_count = int(np.sum(bldg_mask_eval))
    print(f"      Building Evaluation Pixels: {bldg_count:,} (>= 48.0m elevation)", flush=True)

    bldg_metrics_a = calculate_metrics(dsm_a, ref_eval, valid_mask=bldg_mask_eval)
    bldg_metrics_b = calculate_metrics(dsm_b, ref_eval, valid_mask=bldg_mask_eval)
    bldg_metrics_c = calculate_metrics(dsm_c, ref_eval, valid_mask=bldg_mask_eval)

    print(f"\n--- GLOBAL METRICS ---")
    print(f"   A. Depth Only (Prior)        | MAE: {metrics_a['mae']:.4f} m | RMSE: {metrics_a['rmse']:.4f} m | r: {metrics_a['correlation']:.4f} | R²: {metrics_a['r2']:.4f}")
    print(f"   B. Depth + 5 GCPs            | MAE: {metrics_b['mae']:.4f} m | RMSE: {metrics_b['rmse']:.4f} m | r: {metrics_b['correlation']:.4f} | R²: {metrics_b['r2']:.4f}")
    print(f"   C. Depth + 5 GCPs + Shadow   | MAE: {metrics_c['mae']:.4f} m | RMSE: {metrics_c['rmse']:.4f} m | r: {metrics_c['correlation']:.4f} | R²: {metrics_c['r2']:.4f}")

    print(f"\n--- BUILDING-FOCUSED METRICS (Ref >= 48m) ---")
    print(f"   A. Depth Only (Prior)        | MAE: {bldg_metrics_a['mae']:.4f} m | RMSE: {bldg_metrics_a['rmse']:.4f} m | r: {bldg_metrics_a['correlation']:.4f} | R²: {bldg_metrics_a['r2']:.4f}")
    print(f"   B. Depth + 5 GCPs            | MAE: {bldg_metrics_b['mae']:.4f} m | RMSE: {bldg_metrics_b['rmse']:.4f} m | r: {bldg_metrics_b['correlation']:.4f} | R²: {bldg_metrics_b['r2']:.4f}")
    print(f"   C. Depth + 5 GCPs + Shadow   | MAE: {bldg_metrics_c['mae']:.4f} m | RMSE: {bldg_metrics_c['rmse']:.4f} m | r: {bldg_metrics_c['correlation']:.4f} | R²: {bldg_metrics_c['r2']:.4f}")

    # [6/6] Save JSON & Visualizations
    print(f"\n[6/6] Generating Output Figures and JSON...", flush=True)

    json_record = {
        "tile_id": tile_id,
        "solar_metadata_available": (solar_elevation is not None),
        "mode": shadow_summary["mode"],
        "reason": shadow_summary.get("reason", "Solar elevation metadata available"),
        "shadow_statistics": {
            "shadow_coverage_pct": shadow_coverage_pct,
            "num_shadow_candidates": pair_data["num_shadow_candidates"],
            "num_accepted_pairs": pair_data["num_accepted_pairs"],
            "mean_shadow_length_m": pair_data["mean_shadow_length_m"],
            "median_shadow_length_m": pair_data["median_shadow_length_m"],
            "num_shadow_height_constraints": shadow_summary["num_height_constraints"],
            "mean_shadow_height_m": shadow_summary["mean_shadow_height_m"]
        },
        "gcp_calibration": {
            "k": 5,
            "strategy": "grid_stratified",
            "scale_a": scale_a,
            "offset_b": offset_b
        },
        "global_metrics": {
            "depth_only": metrics_a,
            "depth_5_gcp": metrics_b,
            "depth_5_gcp_shadow": metrics_c
        },
        "building_metrics": {
            "depth_only": bldg_metrics_a,
            "depth_5_gcp": bldg_metrics_b,
            "depth_5_gcp_shadow": bldg_metrics_c
        }
    }

    json_path = os.path.join(output_dir, "metrics", "shadow_experiment.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(json_record, fj, indent=2)
    print(f"      Saved JSON to: {json_path}", flush=True)

    # Figure 1: Shadow Detection Breakdown (6 Subplots)
    print("      Generating Figure 1: shadow_detection.png...", flush=True)
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 13), dpi=150)
    sub_vis = 6
    rgb_vis = rgb[::sub_vis, ::sub_vis]
    conf_vis = shadow_conf[::sub_vis, ::sub_vis]
    mask_vis = shadow_mask[::sub_vis, ::sub_vis]

    # 1. RGB
    axes1[0, 0].imshow(rgb_vis)
    axes1[0, 0].set_title(f"1. Potsdam RGB Orthophoto (Tile {tile_id})", fontsize=11, fontweight="bold")
    axes1[0, 0].axis("off")

    # 2. Continuous Shadow Confidence
    im_c = axes1[0, 1].imshow(conf_vis, cmap="inferno", vmin=0, vmax=1.0)
    axes1[0, 1].set_title("2. Continuous Shadow Confidence S(x,y) ∈ [0, 1]", fontsize=11, fontweight="bold")
    axes1[0, 1].axis("off")
    cb1 = fig1.colorbar(im_c, ax=axes1[0, 1], fraction=0.046, pad=0.04)
    cb1.set_label("Confidence")

    # 3. Filtered Binary Shadow Mask
    axes1[0, 2].imshow(rgb_vis)
    axes1[0, 2].imshow(mask_vis, cmap="cool", alpha=0.5)
    axes1[0, 2].set_title(f"3. Morphologically Filtered Shadow Mask\n({shadow_coverage_pct:.2f}% Area Coverage)", fontsize=11, fontweight="bold")
    axes1[0, 2].axis("off")

    # 4. Building-Shadow Overlay with Vectors
    axes1[1, 0].imshow(rgb_vis)
    # Plot sample of accepted shadow bounding boxes/centroids
    for p in pair_data["pairs"][:60]:
        r_start, r_stop, c_start, c_stop = p["bbox"]
        c_ctr = (c_start + c_stop) / (2.0 * sub_vis)
        r_ctr = (r_start + r_stop) / (2.0 * sub_vis)
        axes1[1, 0].scatter(c_ctr, r_ctr, color="yellow", s=15, marker="o")
    axes1[1, 0].set_title(f"4. Detected Cast Shadow Centroids ({pair_data['num_accepted_pairs']} Accepted)", fontsize=11, fontweight="bold")
    axes1[1, 0].axis("off")

    # 5. Shadow Length Distribution
    lens = pair_data["shadow_lengths_m"]
    if len(lens) > 0:
        axes1[1, 1].hist(lens, bins=30, color="#2980b9", edgecolor="black", alpha=0.8)
        axes1[1, 1].axvline(np.mean(lens), color="red", linestyle="--", label=f"Mean: {np.mean(lens):.2f}m")
        axes1[1, 1].axvline(np.median(lens), color="green", linestyle=":", label=f"Median: {np.median(lens):.2f}m")
    axes1[1, 1].set_title("5. Measured Shadow Length Distribution (Ground Meters)", fontsize=11, fontweight="bold")
    axes1[1, 1].set_xlabel("Shadow Length (m)")
    axes1[1, 1].set_ylabel("Count")
    axes1[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes1[1, 1].legend()

    # 6. Shadow Direction Distribution
    dirs = pair_data["shadow_directions_deg"]
    if len(dirs) > 0:
        axes1[1, 2].hist(dirs, bins=36, color="#27ae60", edgecolor="black", alpha=0.8)
        axes1[1, 2].set_title("6. Shadow Principal Orientation Distribution", fontsize=11, fontweight="bold")
        axes1[1, 2].set_xlabel("Orientation (degrees)")
        axes1[1, 2].set_ylabel("Count")
        axes1[1, 2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("DepthWizard M2: RGB Shadow Detection & Geometric Characterization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "figures", "shadow_detection.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved detection figure to: {fig1_path}", flush=True)

    # Figure 2: Shadow Comparison (6 Subplots)
    print("      Generating Figure 2: shadow_comparison.png...", flush=True)
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 13), dpi=150)
    
    ref_vis = ref_dsm[::sub_vis, ::sub_vis]
    init_vis = initial_dsm[::sub_vis, ::sub_vis]
    const_vis = constrained_dsm[::sub_vis, ::sub_vis]
    err_vis = compute_error_map(constrained_dsm, ref_dsm, valid_mask=valid_mask)[::sub_vis, ::sub_vis]

    vmin_h, vmax_h = float(np.min(ref_vis)), float(np.max(ref_vis))

    # 1. RGB
    axes2[0, 0].imshow(rgb_vis)
    axes2[0, 0].set_title(f"1. Potsdam RGB Orthophoto (Tile {tile_id})", fontsize=11, fontweight="bold")
    axes2[0, 0].axis("off")

    # 2. Shadow Confidence
    im_sc = axes2[0, 1].imshow(conf_vis, cmap="inferno", vmin=0, vmax=1.0)
    axes2[0, 1].set_title(f"2. Shadow Confidence Map S(x,y)\n({shadow_coverage_pct:.2f}% Shadow Coverage)", fontsize=11, fontweight="bold")
    axes2[0, 1].axis("off")
    cb_sc = fig2.colorbar(im_sc, ax=axes2[0, 1], fraction=0.046, pad=0.04)
    cb_sc.set_label("Confidence")

    # 3. Initial Metric DSM (5 GCPs)
    im_init = axes2[0, 2].imshow(init_vis, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes2[0, 2].scatter(gcp_cols / sub_vis, gcp_rows / sub_vis, c="red", s=50, marker="x", label="5 GCP Anchors")
    axes2[0, 2].set_title(f"3. Initial Metric DSM (5 GCPs)\nMAE: {metrics_b['mae']:.2f}m, R²: {metrics_b['r2']:.2f}", fontsize=11, fontweight="bold")
    axes2[0, 2].axis("off")
    axes2[0, 2].legend(loc="upper right", fontsize=9)
    cb_init = fig2.colorbar(im_init, ax=axes2[0, 2], fraction=0.046, pad=0.04)
    cb_init.set_label("Elevation (m)")

    # 4. Shadow-Constrained DSM (Mode B)
    im_con = axes2[1, 0].imshow(const_vis, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes2[1, 0].set_title(f"4. Shadow-Constrained DSM (Mode B)\nMAE: {metrics_c['mae']:.2f}m, R²: {metrics_c['r2']:.2f}", fontsize=11, fontweight="bold")
    axes2[1, 0].axis("off")
    cb_con = fig2.colorbar(im_con, ax=axes2[1, 0], fraction=0.046, pad=0.04)
    cb_con.set_label("Elevation (m)")

    # 5. LiDAR Reference DSM Ground Truth
    im_ref = axes2[1, 1].imshow(ref_vis, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes2[1, 1].set_title("5. LiDAR Reference DSM Ground Truth\n(ISPRS Potsdam Tile 2_10)", fontsize=11, fontweight="bold")
    axes2[1, 1].axis("off")
    cb_ref = fig2.colorbar(im_ref, ax=axes2[1, 1], fraction=0.046, pad=0.04)
    cb_ref.set_label("Elevation (m)")

    # 6. Absolute Error Map
    vmax_err = float(np.nanpercentile(err_vis, 95))
    im_err = axes2[1, 2].imshow(err_vis, cmap="magma", vmin=0, vmax=max(5.0, vmax_err))
    axes2[1, 2].set_title(f"6. Absolute Error Map\nRMSE: {metrics_c['rmse']:.2f}m, Max: {np.nanmax(err_vis):.2f}m", fontsize=11, fontweight="bold")
    axes2[1, 2].axis("off")
    cb_err = fig2.colorbar(im_err, ax=axes2[1, 2], fraction=0.046, pad=0.04)
    cb_err.set_label("Absolute Error (m)")

    plt.suptitle("DepthWizard M2: Initial vs Shadow-Constrained Metric DSM Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "figures", "shadow_comparison.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved comparison figure to: {fig2_path}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("SHADOW EXPERIMENT COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    run_shadow_experiment()
