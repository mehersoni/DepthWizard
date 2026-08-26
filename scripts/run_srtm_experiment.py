import os
import sys
import time
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth.depth_model import estimate_depth, load_model
from calibration.srtm_anchor import fetch_srtm_dem, fit_srtm_anchor, apply_srtm_calibration, extract_terrain_candidates
from evaluation.metrics import calculate_metrics, compute_error_map


def run_srtm_experiment(
    tile_id="2_10",
    output_dir="outputs"
):
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dsm"), exist_ok=True)

    print("=" * 80, flush=True)
    print("DepthWizard M2 - Scene-Specific SRTM Coarse Terrain Anchoring Experiment", flush=True)
    print("=" * 80, flush=True)

    rgb_path = f"data/potsdam/2_Ortho_RGB/top_potsdam_{tile_id}_RGB.tif"
    parts = tile_id.split("_")
    dsm_str = f"{int(parts[0]):02d}_{int(parts[1]):02d}"
    ref_dsm_path = f"data/potsdam/1_DSM/dsm_potsdam_{dsm_str}.tif"

    # [1/6] Load RGB
    print(f"\n[1/6] Loading RGB image: {rgb_path}...", flush=True)
    with rasterio.open(rgb_path) as src_rgb:
        rgb = src_rgb.read().transpose(1, 2, 0)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb_meta = src_rgb.meta.copy()

    # [2/6] Depth Anything V2 relative depth
    print(f"[2/6] Running Depth Anything V2 inference on {rgb.shape[0]}x{rgb.shape[1]} image...", flush=True)
    model, proc, dev = load_model()
    t0 = time.time()
    depth = estimate_depth(rgb, model=model, processor=proc, device=dev)
    dt_depth = time.time() - t0
    print(f"      Relative depth estimated in {dt_depth:.2f}s (range: [{np.min(depth):.4f}, {np.max(depth):.4f}])", flush=True)

    # [3/6] Fetch & Align Coarse SRTM DEM
    print(f"\n[3/6] Retrieving & Aligning Coarse SRTM / Copernicus DEM to image grid...", flush=True)
    srtm_dem, srtm_meta = fetch_srtm_dem(rgb_path, zoom=14, cache_dir="data/dem_cache")
    print(f"      SRTM Source        : {srtm_meta['source']}", flush=True)
    print(f"      SRTM Native Res    : ~{srtm_meta['resolution_m']}m", flush=True)
    print(f"      SRTM Grid Alignment: {srtm_dem.shape} (CRS: {srtm_meta['crs']})", flush=True)
    print(f"      SRTM Elevation     : [{srtm_meta['min_elevation_m']:.2f}, {srtm_meta['max_elevation_m']:.2f}] m (Mean: {srtm_meta['mean_elevation_m']:.2f} m)", flush=True)

    # [4/6] Perform Scene-Specific SRTM Calibration (WITHOUT reference DSM)
    print(f"\n[4/6] Performing Scene-Specific SRTM Elevation Calibration...", flush=True)
    terrain_pct = 25.0
    scale_prior = 6.9373  # Learned scale prior from multi-scene cross-validation

    # Method 1: SRTM Ground Anchor with Scale Prior
    a_srtm, b_srtm, diag_srtm = fit_srtm_anchor(
        depth=depth,
        srtm_dem=srtm_dem,
        terrain_percentile=terrain_pct,
        scale_prior=scale_prior,
        method="robust_anchor"
    )
    pred_dsm_srtm = apply_srtm_calibration(depth, a_srtm, b_srtm)
    print(f"      [SRTM Anchored] Scale a = {a_srtm:.4f}, Offset b = {b_srtm:.4f} m", flush=True)
    print(f"      Terrain Anchors : {diag_srtm['anchor_pixels']} pixels ({diag_srtm['anchor_pixels']/depth.size*100:.1f}%)", flush=True)
    print(f"      Median Terrain D: {diag_srtm['median_terrain_relative_depth']:.4f} -> Median SRTM H: {diag_srtm['median_srtm_terrain_elevation_m']:.2f} m", flush=True)

    # Method 2: Fixed Generalization Calibration (Without scene-specific SRTM)
    # Using average multi-scene parameters learned from other tiles
    a_fixed = 6.9373
    b_fixed = 35.7876
    pred_dsm_fixed = apply_srtm_calibration(depth, a_fixed, b_fixed)

    # [5/6] Load Ground Truth LiDAR Reference DSM (STRICTLY FOR EVALUATION)
    print(f"\n[5/6] Loading Ground Truth LiDAR Reference DSM for evaluation...", flush=True)
    with rasterio.open(ref_dsm_path) as src_ref:
        ref_dsm = src_ref.read(1).astype(np.float32)
        ref_nodata = src_ref.nodata if src_ref.nodata is not None else -9999.0

    valid_mask = np.isfinite(ref_dsm) & (ref_dsm != ref_nodata) & (np.abs(ref_dsm) < 1e5)
    r_valid = ref_dsm[valid_mask]
    ref_min, ref_max = float(np.min(r_valid)), float(np.max(r_valid))
    ref_span = ref_max - ref_min

    # Calculate metrics
    m_srtm = calculate_metrics(pred_dsm_srtm, ref_dsm, valid_mask=valid_mask, nodata=ref_nodata)
    m_fixed = calculate_metrics(pred_dsm_fixed, ref_dsm, valid_mask=valid_mask, nodata=ref_nodata)
    m_raw_srtm = calculate_metrics(srtm_dem, ref_dsm, valid_mask=valid_mask, nodata=ref_nodata)

    p_valid_srtm = pred_dsm_srtm[valid_mask]
    pred_min, pred_max = float(np.min(p_valid_srtm)), float(np.max(p_valid_srtm))
    pred_span = pred_max - pred_min
    span_ratio = pred_span / ref_span

    # Error maps
    err_srtm = compute_error_map(pred_dsm_srtm, ref_dsm, valid_mask=valid_mask)

    # Save predicted GeoTIFF
    out_tif_path = os.path.join(output_dir, "dsm", "potsdam_srtm_predicted.tif")
    out_meta = rgb_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": -9999.0
    })
    with rasterio.open(out_tif_path, "w", **out_meta) as dst_tif:
        dst_tif.write(pred_dsm_srtm, 1)

    # [6/6] Save metrics JSON and Figure
    metrics_record = {
        "tile_id": tile_id,
        "srtm_source_info": {
            "source": srtm_meta["source"],
            "native_resolution_m": srtm_meta["resolution_m"],
            "crs": srtm_meta["crs"],
            "spatial_coverage": f"{ref_dsm.shape[0]}x{ref_dsm.shape[1]} pixels (300m x 300m)",
            "valid_srtm_pixels": srtm_meta["valid_pixels"],
            "srtm_elevation_range_m": [srtm_meta["min_elevation_m"], srtm_meta["max_elevation_m"]],
            "srtm_mean_elevation_m": srtm_meta["mean_elevation_m"]
        },
        "calibration_details": {
            "method": "Scene-Specific SRTM Ground Terrain Anchoring",
            "terrain_candidate_selection": "Lowest 25% relative depth values representing bare ground/roads/lawns",
            "anchor_pixels_used": diag_srtm["anchor_pixels"],
            "scale_a": a_srtm,
            "offset_b": b_srtm,
            "formula": f"H(x,y) = {a_srtm:.4f} * D(x,y) + {b_srtm:.4f}m"
        },
        "reference_dsm_stats": {
            "min_m": ref_min,
            "max_m": ref_max,
            "span_m": ref_span
        },
        "predicted_dsm_stats": {
            "min_m": pred_min,
            "max_m": pred_max,
            "span_m": pred_span,
            "span_compression_ratio": span_ratio
        },
        "evaluation_metrics": {
            "scene_specific_srtm_anchored": {
                "mae_m": m_srtm["mae"],
                "rmse_m": m_srtm["rmse"],
                "pearson_correlation": m_srtm["correlation"],
                "r2": m_srtm["r2"]
            },
            "fixed_heldout_calibration_baseline": {
                "mae_m": m_fixed["mae"],
                "rmse_m": m_fixed["rmse"],
                "pearson_correlation": m_fixed["correlation"],
                "r2": m_fixed["r2"]
            },
            "raw_srtm_dem_alone": {
                "mae_m": m_raw_srtm["mae"],
                "rmse_m": m_raw_srtm["rmse"],
                "pearson_correlation": m_raw_srtm["correlation"],
                "r2": m_raw_srtm["r2"]
            }
        },
        "improvement_over_fixed": {
            "mae_reduction_m": m_fixed["mae"] - m_srtm["mae"],
            "mae_reduction_percent": ((m_fixed["mae"] - m_srtm["mae"]) / m_fixed["mae"]) * 100.0,
            "r2_improvement": m_srtm["r2"] - m_fixed["r2"]
        }
    }

    json_path = os.path.join(output_dir, "metrics", "srtm_experiment.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(metrics_record, f_json, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}", flush=True)

    # 6-Panel Figure
    print(f"Generating 6-panel visualization figure...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=150)

    sub = 6
    rgb_s = rgb[::sub, ::sub]
    depth_s = depth[::sub, ::sub]
    srtm_s = srtm_dem[::sub, ::sub]
    pred_s = pred_dsm_srtm[::sub, ::sub]
    ref_s = ref_dsm[::sub, ::sub]
    err_s = err_srtm[::sub, ::sub]

    vmin_h = float(np.min(ref_s))
    vmax_h = float(np.max(ref_s))
    vmax_err = float(np.nanpercentile(err_s, 95))

    # 1. RGB
    axes[0, 0].imshow(rgb_s)
    axes[0, 0].set_title(f"1. Potsdam RGB (Tile {tile_id})", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Relative Depth
    im2 = axes[0, 1].imshow(depth_s, cmap="plasma")
    axes[0, 1].set_title("2. Monocular Relative Depth D(x,y)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    cb2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb2.set_label("Disparity (Agnostic)")

    # 3. SRTM Coarse Elevation
    im3 = axes[0, 2].imshow(srtm_s, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes[0, 2].set_title(f"3. SRTM Coarse DEM Anchor (Mean: {srtm_meta['mean_elevation_m']:.1f}m)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    cb3 = fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cb3.set_label("Elevation (m)")

    # 4. Predicted Metric DSM
    im4 = axes[1, 0].imshow(pred_s, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes[1, 0].set_title(f"4. SRTM-Anchored Metric DSM (MAE: {m_srtm['mae']:.2f}m, R²: {m_srtm['r2']:.2f})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    cb4 = fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb4.set_label("Elevation (m)")

    # 5. Ground Truth Reference DSM
    im5 = axes[1, 1].imshow(ref_s, cmap="terrain", vmin=vmin_h, vmax=vmax_h)
    axes[1, 1].set_title(f"5. LiDAR Reference DSM Ground Truth", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    cb5 = fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb5.set_label("Elevation (m)")

    # 6. Absolute Error Map
    im6 = axes[1, 2].imshow(err_s, cmap="magma", vmin=0, vmax=vmax_err)
    axes[1, 2].set_title(f"6. Absolute Error |H_pred - H_ref| (RMSE: {m_srtm['rmse']:.2f}m)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")
    cb6 = fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cb6.set_label("Error (m)")

    plt.suptitle(f"DepthWizard M2: Scene-Specific SRTM Coarse Terrain Anchoring on Potsdam Tile {tile_id}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "srtm_experiment.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {fig_path}", flush=True)

    # Print Summary Table
    print("\n" + "=" * 90, flush=True)
    print("SRTM ANCHORING EXPERIMENT RESULTS ON POTSDAM TILE " + tile_id, flush=True)
    print("=" * 90, flush=True)
    print(f"{'Method / Configuration':<38} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'Pearson r':<10} | {'R²':<8}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'A. Fixed Held-Out Calibration':<38} | {m_fixed['mae']:<9.4f} | {m_fixed['rmse']:<9.4f} | {m_fixed['correlation']:<10.4f} | {m_fixed['r2']:<8.4f}", flush=True)
    print(f"{'B. Scene-Specific SRTM Anchor':<38} | {m_srtm['mae']:<9.4f} | {m_srtm['rmse']:<9.4f} | {m_srtm['correlation']:<10.4f} | {m_srtm['r2']:<8.4f}", flush=True)
    print(f"{'C. Raw SRTM DEM Alone (No Monocular)':<38} | {m_raw_srtm['mae']:<9.4f} | {m_raw_srtm['rmse']:<9.4f} | {m_raw_srtm['correlation']:<10.4f} | {m_raw_srtm['r2']:<8.4f}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    run_srtm_experiment()
