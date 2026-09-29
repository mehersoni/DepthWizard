"""
DepthWizard - ISRO Cartosat Evaluation & Benchmarking Pipeline
Evaluates fine-tuned Depth Anything V2 on Cartosat satellite optical scenes
against reference CartoDEM, LiDAR, or spaceborne DEMs (SRTM / ALOS AW3D30).

Computes standard satellite elevation metrics:
  - Pearson Linear Correlation (r)
  - Mean Absolute Error (MAE in meters)
  - Root Mean Square Error (RMSE in meters)
  - LE90 (Linear Error at 90% Confidence, standard ISRO NRSC photogrammetry metric)
  - LE68 (Linear Error at 68% Confidence / 1-sigma)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_io.cartosat_adapter import load_cartosat_image
from depth.depth_model import load_model, estimate_depth
from depth.tiled_inference import estimate_depth_tiled
from calibration.rdsm import make_rdsm
from calibration.metric import fit_scale_offset, apply_scale_offset
from evaluation.metrics import calculate_metrics, compute_error_map

DEFAULT_CHECKPOINT = "checkpoints/depth_anything_v2_aerial"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DepthWizard on Cartosat satellite data")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to Cartosat image (.tif/.tiff), 16-bit or 8-bit, PAN or VNIR")
    parser.add_argument("--reference_dem", type=str, default=None,
                        help="Optional path to reference DEM (CartoDEM, LiDAR, SRTM, AW3D30)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to fine-tuned Depth Anything V2 checkpoint")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for large Cartosat scenes")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Tile overlap ratio")
    parser.add_argument("--output_dir", type=str, default="output/cartosat_evaluation",
                        help="Directory to save evaluation results and output rasters")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device: 'cuda', 'cpu', or auto-detect")
    return parser.parse_args()


def calculate_satellite_metrics(pred_dsm: np.ndarray, ref_dsm: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
    """
    Computes photogrammetric accuracy metrics including LE90 and LE68.
    """
    base_metrics = calculate_metrics(pred_dsm, ref_dsm, valid_mask=valid_mask)

    p = pred_dsm[valid_mask]
    r = ref_dsm[valid_mask]
    abs_errors = np.abs(p - r)

    # Standard USGS / ISRO LE90 and LE68
    le90 = float(np.percentile(abs_errors, 90))
    le68 = float(np.percentile(abs_errors, 68))

    base_metrics["le90"] = le90
    base_metrics["le68"] = le68
    return base_metrics


def evaluate_cartosat(args):
    print("=" * 70)
    print("DepthWizard - Cartosat Satellite Evaluation & Inference Pipeline")
    print(f"Input Scene:      {args.image}")
    print(f"Reference DEM:    {args.reference_dem or 'None (Zero-Shot / Relative mode)'}")
    print(f"Model Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # 1. Ingest Cartosat optical scene
    print("\n[1/4] Ingesting Cartosat Optical Scene...")
    carto = load_cartosat_image(args.image, stretch_dynamic_range=True)
    rgb = carto["rgb"]
    h, w, _ = rgb.shape

    print(f"  Dimensions: {w}x{h} px | Bands: {carto['band_count']} ({'Panchromatic' if carto['is_panchromatic'] else 'Multispectral'})")
    if carto["gsd_x"] and carto["gsd_y"]:
        print(f"  Ground Sampling Distance (GSD): {carto['gsd_x']:.2f}m x {carto['gsd_y']:.2f}m")
    if carto["solar_elevation"]:
        print(f"  Solar Elevation: {carto['solar_elevation']:.1f} deg | Azimuth: {carto['solar_azimuth']} deg")
    if carto["crs"]:
        print(f"  CRS: {carto['crs']}")

    # 2. Run Monocular Inference using Fine-Tuned Model
    print("\n[2/4] Running Depth Estimation with Fine-Tuned Model...")
    t0 = time.time()
    if max(h, w) > 768:
        print(f"  Large scene detected ({w}x{h}). Using blended tiled inference (tile_size={args.tile_size})...")
        rel_depth, _ = estimate_depth_tiled(
            rgb,
            tile_size=args.tile_size,
            overlap=args.overlap,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    else:
        model, proc, dev = load_model(checkpoint_path=args.checkpoint, device=args.device)
        rel_depth = estimate_depth(rgb, model=model, processor=proc, device=dev)
    inference_time = time.time() - t0
    print(f"  Inference completed in {inference_time:.2f}s ({inference_time * 1000 / (h*w/1e6):.1f} ms/Mpx)")

    # 3. Calibration against Reference DEM (if provided)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_summary = None
    calibrated_dsm = None

    if args.reference_dem and os.path.exists(args.reference_dem):
        print(f"\n[3/4] Aligning and Calibrating against Reference DEM: '{args.reference_dem}'...")
        with rasterio.open(args.reference_dem) as src_ref:
            ref_data = np.zeros((h, w), dtype=np.float32)
            if carto["crs"] and carto["transform"]:
                reproject(
                    source=rasterio.band(src_ref, 1),
                    destination=ref_data,
                    src_transform=src_ref.transform,
                    src_crs=src_ref.crs,
                    dst_transform=carto["transform"],
                    dst_crs=carto["crs"],
                    resampling=Resampling.bilinear
                )
            else:
                # Resize directly
                ref_raw = src_ref.read(1)
                from PIL import Image
                ref_resized = Image.fromarray(ref_raw).resize((w, h), Image.Resampling.BILINEAR)
                ref_data = np.array(ref_resized, dtype=np.float32)

            ref_nodata = src_ref.nodata if src_ref.nodata is not None else -9999.0

        scale_a, offset_b, valid_mask = fit_scale_offset(rel_depth, ref_data, nodata=ref_nodata)
        print(f"  Valid overlap pixels: {int(np.sum(valid_mask)):,} ({100 * np.sum(valid_mask)/(h*w):.1f}%)")

        if np.sum(valid_mask) > 10:
            calibrated_dsm = apply_scale_offset(rel_depth, scale_a, offset_b)

            metrics_summary = calculate_satellite_metrics(calibrated_dsm, ref_data, valid_mask=valid_mask)
            metrics_summary["scale_a"] = float(scale_a)
            metrics_summary["offset_b"] = float(offset_b)
            metrics_summary["inference_time_s"] = float(inference_time)

            print("\n" + "=" * 65)
            print("CARTOSAT ELEVATION BENCHMARK EVALUATION RESULTS")
            print("=" * 65)
            print(f"  Pearson Correlation (r):  {metrics_summary['correlation']:.4f}")
            print(f"  Mean Absolute Error (MAE): {metrics_summary['mae']:.2f} meters")
            print(f"  Root Mean Square (RMSE):  {metrics_summary['rmse']:.2f} meters")
            print(f"  LE90 (90% Confidence):    {metrics_summary['le90']:.2f} meters")
            print(f"  LE68 (68% Confidence / 1-sigma): {metrics_summary['le68']:.2f} meters")
            print(f"  Coefficient of Det. (R2): {metrics_summary['r2']:.4f}")
            print("=" * 65)
    else:
        print("\n[3/4] No reference DEM provided. Generating standardized rDSM [0, 1]...")
        calibrated_dsm = make_rdsm(rel_depth)

    # 4. Export Output Rasters
    print("\n[4/4] Exporting Output Rasters...")
    output_tif = os.path.join(args.output_dir, "cartosat_predicted_dsm.tif")

    if carto["crs"] and carto["transform"]:
        meta = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "float32",
            "crs": carto["crs"],
            "transform": carto["transform"],
            "nodata": -9999.0
        }
        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(calibrated_dsm.astype(np.float32), 1)
        print(f"  Exported Georeferenced GeoTIFF: '{output_tif}'")
    else:
        # Non-georeferenced fallback
        output_png = os.path.join(args.output_dir, "cartosat_predicted_dsm.png")
        from PIL import Image
        norm_vis = ((calibrated_dsm - calibrated_dsm.min()) / (calibrated_dsm.max() - calibrated_dsm.min() + 1e-6) * 255.0).astype(np.uint8)
        Image.fromarray(norm_vis).save(output_png)
        print(f"  Exported Grayscale Preview: '{output_png}'")

    if metrics_summary is not None:
        metrics_file = os.path.join(args.output_dir, "cartosat_benchmark_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  Exported Benchmark Metrics: '{metrics_file}'")

    print("\nCartosat Evaluation Complete!")
    return metrics_summary


if __name__ == "__main__":
    args = parse_args()
    evaluate_cartosat(args)
