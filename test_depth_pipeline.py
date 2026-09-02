"""
M1 Perception Module - Phase 1 Dense Depth Extraction Test Runner

Executes the complete Monocular Relative Depth pipeline on demoImages/sat2.png.
Validates input, runs Hugging Face inference, calculates tensor statistics,
exports raw math (.npy), exports M5 frontend bridge (.json), and outputs diagnostics.
"""

import os
import sys
import numpy as np

# Ensure root workspace directory is in sys.pat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth.preprocess import load_and_validate_image
from depth.models import DepthEstimator
from depth.export import export_to_m5_json
from depth.visualize import generate_diagnostic_plot
from depth.tiles import generate_overlapping_tiles, solve_global_alignment

def run_depth_pipeline():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(root_dir, "demoImages", "sat1.jpg")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 120)
    print(" M1 PERCEPTION MODULE - PHASE 1 DENSE DEPTH EXTRACTION REPORT (sat1.jpg) ")
    print("=" * 120)

    # 1. Preprocessing
    print(f"[Stage 1] Validating and loading image: {image_path}")
    result = load_and_validate_image(image_path)
    
    if result is None:
        print("[!] PIPELINE BLOCKED: Image validation failed.")
        sys.exit(1)
        
    image, metadata = result
    print(f"         Resolution: {image.width} x {image.height} pixels (RGB)")
    print(f"         Georeferenced: {metadata['is_georeferenced']}")

    # 2. Model Initialization
    print("\n[Stage 2] Initializing Depth Anything V2 Base...")
    estimator = DepthEstimator()
    if not estimator.initialize():
        print("[!] PIPELINE BLOCKED: Model initialization failed.")
        sys.exit(1)

    

   # 3. Inference with Tiling
    print("\n[Stage 3] Executing tiled neural depth inference...")
    
    tile_predictions = {}
    
    # 25% overlap on a 1024 tile
    for crop, box in generate_overlapping_tiles(image, tile_size=128, overlap_ratio=0.25):
        # Explicitly call our 2-view TTA mode
        inference_result = estimator.predict_with_confidence(crop, tta_mode="2-view")
        
        if inference_result is not None:
            # Unpack the new dictionary payload
            tile_predictions[box] = {
                "depth": inference_result["relative_depth"],
                "height": inference_result["relative_height"],
                "confidence": inference_result["quality_map"]
            }
        
    print(f"         Successfully processed {len(tile_predictions)} individual tiles.")

    # Prepare arrays for stitching
    depth_array = np.zeros((image.height, image.width), dtype=np.float32)
    height_array = np.zeros((image.height, image.width), dtype=np.float32)
    confidence_array = np.zeros((image.height, image.width), dtype=np.float32)
    
    print("         Stitching tiles and resolving global scale ambiguity...")

    # Stitch the relative depth (DSM)
    final_global_depth = solve_global_alignment(tile_predictions, image.width, image.height)
    depth_array = final_global_depth
    
    # Naive stitching for height and confidence maps
    for box, data in tile_predictions.items():
        x1, y1, x2, y2 = box
        height_array[y1:y2, x1:x2] = data["height"]
        confidence_array[y1:y2, x1:x2] = data["confidence"]

    # 4. Export Assets
    # 4. Export Assets
    print("\n[Stage 4] Exporting mathematical arrays and bridging assets...")
    
    # ---------------------------------------------------------
    # AUTHORITATIVE EXPORT (For M2 Scale & M3 Geospatial)
    # ---------------------------------------------------------
    npy_depth_path = os.path.join(output_dir, "canonical_depth_m1.npy")
    npy_height_path = os.path.join(output_dir, "canonical_height_m1.npy")
    npy_conf_path = os.path.join(output_dir, "canonical_quality_m1.npy")
    
    np.save(npy_depth_path, depth_array)
    np.save(npy_height_path, height_array)
    np.save(npy_conf_path, confidence_array)
    print("         -> Canonical float32 arrays exported successfully.")
    
    # ---------------------------------------------------------
    # LOSSY PREVIEW EXPORT (For M5 Visualization UI)
    # ---------------------------------------------------------
    json_path = os.path.join(output_dir, "terrain_data.json")
    
    # We pass the isolated 'height_array' (buildings only) to the visualizer!
    export_to_m5_json(height_array, json_path, grid_size=128)
    print("         -> M5 UI preview JSON exported successfully.")
    
    # Generate Plots
    generate_diagnostic_plot(image, depth_array, output_dir)

    # 5. Diagnostic Table
    print("\n" + "=" * 120)
    print(" STEP 1 — RELATIVE DEPTH TENSOR DIAGNOSTICS ")
    print("=" * 120)
    print(f"{'Array Shape':<18} | {'Dtype':<10} | {'Min Depth':<12} | {'Max Depth':<12} | {'Mean Depth':<12} | {'NaN Count':<10}")
    print("-" * 120)
    
    h, w = depth_array.shape
    print(
        f"{str((h, w)):<18} | "
        f"{str(depth_array.dtype):<10} | "
        f"{np.min(depth_array):<12.4f} | "
        f"{np.max(depth_array):<12.4f} | "
        f"{np.mean(depth_array):<12.4f} | "
        f"{int(np.isnan(depth_array).sum()):<10}"
    )
    print("=" * 120)
    
    print("\n[STEP 1.5 — FINAL PIPELINE STATUS]")
    print(f"  Raw Math Export (.npy) : PASS -> {npy_depth_path}")
    print(f"  M5 JSON Bridge (.json) : PASS -> {json_path}")
    print(f"  Diagnostic Plot (.png) : PASS -> output/m1_depth_diagnostics.png")
    print(f"  Metric Calibration     : PENDING (Awaiting M3 Scale Calibration integration)")
    print("=" * 120)

if __name__ == "__main__":
    run_depth_pipeline()