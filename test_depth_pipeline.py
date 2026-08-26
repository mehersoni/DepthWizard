"""
M1 Perception Module - Phase 1 Dense Depth Extraction Test Runner

Executes the complete Monocular Relative Depth pipeline on demoImages/sat2.png.
Validates input, runs Hugging Face inference, calculates tensor statistics,
exports raw math (.npy), exports M5 frontend bridge (.json), and outputs diagnostics.
"""

import os
import sys
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth.preprocess import load_and_validate_image
from depth.models import DepthEstimator
from depth.export import export_to_m5_json
from depth.visualize import generate_diagnostic_plot

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
    image = load_and_validate_image(image_path)
    if image is None:
        print("[!] PIPELINE BLOCKED: Image validation failed.")
        sys.exit(1)
    
    print(f"         Resolution: {image.width} x {image.height} pixels (RGB)")

    # 2. Model Initialization
    print("\n[Stage 2] Initializing Depth Anything V2 Small...")
    estimator = DepthEstimator()
    if not estimator.initialize():
        print("[!] PIPELINE BLOCKED: Model initialization failed.")
        sys.exit(1)

    # 3. Inference
    print("\n[Stage 3] Executing neural depth inference...")
    depth_array = estimator.predict(image)
    if depth_array is None:
        print("[!] PIPELINE BLOCKED: Inference failed.")
        sys.exit(1)

    # 4. Export Assets
    print("\n[Stage 4] Exporting mathematical arrays and bridging assets...")
    
    # Save raw array for M3 Scale Calibration
    npy_path = os.path.join(output_dir, "m1_raw_depth.npy")
    np.save(npy_path, depth_array)
    
    # Save JSON bridge for M5 3D Viewer
    json_path = os.path.join(output_dir, "terrain_data.json")
    export_to_m5_json(depth_array, json_path, grid_size=128)
    
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
    print(f"  Raw Math Export (.npy) : PASS -> {npy_path}")
    print(f"  M5 JSON Bridge (.json) : PASS -> {json_path}")
    print(f"  Diagnostic Plot (.png) : PASS -> output/m1_depth_diagnostics.png")
    print(f"  Metric Calibration     : PENDING (Awaiting M3 Scale Calibration integration)")
    print("=" * 120)

if __name__ == "__main__":
    run_depth_pipeline()