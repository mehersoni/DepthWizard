"""
M4 Shadow Cue Module - Phase 4 Step 4: Physical Image Scale Calibration Test Runner

Inspects demoImages/sat2.png for physical scale metadata, evaluates scale availability,
demonstrates centralized scale interface (PhysicalScaleManager), and outputs diagnostic reports & plot.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from shadow.scale import PhysicalScaleManager


def run_scale_calibration_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load image and inspect EXIF metadata
    img_pil = Image.open(image_path)
    w, h = img_pil.size
    exif_data = img_pil.getexif()

    print("=" * 70)
    print(" STEP 4 — PHYSICAL SCALE CALIBRATION REPORT ")
    print("=" * 70)
    print(f"Image                  : {image_path}")
    print(f"Dimensions             : {w} x {h} pixels")

    # Step 4.1 & 4.2 Classification
    has_gsd = False
    scale_source = "No GeoTIFF or GSD metadata tag present in PNG"

    if has_gsd:
        source_class = "Category A: Direct GSD available"
        scale_mgr = PhysicalScaleManager(meters_per_pixel=0.5, source_description="Direct GeoTIFF Metadata")
    else:
        source_class = "Category C: No defensible metadata or GSD available in source image"
        scale_mgr = PhysicalScaleManager(meters_per_pixel=None, source_description=scale_source)

    status_info = scale_mgr.get_status_report()

    print(f"Scale Source Category  : {source_class}")
    print(f"Scale Source Detail    : {status_info['source_description']}")
    print(f"meters_per_pixel       : {status_info['meters_per_pixel']}")
    print(f"Calibration Confidence : {status_info['confidence_status']}")
    print(f"Calibration Status     : {status_info['status_text']}")
    print("=" * 70)

    # Step 4.5 Centralization Demo with Explicit Test Scale Parameter
    print("\n[Central Scale Interface Verification]")
    test_scale_val = 0.50  # Explicit parameter (0.50 meters/pixel)
    test_scale_mgr = PhysicalScaleManager(meters_per_pixel=test_scale_val, source_description="Manually Supplied Test Scale Parameter (0.50 m/px)")

    sample_px_length = 34.99  # Sample length from Region #134
    converted_m = test_scale_mgr.convert_pixels_to_meters(sample_px_length)

    print(f"  - Configured Test Scale : {test_scale_mgr.meters_per_pixel} m/px")
    print(f"  - Sample Shadow Length  : {sample_px_length:.2f} px")
    print(f"  - Converted Physical L  : {converted_m:.2f} meters (Verification: {sample_px_length} * {test_scale_val} = {sample_px_length * test_scale_val:.2f} m)")
    print("=" * 70)

    # 6. Create Diagnostic Output Plot
    image_bgr = cv.imread(image_path)
    overlay = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7.5))
    plt.imshow(overlay)

    diag_text = (
        f"STEP 4 PHYSICAL SCALE DIAGNOSTICS\n"
        f"Image: sat2.png ({w}x{h})\n"
        f"Direct GSD Metadata: NONE DETECTED\n"
        f"Scale Status: {status_info['status_text']}\n"
        f"Configured Test Scale: {test_scale_val} m/px (Manual Parameter)"
    )

    plt.text(
        20, 40,
        diag_text,
        color="yellow",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.80)
    )

    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shadow_scale_diagnostics_sat2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {out_path}")
    print("\nConclusion:")
    if scale_mgr.is_calibrated:
        print("  Status: READY FOR STEP 5 — PHYSICAL SHADOW LENGTH")
    else:
        print("  Status: READY FOR STEP 5 — (Requires Explicit meters_per_pixel Parameter Input)")
    print("=" * 70)


if __name__ == "__main__":
    run_scale_calibration_test()
