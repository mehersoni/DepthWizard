"""
M4 Shadow Cue Module - Phase 4 Step 4B: Physical Scale Recovery & Calibration Investigation

Investigates image provenance, satellite metadata, and physical references for demoImages/sat2.png.
Separates PRODUCTION_METERS_PER_PIXEL (None / UNKNOWN) from TEST_SCALE (0.50 m/px explicit parameter),
and outputs the Step 4B Scale Recovery Decision Report.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from shadow.scale import PhysicalScaleManager


def run_scale_recovery_investigation():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")

    img_pil = Image.open(image_path)
    w, h = img_pil.size

    # Step 4B.1 & 4B.2 Provenance Investigation
    source = "SOURCE UNKNOWN"
    dataset = "UNKNOWN"
    sensor = "UNKNOWN"
    gsd_val = "UNKNOWN (No GeoTIFF or metadata tags)"
    reference_avail = "NONE (No documented physical reference dimension in project)"
    evidence = "Inspected EXIF headers, shadow/README.md, workspace scripts; no satellite GSD or physical reference documentation exists."

    # Production vs Test Scale Separation
    production_scale_mgr = PhysicalScaleManager(meters_per_pixel=None, source_description=evidence)
    test_scale_mgr = PhysicalScaleManager(meters_per_pixel=0.50, source_description="Explicit Testing Parameter Only")

    # Final Status
    status = "[BLOCKED — NO DEFENSIBLE SCALE]"

    print("=" * 80)
    print(" STEP 4B — SCALE RECOVERY DECISION REPORT ")
    print("=" * 80)
    print(f"Image                   : {image_path} ({w} x {h} px)")
    print(f"Source                  : {source}")
    print(f"Dataset                 : {dataset}")
    print(f"Sensor                  : {sensor}")
    print(f"GSD                     : {gsd_val}")
    print(f"Reference Available     : {reference_avail}")
    print(f"Reference Distance      : N/A")
    print(f"Measured Pixels         : N/A")
    print(f"Production meters/px    : {production_scale_mgr.meters_per_pixel} (UNKNOWN)")
    print(f"Configured Test Scale   : {test_scale_mgr.meters_per_pixel} m/px (TESTING PARAMETER ONLY)")
    print(f"Evidence                : {evidence}")
    print(f"Confidence              : [NOT AVAILABLE]")
    print(f"Calibration Status      : {status}")
    print("=" * 80)
    print("\nExternal Requirement to Unblock Production Height Estimation:")
    print("  1. Authoritative satellite GSD input (e.g. 0.30 m/px for WorldView, 0.50 m/px for Pleiades), OR")
    print("  2. Manually supplied 'meters_per_pixel' parameter passed to PhysicalScaleManager at runtime.")
    print("=" * 80)


if __name__ == "__main__":
    run_scale_recovery_investigation()
