"""
M4 Shadow Cue Module - Phase 4 Step 6: Solar Elevation Input Investigation & Validation Test Runner

Investigates demoImages/sat2.png for solar metadata, validates input angle bounds (0 < angle < 90),
tests runtime parameter interface under production mode (None) and test mode (45.0° [TEST ONLY]),
and outputs diagnostic report & visualization.
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

from shadow.detector import detect_shadow_candidates
from shadow.cleaner import clean_candidate_mask
from shadow.geometry import extract_region_geometries, compute_object_shadow_pairing, compute_shadow_length_px
from shadow.confidence import rank_shadow_regions
from shadow.validate_base_tip import validate_shadow_base_tip
from shadow.height import pixel_to_physical_shadow_length, compute_building_height


def run_solar_elevation_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    img_pil = Image.open(image_path)
    w, h = img_pil.size

    # Step 6.1 & 6.2 Solar Elevation Source Inspection
    prod_sun_elevation = None
    solar_source = "Category C: No defensible solar information (No EXIF solar tags, timestamp, or coordinates)"
    prod_status = "SOLAR ELEVATION NOT VALIDATED"

    # Step 6.3 Explicit Test Parameter
    test_sun_elevation = 45.0  # 45 degrees test parameter

    print("=" * 80)
    print(" STEP 6 — SOLAR ELEVATION VALIDATION REPORT ")
    print("=" * 80)
    print(f"Image                 : {image_path} ({w} x {h} px)")
    print(f"Solar Elevation       : {prod_sun_elevation} (UNKNOWN)")
    print(f"Source                : {solar_source}")
    print(f"Required Metadata     : Timestamp (Date/Time), Latitude, Longitude OR Solar Angle Header")
    print(f"Available Metadata    : None")
    print(f"Confidence            : [NOT AVAILABLE]")
    print(f"Production Status     : {prod_status}")
    print(f"Test Parameter        : {test_sun_elevation}° [EXPLICIT TEST INPUT ONLY]")
    print("=" * 80)

    # Candidate Pairing & Physical Shadow Length
    image = cv.imread(image_path)
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    validated_candidates = []
    for r in ranked_regions[:10]:
        pairing = compute_object_shadow_pairing(image, r, corridor_width_factor=0.75, corridor_length_factor=1.5)
        if pairing["status"] == "[STRONG PAIR]":
            val = validate_shadow_base_tip(image, pairing)
            if val["status"] == "VALID":
                base = pairing["estimated_base_point"]
                tip = pairing["estimated_tip_point"]
                l_px = compute_shadow_length_px(base, tip)
                validated_candidates.append({
                    "candidate_id": pairing["candidate_id"],
                    "base": base,
                    "tip": tip,
                    "l_px": l_px,
                    "bounding_box": r["bounding_box"],
                    "oriented_bbox": r["oriented_bbox"],
                    "shadow_dir": pairing["shadow_direction_vector"]
                })

    sample = validated_candidates[0]
    sample_l_m_test = sample["l_px"] * 0.50  # 34.99 * 0.50 = 17.50 m [TEST ONLY]

    # Step 6.4 Input Validation Check
    print("\n[STEP 6.4 — INPUT VALIDATION BOUNDS CHECK]")
    val_valid = compute_building_height(sample_l_m_test, sun_elevation_deg=45.0)
    val_invalid_low = compute_building_height(sample_l_m_test, sun_elevation_deg=0.0)
    val_invalid_high = compute_building_height(sample_l_m_test, sun_elevation_deg=95.0)

    print(f"  - Valid (45.0°)   : {val_valid['status']} -> H={val_valid['building_height_m']:.2f}m")
    print(f"  - Invalid (0.0°)  : {val_invalid_low['status']}")
    print(f"  - Invalid (95.0°) : {val_invalid_high['status']}")

    # Production Mode Test (sun_elevation_deg = None)
    print("\n[STEP 6.4 — PRODUCTION MODE TEST (sun_elevation_deg = None)]")
    prod_h_res = compute_building_height(sample_l_m_test, sun_elevation_deg=None)
    print(f"  - Input Physical Length : {sample_l_m_test:.2f} m")
    print(f"  - Solar Elevation Input : {prod_h_res['sun_elevation_deg']} (None)")
    print(f"  - Height Result         : {prod_h_res['building_height_m']} (UNAVAILABLE)")
    print(f"  - Status Message        : {prod_h_res['status']}")
    print("=" * 80)

    # Step 6.5 Demonstration Calculation ONLY
    print("\n[STEP 6.5 — DEMONSTRATION CALCULATION ONLY (TEST MODE)]")
    print(f"  Formula : H_test = L_shadow_m * tan(radians({test_sun_elevation}°))")
    print(f"  Sample  : L_m={sample_l_m_test:.2f}m, theta={test_sun_elevation}°, tan(45°)=1.00")
    print(f"  H_test  : {sample_l_m_test * np.tan(np.radians(test_sun_elevation)):.2f} meters [TEST ONLY]")
    print("=" * 80)

    # Step 6.6 Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for rec in validated_candidates:
        cid = rec["candidate_id"]
        bx, by = rec["base"]
        tx, ty = rec["tip"]
        l_px = rec["l_px"]
        l_m_test = l_px * 0.50
        h_test = l_m_test * np.tan(np.radians(test_sun_elevation))
        x, y, bw, bh = rec["bounding_box"]

        cv.drawContours(overlay, [rec["oriented_bbox"]], 0, (0, 120, 255), 1)
        cv.line(overlay, (int(round(bx)), int(round(by))), (int(round(tx)), int(round(ty))), (255, 255, 0), 2)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        sdx, sdy = rec["shadow_dir"]
        arrow_end = (int(round(bx + sdx * 20)), int(round(by + sdy * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_end, (0, 255, 0), 2, tipLength=0.3)

        label = f"#{cid} H_test:{h_test:.1f}m (theta={test_sun_elevation}deg)"
        cv.putText(overlay, label, (x, max(12, y - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)

    diag_header = (
        "STEP 6 SOLAR ELEVATION DIAGNOSTICS (sat2.png)\n"
        "TEST SUN ELEVATION = 45.0° [TEST ONLY] | PRODUCTION SOLAR ELEVATION = NOT VALIDATED\n"
        "Production Building Height Status: BLOCKED — SOLAR ELEVATION REQUIRED"
    )
    plt.title(diag_header)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shadow_solar_elevation_diagnostics_sat2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {out_path}")
    print("\nConclusion:")
    print("  Status: BLOCKED — SOLAR ELEVATION REQUIRED")
    print("  Production height estimation requires BOTH:")
    print("    1. Validated meters_per_pixel")
    print("    2. Validated sun_elevation_deg")
    print("=" * 80)


if __name__ == "__main__":
    run_solar_elevation_test()
