"""
M4 Shadow Cue Module - Phase 4 Step 5: Physical Shadow Length Conversion Test Runner

Tests pixel_to_physical_shadow_length() under both explicitly supplied test scale (0.50 m/px [TEST ONLY])
and uncalibrated production scale (meters_per_pixel = None).
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

from shadow.detector import detect_shadow_candidates
from shadow.cleaner import clean_candidate_mask
from shadow.geometry import extract_region_geometries, compute_object_shadow_pairing, compute_shadow_length_px
from shadow.confidence import rank_shadow_regions
from shadow.validate_base_tip import validate_shadow_base_tip
from shadow.height import pixel_to_physical_shadow_length


def run_physical_length_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 110)
    print(" STEP 5 — PHYSICAL SHADOW LENGTH CONVERSION TEST REPORT (sat2.png) ")
    print("=" * 110)

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Extract pipeline candidates & pairings
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

    # STEP 5.3 & 5.4 Test Scale Conversion (0.50 m/px explicit parameter)
    test_scale = 0.50
    test_records = []
    test_lengths_m = []
    px_lengths = []

    print("\n" + "=" * 110)
    print(" STEP 5.4 — DIAGNOSTIC TABLE: TEST PHYSICAL LENGTHS (meters_per_pixel = 0.50 m/px [TEST ONLY]) ")
    print("=" * 110)
    print(f"{'Pair ID':<8} | {'Shadow Length (px)':<18} | {'Test Scale (m/px)':<18} | {'TEST Shadow Length (m)':<25}")
    print("-" * 110)

    for rec in validated_candidates:
        cid = rec["candidate_id"]
        l_px = rec["l_px"]
        px_lengths.append(l_px)

        res_test = pixel_to_physical_shadow_length(l_px, meters_per_pixel=test_scale)
        l_m_test = res_test["physical_shadow_length_m"]
        test_lengths_m.append(l_m_test)

        test_records.append({
            "candidate_id": cid,
            "base": rec["base"],
            "tip": rec["tip"],
            "l_px": l_px,
            "l_m_test": l_m_test,
            "bounding_box": rec["bounding_box"],
            "oriented_bbox": rec["oriented_bbox"],
            "shadow_dir": rec["shadow_dir"]
        })

        print(f"#{cid:<7} | {l_px:<18.2f} | {test_scale:<18.2f} | {l_m_test:<25.2f} [TEST]")

    print("=" * 110)

    # STEP 5.6 Production-Scale Test (meters_per_pixel = None)
    print("\n[STEP 5.6 — PRODUCTION SCALE TEST (meters_per_pixel = None)]")
    prod_sample = pixel_to_physical_shadow_length(validated_candidates[0]["l_px"], meters_per_pixel=None)
    print(f"  - Input Pixel Length       : {validated_candidates[0]['l_px']:.2f} px")
    print(f"  - Production Scale         : {prod_sample['meters_per_pixel']} (None)")
    print(f"  - Physical Length Output   : {prod_sample['physical_shadow_length_m']} (UNAVAILABLE)")
    print(f"  - Status Message           : {prod_sample['status']}")
    print("=" * 110)

    # STEP 5.7 Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for rec in test_records:
        cid = rec["candidate_id"]
        bx, by = rec["base"]
        tx, ty = rec["tip"]
        l_px = rec["l_px"]
        l_m = rec["l_m_test"]
        x, y, bw, bh = rec["bounding_box"]

        # Contour
        cv.drawContours(overlay, [rec["oriented_bbox"]], 0, (0, 120, 255), 1)

        # Line BASE -> TIP
        cv.line(overlay, (int(round(bx)), int(round(by))), (int(round(tx)), int(round(ty))), (255, 255, 0), 2)

        # BASE Circle
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)

        # TIP Square
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        # Direction Arrow
        sdx, sdy = rec["shadow_dir"]
        arrow_end = (int(round(bx + sdx * 20)), int(round(by + sdy * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_end, (0, 255, 0), 2, tipLength=0.3)

        label = f"#{cid} {l_px:.1f}px -> {l_m:.1f}m[TEST]"
        cv.putText(overlay, label, (x, max(12, y - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)

    diag_header = (
        "STEP 5 PHYSICAL SHADOW LENGTH DIAGNOSTICS (sat2.png)\n"
        "TEST SCALE = 0.50 m/px [TEST ONLY] | PRODUCTION SCALE = NOT CALIBRATED\n"
        "[Red Circle=BASE, Cyan Square=TIP, Yellow Line=BASE-TIP Segment]"
    )
    plt.title(diag_header)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shadow_physical_length_diagnostics_sat2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {out_path}")

    # STEP 5.8 Summary Breakdown
    print("\n" + "=" * 110)
    print(" STEP 5 SUMMARY BREAKDOWN ")
    print("=" * 110)
    print(f"Number of Validated Pairs         : {len(validated_candidates)}")
    print(f"Pixel Length Range                : {np.min(px_lengths):.2f} px — {np.max(px_lengths):.2f} px")
    print(f"Test Scale                        : {test_scale:.2f} m/px [TEST ONLY]")
    print(f"Test Physical-Length Range        : {np.min(test_lengths_m):.2f} m — {np.max(test_lengths_m):.2f} m [TEST]")
    print(f"Production Scale                  : None (UNAVAILABLE)")
    print(f"Production Physical-Length Status : {prod_sample['status']}")
    print("=" * 110)


if __name__ == "__main__":
    run_physical_length_test()
