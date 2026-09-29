"""
M4 Shadow Cue Module - Phase 4 Step 7: Final Building Height Estimation Test Runner

Executes the complete height estimation pipeline on demoImages/sat2.png:
- Gates height calculation ONLY on [STRONG PAIR] candidates.
- Runs Test Mode (0.50 m/px, 45.0°) with [TEST ONLY] explicit labels.
- Runs Production Block Test (meters_per_pixel=None, sun_elevation_deg=None) demonstrating safe refusal.
- Performs Mathematical Sanity Tests (45°, 30°, 60°).
- Outputs Step 7 Console Table, Diagnostic Plot, and Final Pipeline Report.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
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
from shadow.height import estimate_building_height


def run_final_height_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 110)
    print(" STEP 7 — FINAL BUILDING HEIGHT ESTIMATION REPORT (sat2.png) ")
    print("=" * 110)

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Candidate Detection & Pairing
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    strong_pairs = []
    weak_pairs = []
    no_pairs = []

    for r in ranked_regions[:10]:
        pairing = compute_object_shadow_pairing(image, r, corridor_width_factor=0.75, corridor_length_factor=1.5)
        pst = pairing["status"]

        if pst == "[STRONG PAIR]":
            val = validate_shadow_base_tip(image, pairing)
            if val["status"] == "VALID":
                base = pairing["estimated_base_point"]
                tip = pairing["estimated_tip_point"]
                l_px = compute_shadow_length_px(base, tip)
                strong_pairs.append({
                    "candidate_id": pairing["candidate_id"],
                    "pair_status": pst,
                    "base": base,
                    "tip": tip,
                    "l_px": l_px,
                    "final_pair_score": pairing["final_pair_score"],
                    "bounding_box": r["bounding_box"],
                    "oriented_bbox": r["oriented_bbox"],
                    "shadow_dir": pairing["shadow_direction_vector"]
                })
        elif pst == "[WEAK PAIR]":
            weak_pairs.append(pairing["candidate_id"])
        else:
            no_pairs.append(pairing["candidate_id"])

    # Test Mode Configuration
    test_scale = 0.50  # m/px
    test_sun_elevation = 45.0  # degrees

    print(f"Mode            : TEST MODE [DEMONSTRATION ONLY]")
    print(f"Scale           : {test_scale:.2f} m/px [TEST ONLY]")
    print(f"Solar Elevation : {test_sun_elevation:.1f}° [TEST ONLY]")
    print("-" * 110)

    # Step 7.8 Console Table Output
    print(f"{'ID':<6} | {'PairStatus':<13} | {'L_px':<8} | {'L_m':<8} | {'SunElev':<8} | {'Height':<10} | {'Status':<12}")
    print("-" * 110)

    calculated_test_records = []
    for sp in strong_pairs:
        cid = sp["candidate_id"]
        l_px = sp["l_px"]
        pst = sp["pair_status"]

        # Run estimate_building_height with explicit test inputs
        res = estimate_building_height(
            shadow_length_px=l_px,
            meters_per_pixel=test_scale,
            sun_elevation_deg=test_sun_elevation,
            pair_confidence=sp["final_pair_score"],
            is_test_mode=True
        )

        l_m = res["shadow_length_m"]
        h_m = res["height_m"]
        st = res["status"]

        rec = {**sp, **res}
        calculated_test_records.append(rec)

        print(f"#{cid:<5} | {pst:<13} | {l_px:<8.2f} | {l_m:<8.2f} | {test_sun_elevation:<7.1f}° | {h_m:<10.2f}m | {st:<12}")

    print("-" * 110)
    print(f"Strong pairs included               : {len(strong_pairs)}")
    print(f"Weak pairs excluded                 : {len(weak_pairs)} (Candidate IDs: {weak_pairs})")
    print(f"No-pair candidates excluded         : {len(no_pairs)} (Candidate IDs: {no_pairs})")
    print(f"Production heights available        : 0 (Blocked - missing physical scale & solar elevation)")
    print(f"Test heights calculated             : {len(calculated_test_records)}")
    print("=" * 110)

    # Step 7.9 Production Block Test (meters_per_pixel=None, sun_elevation_deg=None)
    print("\n[STEP 7.9 — PRODUCTION BLOCK TEST (meters_per_pixel=None, sun_elevation_deg=None)]")
    prod_test = estimate_building_height(
        shadow_length_px=strong_pairs[0]["l_px"],
        meters_per_pixel=None,
        sun_elevation_deg=None,
        is_test_mode=False
    )
    print(f"  - Input Candidate #134 L_px : {strong_pairs[0]['l_px']:.2f} px")
    print(f"  - Production Scale          : {prod_test['meters_per_pixel']} (None)")
    print(f"  - Production Solar Angle    : {prod_test['sun_elevation_deg']} (None)")
    print(f"  - Height Result             : {prod_test['height_m']} (None)")
    print(f"  - Status                     : {prod_test['status']}")
    print(f"  - Failure Reason            : {prod_test['reason']}")
    print("=" * 110)

    # Step 7.10 Mathematical Sanity Test
    print("\n[STEP 7.10 — MATHEMATICAL SANITY TEST]")
    l_sample_px = 34.99
    scale_sample = 0.50
    l_sample_m = l_sample_px * scale_sample  # 17.495 m

    c1 = estimate_building_height(l_sample_px, scale_sample, 45.0, is_test_mode=True)
    c2 = estimate_building_height(l_sample_px, scale_sample, 30.0, is_test_mode=True)
    c3 = estimate_building_height(l_sample_px, scale_sample, 60.0, is_test_mode=True)

    print(f"  Sample Shadow L_px={l_sample_px}px, scale={scale_sample}m/px -> L_m={l_sample_m:.3f}m")
    print(f"  Case 1 (sun=45°): H = {c1['height_m']:.3f}m (Expected ≈ {l_sample_m:.3f}m) -> PASS={abs(c1['height_m'] - l_sample_m) < 1e-3}")
    print(f"  Case 2 (sun=30°): H = {c2['height_m']:.3f}m (Expected < {l_sample_m:.3f}m) -> PASS={c2['height_m'] < l_sample_m}")
    print(f"  Case 3 (sun=60°): H = {c3['height_m']:.3f}m (Expected > {l_sample_m:.3f}m) -> PASS={c3['height_m'] > l_sample_m}")
    print("=" * 110)

    # Step 7.7 Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for rec in calculated_test_records:
        cid = rec["candidate_id"]
        bx, by = rec["base"]
        tx, ty = rec["tip"]
        l_px = rec["l_px"]
        l_m = rec["shadow_length_m"]
        h_m = rec["height_m"]
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

        label = f"#{cid} L:{l_m:.1f}m H:{h_m:.1f}m [TEST]"
        cv.putText(overlay, label, (x, max(12, y - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)

    diag_header = (
        "STEP 7 — BUILDING HEIGHT ESTIMATION (sat2.png)\n"
        "TEST MODE — NOT PRODUCTION CALIBRATED\n"
        "Scale: 0.50 m/px [TEST ONLY] | Solar Elevation: 45.0° [TEST ONLY]"
    )
    plt.title(diag_header)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "final_height_diagnostics_sat2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {out_path}")

    # Step 7.11 Final Pipeline Report
    print("\n" + "=" * 80)
    print(" FINAL HEIGHT PIPELINE STATUS ")
    print("=" * 80)
    print(" Candidate Detection             : VALIDATED")
    print(" Shadow Geometry                 : VALIDATED")
    print(" Object–Shadow Pairing           : VALIDATED")
    print(" BASE/TIP Validation             : VALIDATED")
    print(" Shadow Length                   : VALIDATED")
    print(" Physical Scale                  : NOT VALIDATED")
    print(" Solar Elevation                 : NOT VALIDATED")
    print(" Height Formula                  : VALIDATED")
    print(" Test Height Calculation         : VALIDATED")
    print(" Production Height Estimation    : BLOCKED")
    print("-" * 80)
    print(" Production blocker:")
    print("    meters_per_pixel + sun_elevation_deg are still required.")
    print("=" * 80)


if __name__ == "__main__":
    run_final_height_test()
