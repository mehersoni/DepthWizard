"""
M4 Shadow Cue Module - Phase 4 Step 2: BASE and TIP Endpoint Validation

Validates coordinate correctness, image boundary containment, non-zero length,
and directional vector consistency of estimated_base_point and estimated_tip_point
for all [STRONG PAIR] candidates on demoImages/sat2.png.
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
from shadow.geometry import extract_region_geometries, compute_object_shadow_pairing
from shadow.confidence import rank_shadow_regions


def validate_shadow_base_tip(image: np.ndarray, pairing_record: dict) -> dict:
    """
    Validate BASE and TIP points for a given candidate pairing record.

    Checks:
    1. Coordinates exist and are finite numeric floats.
    2. Points lie strictly within image dimensions.
    3. Distance between BASE and TIP > 0.
    4. Angular error between (TIP - BASE) vector and shadow_direction_vector <= threshold.
    5. Assigns status: VALID, WARNING, or INVALID.
    """
    h, w = image.shape[:2]
    cid = pairing_record["candidate_id"]
    base = pairing_record["estimated_base_point"]
    tip = pairing_record["estimated_tip_point"]
    shadow_dir = pairing_record["shadow_direction_vector"]

    bx, by = base
    tx, ty = tip

    # 1. Finite numeric coordinates check
    is_finite = np.isfinite(bx) and np.isfinite(by) and np.isfinite(tx) and np.isfinite(ty)
    if not is_finite:
        return {
            "candidate_id": cid,
            "status": "INVALID",
            "reason": "Non-finite coordinates",
            "dist_px": 0.0,
            "dir_error_deg": 180.0
        }

    # 2. Image boundary check
    in_bounds = (0 <= bx < w) and (0 <= by < h) and (0 <= tx < w) and (0 <= ty < h)
    if not in_bounds:
        return {
            "candidate_id": cid,
            "status": "INVALID",
            "reason": "Coordinates out of image bounds",
            "dist_px": 0.0,
            "dir_error_deg": 180.0
        }

    # 3. Distance check
    dx = tx - bx
    dy = ty - by
    dist_px = float(np.hypot(dx, dy))
    if dist_px < 1e-3:
        return {
            "candidate_id": cid,
            "status": "INVALID",
            "reason": "Zero-length BASE-TIP vector",
            "dist_px": 0.0,
            "dir_error_deg": 180.0
        }

    # 4. Direction consistency check
    calc_vec = np.array([dx, dy], dtype=np.float64) / dist_px
    given_vec = np.array(shadow_dir, dtype=np.float64)

    dot_prod = float(np.clip(np.dot(calc_vec, given_vec), -1.0, 1.0))
    dir_error_deg = float(np.degrees(np.arccos(dot_prod)))

    # Status classification based on explicit criteria
    if dir_error_deg <= 5.0 and dist_px >= 5.0:
        status = "VALID"
        reason = "Passes all spatial & directional checks"
    elif dir_error_deg <= 25.0:
        status = "WARNING"
        reason = "Minor direction alignment variance"
    else:
        status = "INVALID"
        reason = f"Direction error ({dir_error_deg:.1f} deg) exceeds tolerance"

    return {
        "candidate_id": cid,
        "base": (bx, by),
        "tip": (tx, ty),
        "dist_px": dist_px,
        "dir_error_deg": dir_error_deg,
        "status": status,
        "reason": reason
    }


def run_base_tip_validation():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Extract pipeline candidates & pairings
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    # Process all candidates to find [STRONG PAIR] records
    strong_pairs = []
    for r in ranked_regions[:10]:
        pairing = compute_object_shadow_pairing(image, r, corridor_width_factor=0.75, corridor_length_factor=1.5)
        if pairing["status"] == "[STRONG PAIR]":
            validation = validate_shadow_base_tip(image, pairing)
            strong_pairs.append((r, pairing, validation))

    # Print STEP 2 Compact Table
    print("\n" + "=" * 135)
    print(" STEP 2 — SHADOW BASE AND TIP VALIDATION TABLE ([STRONG PAIR] CANDIDATES) ")
    print("=" * 135)
    print(f"{'Pair ID':<8} | {'BASE (x, y)':<18} | {'TIP (x, y)':<18} | {'Distance px':<13} | {'Direction Error':<16} | {'Status':<10}")
    print("-" * 135)

    valid_count = 0
    warning_count = 0
    invalid_count = 0

    for r, p, val in strong_pairs:
        cid = val["candidate_id"]
        bx, by = val["base"]
        tx, ty = val["tip"]
        dpx = val["dist_px"]
        err = val["dir_error_deg"]
        st = val["status"]

        if st == "VALID":
            valid_count += 1
        elif st == "WARNING":
            warning_count += 1
        else:
            invalid_count += 1

        print(
            f"#{cid:<7} | "
            f"({bx:<7.1f}, {by:<7.1f}) | "
            f"({tx:<7.1f}, {ty:<7.1f}) | "
            f"{dpx:<13.2f} | "
            f"{err:<16.2f}° | "
            f"{st:<10}"
        )
    print("=" * 135)

    # Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for r, p, val in strong_pairs:
        cid = val["candidate_id"]
        bx, by = val["base"]
        tx, ty = val["tip"]
        st = val["status"]
        err = val["dir_error_deg"]
        dpx = val["dist_px"]

        x, y, bw, bh = r["bounding_box"]

        # Draw Candidate Contour & Oriented Bounding Box
        cv.drawContours(overlay, [r["oriented_bbox"]], 0, (0, 120, 255), 1)

        # Draw Line & Arrow BASE -> TIP
        cv.line(overlay, (int(round(bx)), int(round(by))), (int(round(tx)), int(round(ty))), (255, 255, 0), 2)

        # BASE Marker (Red circle)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 6, (255, 255, 255), 1)

        # TIP Marker (Cyan square)
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        # Arrow BASE -> TIP
        sdx, sdy = p["shadow_direction_vector"]
        arrow_end = (int(round(bx + sdx * 20)), int(round(by + sdy * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_end, (0, 255, 0), 2, tipLength=0.3)

        label = f"#{cid} Dist:{dpx:.1f}px Err:{err:.1f}deg [{st}]"
        cv.putText(overlay, label, (x, max(12, y - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)
    plt.title("STEP 2 BASE and TIP Endpoint Validation Diagnostics (sat2.png)\n[Red Circle=BASE, Cyan Square=TIP, Yellow Line=BASE-TIP Segment, Green Arrow=ShadowDir]")
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "shadow_base_tip_validation_sat2.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {out_path}")
    print("\nSTEP 2 Summary Breakdown:")
    print(f"  - Number of [STRONG PAIR] Candidates Tested: {len(strong_pairs)}")
    print(f"  - Number VALID  : {valid_count}")
    print(f"  - Number WARNING: {warning_count}")
    print(f"  - Number INVALID: {invalid_count}")
    print("=" * 135)


if __name__ == "__main__":
    run_base_tip_validation()
