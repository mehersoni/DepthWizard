"""
M4 Shadow Cue Module - Phase 4 Step 3: Geometric Shadow Length in Pixels

Calculates pixel distance L_px = sqrt(dx^2 + dy^2) for all validated [STRONG PAIR] candidates on demoImages/sat2.png,
cross-checks vectors against shadow_direction_vector, generates diagnostic plot & table, and outputs summary statistics.
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


def run_shadow_length_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 135)
    print(" M4 Shadow Cue Module - Phase 4 Step 3: Geometric Shadow Length in Pixels (sat2.png) ")
    print("=" * 135)

    # 1. Load Image
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 2. Candidate Extraction & Pairing
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    # Collect validated [STRONG PAIR] records
    validated_length_records = []
    lengths_px = []

    for r in ranked_regions[:10]:
        pairing = compute_object_shadow_pairing(image, r, corridor_width_factor=0.75, corridor_length_factor=1.5)
        if pairing["status"] == "[STRONG PAIR]":
            val = validate_shadow_base_tip(image, pairing)
            if val["status"] == "VALID":
                base = pairing["estimated_base_point"]
                tip = pairing["estimated_tip_point"]
                dx = tip[0] - base[0]
                dy = tip[1] - base[1]
                l_px = compute_shadow_length_px(base, tip)

                # Cross-check vector normalization
                calc_vec = np.array([dx, dy], dtype=np.float64) / (l_px + 1e-5)
                given_vec = np.array(pairing["shadow_direction_vector"], dtype=np.float64)
                dot_val = float(np.clip(np.dot(calc_vec, given_vec), -1.0, 1.0))
                dir_err = float(np.degrees(np.arccos(dot_val)))

                record = {
                    "candidate_id": pairing["candidate_id"],
                    "base": base,
                    "tip": tip,
                    "dx": dx,
                    "dy": dy,
                    "l_px": l_px,
                    "dir_err": dir_err,
                    "bounding_box": r["bounding_box"],
                    "oriented_bbox": r["oriented_bbox"],
                    "shadow_dir": pairing["shadow_direction_vector"]
                }
                validated_length_records.append(record)
                lengths_px.append(l_px)

    # 3. Print Required Diagnostic Table
    print("\n" + "=" * 135)
    print(" STEP 3 — GEOMETRIC SHADOW LENGTH TABLE (PIXELS) ")
    print("=" * 135)
    print(f"{'Pair ID':<8} | {'BASE (x, y)':<18} | {'TIP (x, y)':<18} | {'dx':<10} | {'dy':<10} | {'Shadow Length (px)':<18}")
    print("-" * 135)

    for rec in validated_length_records:
        cid = rec["candidate_id"]
        bx, by = rec["base"]
        tx, ty = rec["tip"]
        dx, dy = rec["dx"], rec["dy"]
        l_px = rec["l_px"]

        print(
            f"#{cid:<7} | "
            f"({bx:<7.1f}, {by:<7.1f}) | "
            f"({tx:<7.1f}, {ty:<7.1f}) | "
            f"{dx:<10.2f} | "
            f"{dy:<10.2f} | "
            f"{l_px:<18.2f}"
        )
    print("=" * 135)

    # 4. Summary Statistics
    num_pairs = len(lengths_px)
    min_len = float(np.min(lengths_px)) if num_pairs > 0 else 0.0
    max_len = float(np.max(lengths_px)) if num_pairs > 0 else 0.0
    mean_len = float(np.mean(lengths_px)) if num_pairs > 0 else 0.0
    median_len = float(np.median(lengths_px)) if num_pairs > 0 else 0.0

    print("\nSTEP 3 Summary Statistics:")
    print(f"  - Number of Validated Pairs: {num_pairs}")
    print(f"  - Minimum Shadow Length    : {min_len:.2f} px")
    print(f"  - Maximum Shadow Length    : {max_len:.2f} px")
    print(f"  - Mean Shadow Length       : {mean_len:.2f} px")
    print(f"  - Median Shadow Length     : {median_len:.2f} px")

    # 5. Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for rec in validated_length_records:
        cid = rec["candidate_id"]
        bx, by = rec["base"]
        tx, ty = rec["tip"]
        l_px = rec["l_px"]
        x, y, bw, bh = rec["bounding_box"]

        # Contour / Bounding Box
        cv.drawContours(overlay, [rec["oriented_bbox"]], 0, (0, 120, 255), 1)

        # BASE-TIP Line (Yellow)
        cv.line(overlay, (int(round(bx)), int(round(by))), (int(round(tx)), int(round(ty))), (255, 255, 0), 2)

        # BASE Marker (Red Circle)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 6, (255, 255, 255), 1)

        # TIP Marker (Cyan Square)
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        # Shadow Direction Arrow (Green)
        sdx, sdy = rec["shadow_dir"]
        arrow_end = (int(round(bx + sdx * 20)), int(round(by + sdy * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_end, (0, 255, 0), 2, tipLength=0.3)

        label = f"#{cid} L:{l_px:.1f}px"
        cv.putText(overlay, label, (x, max(12, y - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)
    plt.title("STEP 3 Shadow Length in Pixels Diagnostics (sat2.png)\n[Red Circle=BASE, Cyan Square=TIP, Yellow Line=BASE-TIP Segment, Green Arrow=ShadowDir]")
    plt.axis("off")
    plt.tight_layout()

    output_plot_path = os.path.join(output_dir, "shadow_length_diagnostics_sat2.png")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {output_plot_path}")
    print("=" * 135)


if __name__ == "__main__":
    run_shadow_length_test()
