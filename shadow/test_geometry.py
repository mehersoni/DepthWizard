"""
M4 Shadow Cue Module - Phase 2 Shadow Direction Geometry Test Runner

Tests directional geometry extraction on demoImages/sat2.png.
Estimates shadow base (object-facing end) vs. shadow tip (shadow tail end),
calculates normalized unit vectors, and produces diagnostic output and console table.
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
from shadow.geometry import extract_region_geometries, compute_shadow_directional_geometry
from shadow.confidence import rank_shadow_regions


def run_shadow_direction_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 175)
    print(" M4 Shadow Cue Module - Phase 2 Shadow Direction Geometry Analysis (sat2.png) ")
    print("=" * 175)

    # 1. Load Image
    print(f"[Stage 1] Loading image: {image_path}")
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    h, w, c = image.shape
    print(f"         Image Resolution: {w} x {h} pixels ({c} channels)")

    # 2. Candidate Detection & Morphological Cleaning
    print("\n[Stage 2] Detecting raw candidates (v_max=125) and applying morphological cleaning...")
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)

    # 3. Geometry Feature Extraction & Candidate Ranking
    print("\n[Stage 3] Extracting region geometries & ranking top candidate regions...")
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    print(f"         Total valid candidate regions extracted: {len(ranked_regions)}")

    # Select TOP 10 candidates
    top10_candidates = ranked_regions[:10]

    # 4. Compute Directional Geometry for TOP 10 Candidates
    directional_records = []
    high_conf_count = 0
    ambiguous_count = 0

    for candidate in top10_candidates:
        dir_geo = compute_shadow_directional_geometry(image, candidate, sampling_distance=5)
        directional_records.append((candidate, dir_geo))
        if dir_geo["is_ambiguous"]:
            ambiguous_count += 1
        else:
            high_conf_count += 1

    # 5. Print Required Console Table
    print("\n" + "=" * 175)
    print(" DIRECTIONAL GEOMETRY DIAGNOSTIC TABLE (TOP 10 CANDIDATES) ")
    print("=" * 175)
    print(f"{'ID':<4} | {'Orientation':<12} | {'Base (x, y)':<18} | {'Tip (x, y)':<18} | {'ShadowDir (dx, dy)':<22} | {'ObjectDir (dx, dy)':<22} | {'DirectionConf':<15} | {'Status':<10}")
    print("-" * 175)

    for candidate, d in directional_records:
        cid = d["candidate_id"]
        orient = d["orientation_deg"]
        bx, by = d["estimated_base_point"]
        tx, ty = d["estimated_tip_point"]
        sdx, sdy = d["shadow_direction_vector"]
        odx, ody = d["object_search_direction_vector"]
        dconf = d["direction_confidence"]
        status_str = "[UNCERTAIN]" if d["is_ambiguous"] else "[OK]"

        print(
            f"{cid:<4} | "
            f"{orient:<12.1f} | "
            f"({bx:<7.1f}, {by:<7.1f}) | "
            f"({tx:<7.1f}, {ty:<7.1f}) | "
            f"({sdx:<9.3f}, {sdy:<9.3f}) | "
            f"({odx:<9.3f}, {ody:<9.3f}) | "
            f"{dconf:<15.3f} | "
            f"{status_str:<10}"
        )
    print("=" * 175)

    # 6. Create Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for candidate, d in directional_records:
        cid = d["candidate_id"]
        x, y, bw, bh = candidate["bounding_box"]
        cx, cy = candidate["centroid"]
        bx, by = d["estimated_base_point"]
        tx, ty = d["estimated_tip_point"]
        sdx, sdy = d["shadow_direction_vector"]
        odx, ody = d["object_search_direction_vector"]
        dconf = d["direction_confidence"]

        # Draw Bounding Box & Contour
        cv.rectangle(overlay, (x, y), (x + bw, y + bh), (50, 200, 50), 1)
        cv.drawContours(overlay, [candidate["oriented_bbox"]], 0, (0, 120, 255), 1)

        # Draw Centroid (Blue dot)
        cv.circle(overlay, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

        # Draw BASE Marker (Red circle)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 6, (255, 255, 255), 1)

        # Draw TIP Marker (Cyan triangle/square)
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        # Arrow 1: BASE -> TIP (Shadow extension vector, Green arrow)
        arrow_tip_end = (int(round(bx + sdx * 20)), int(round(by + sdy * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_tip_end, (0, 255, 0), 2, tipLength=0.3)

        # Arrow 2: BASE -> OBJECT (Object search vector, Magenta arrow)
        arrow_obj_end = (int(round(bx + odx * 20)), int(round(by + ody * 20)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_obj_end, (255, 0, 255), 2, tipLength=0.3)

        # Candidate ID and Confidence Label
        label = f"#{cid} DConf:{dconf:.2f}"
        if d["is_ambiguous"]:
            label += " [UNCERTAIN]"
        cv.putText(
            overlay,
            label,
            (x, max(12, y - 3)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 0),
            1
        )

    plt.figure(figsize=(12, 9))
    plt.imshow(overlay)
    plt.title("Phase 2 M4 Shadow Direction Geometry Diagnostics (sat2.png)\n[Red Circle=BASE, Cyan Square=TIP, Green Arrow=ShadowDir, Magenta Arrow=ObjectDir]")
    plt.axis("off")
    plt.tight_layout()

    output_plot_path = os.path.join(output_dir, "shadow_direction_diagnostics_sat2.png")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {output_plot_path}")
    print("\nSummary Report:")
    print(f"  - Total Candidates Processed (Top Ranked): {len(top10_candidates)}")
    print(f"  - Candidates with High-Confidence Direction: {high_conf_count}")
    print(f"  - Candidates with Ambiguous Direction: {ambiguous_count}")
    print("=" * 175)


if __name__ == "__main__":
    run_shadow_direction_test()
