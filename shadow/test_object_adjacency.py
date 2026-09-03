"""
M4 Shadow Cue Module - Phase 3 Object-Shadow Adjacency Test Runner

Tests Object-Shadow Adjacency & Pairing analysis on demoImages/sat2.png.
Constructs directional search corridors, evaluates local object evidence (intensity, Canny edges),
determines probable building-facing BASE vs. TIP endpoints, and outputs diagnostic visualizations & console table.
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
from shadow.geometry import extract_region_geometries, compute_object_shadow_adjacency
from shadow.confidence import rank_shadow_regions


def run_object_adjacency_test():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat2.png")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 185)
    print(" M4 Shadow Cue Module - Phase 3 Object-Shadow Adjacency Analysis (sat2.png) ")
    print("=" * 185)

    # 1. Load Image
    print(f"[Stage 1] Loading image: {image_path}")
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    h, w, c = image.shape
    print(f"         Resolution: {w} x {h} pixels ({c} channels)")

    # 2. Candidate Detection & Morphological Cleaning
    print("\n[Stage 2] Detecting candidate mask (v_max=125) & applying morphological cleaning...")
    raw_mask = detect_shadow_candidates(image, v_max=125)
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)

    # 3. Geometry Extraction & Candidate Ranking
    print("\n[Stage 3] Extracting geometries & selecting TOP 10 candidates...")
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    top10_candidates = ranked_regions[:10]
    print(f"         Extracted {len(ranked_regions)} total candidates. Selected TOP 10 candidates.")

    # 4. Compute Object-Shadow Adjacency for TOP 10
    adjacency_records = []
    good_count = 0
    ambiguous_count = 0
    weak_count = 0

    for candidate in top10_candidates:
        record = compute_object_shadow_adjacency(
            image,
            candidate,
            corridor_width_factor=0.75,
            corridor_length_factor=1.5,
            min_object_threshold=0.25,
            ambiguity_threshold=0.15
        )
        adjacency_records.append((candidate, record))

        st = record["status"]
        if st == "[GOOD]":
            good_count += 1
        elif st == "[AMBIGUOUS]":
            ambiguous_count += 1
        else:
            weak_count += 1

    # 5. Print Diagnostic Console Table
    print("\n" + "=" * 185)
    print(" PHASE 3 OBJECT-SHADOW ADJACENCY DIAGNOSTIC TABLE (TOP 10 CANDIDATES) ")
    print("=" * 185)
    print(f"{'ID':<4} | {'Orient':<8} | {'ObjectSide':<11} | {'ObjScore':<9} | {'AdjScore':<9} | {'Base (x, y)':<16} | {'Tip (x, y)':<16} | {'ShadowDir (dx, dy)':<20} | {'DConf':<8} | {'Status':<11}")
    print("-" * 185)

    for candidate, d in adjacency_records:
        cid = d["candidate_id"]
        orient = d["orientation_deg"]
        side = d["estimated_object_side"]
        oscore = d["object_score"]
        ascore = d["object_shadow_adjacency_score"]
        bx, by = d["estimated_base_point"]
        tx, ty = d["estimated_tip_point"]
        sdx, sdy = d["shadow_direction_vector"]
        dconf = d["direction_confidence"]
        st = d["status"]

        print(
            f"{cid:<4} | "
            f"{orient:<8.1f} | "
            f"{side:<11} | "
            f"{oscore:<9.3f} | "
            f"{ascore:<9.3f} | "
            f"({bx:<6.1f}, {by:<6.1f}) | "
            f"({tx:<6.1f}, {ty:<6.1f}) | "
            f"({sdx:<8.3f}, {sdy:<8.3f}) | "
            f"{dconf:<8.3f} | "
            f"{st:<11}"
        )
    print("=" * 185)

    # 6. Create Diagnostic Visualization
    overlay = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    for candidate, d in adjacency_records:
        cid = d["candidate_id"]
        x, y, bw, bh = candidate["bounding_box"]
        cx, cy = candidate["centroid"]
        bx, by = d["estimated_base_point"]
        tx, ty = d["estimated_tip_point"]
        sdx, sdy = d["shadow_direction_vector"]
        odx, ody = d["object_search_direction_vector"]
        oscore = d["object_score"]
        ascore = d["object_shadow_adjacency_score"]
        dconf = d["direction_confidence"]
        st = d["status"]

        # Draw Search Corridors (Polygons)
        poly_a = d["corridor_a_corners"]
        poly_b = d["corridor_b_corners"]
        cv.polylines(overlay, [poly_a], True, (255, 200, 0), 1)
        cv.polylines(overlay, [poly_b], True, (0, 200, 255), 1)

        # Draw Candidate Bounding Box & Contour
        cv.rectangle(overlay, (x, y), (x + bw, y + bh), (50, 200, 50), 1)
        cv.drawContours(overlay, [candidate["oriented_bbox"]], 0, (0, 120, 255), 1)

        # Centroid Marker (Blue dot)
        cv.circle(overlay, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

        # BASE Marker (Red circle) & TIP Marker (Cyan square)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 4, (255, 0, 0), -1)
        cv.circle(overlay, (int(round(bx)), int(round(by))), 6, (255, 255, 255), 1)
        cv.rectangle(overlay, (int(round(tx)) - 3, int(round(ty)) - 3), (int(round(tx)) + 3, int(round(ty)) + 3), (0, 255, 255), -1)

        # Arrow 1: BASE -> TIP (Green arrow)
        arrow_tip_end = (int(round(bx + sdx * 22)), int(round(by + sdy * 22)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_tip_end, (0, 255, 0), 2, tipLength=0.3)

        # Arrow 2: BASE -> OBJECT (Magenta arrow)
        arrow_obj_end = (int(round(bx + odx * 22)), int(round(by + ody * 22)))
        cv.arrowedLine(overlay, (int(round(bx)), int(round(by))), arrow_obj_end, (255, 0, 255), 2, tipLength=0.3)

        # Candidate Labels
        label = f"#{cid} Obj:{oscore:.2f} Conf:{dconf:.2f} {st}"
        cv.putText(
            overlay,
            label,
            (x, max(12, y - 3)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 0),
            1
        )

    plt.figure(figsize=(13, 9.5))
    plt.imshow(overlay)
    plt.title("Phase 3 Object-Shadow Adjacency & Pairing Diagnostics (sat2.png)\n[Polygons=Corridors, Red Circle=BASE, Cyan Square=TIP, Green Arrow=ShadowDir, Magenta Arrow=ObjectDir]")
    plt.axis("off")
    plt.tight_layout()

    output_plot_path = os.path.join(output_dir, "shadow_object_adjacency_sat2.png")
    plt.savefig(output_plot_path, dpi=150)
    plt.close()

    print(f"\nSaved diagnostic visualization: {output_plot_path}")
    print("\nPhase 3 Object-Shadow Pairing Breakdown:")
    print(f"  - Total Candidates Tested (Top 10): {len(top10_candidates)}")
    print(f"  - Candidates with STRONG Object Adjacency [GOOD]: {good_count}")
    print(f"  - Candidates with AMBIGUOUS Direction [AMBIGUOUS]: {ambiguous_count}")
    print(f"  - Candidates with WEAK Object Signals [WEAK]: {weak_count}")
    print("=" * 185)


if __name__ == "__main__":
    run_object_adjacency_test()
