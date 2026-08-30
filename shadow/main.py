"""
M4 Shadow Cue Module - End-to-End Test Runner with Shadow Shape & Structural Cleanliness Scoring

Pipeline Flow:
RGB Satellite Image -> Raw Candidate Mask -> Cleaned Mask -> Region Geometries -> Shape & Complexity Analysis -> Confidence Ranking
"""

import json
import os
import sys

# Ensure root workspace directory is in sys.path when running script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow.detector import detect_shadow_candidates
from shadow.cleaner import clean_candidate_mask
from shadow.geometry import extract_region_geometries
from shadow.confidence import rank_shadow_regions


def run_pipeline_demo():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(root_dir, "demoImages", "sat1.jpg")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 185)
    print(" M4 Shadow Cue Module - Final Candidate Confidence Ranking Pipeline ")
    print("=" * 185)
    
    # Stage 1: Load Image
    print(f"[Stage 1] Loading image: {image_path}")
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Sample satellite image not found at {image_path}")
    h, w, c = image.shape
    print(f"         Resolution: {w} x {h} ({h*w:,} total pixels)")

    # Stage 2: Raw Candidate Mask Detection
    print("\n[Stage 2] Detecting raw shadow candidates via HSV thresholding...")
    raw_mask = detect_shadow_candidates(image, v_max=80)
    raw_count = np.count_nonzero(raw_mask)
    print(f"         Raw shadow pixels: {raw_count:,} ({(raw_count / (h*w))*100:.2f}%)")

    # Stage 3: Morphological Cleaning
    print("\n[Stage 3] Cleaning candidate mask via morphological opening & closing...")
    cleaned_mask = clean_candidate_mask(raw_mask, kernel_size=3, open_iterations=1, close_iterations=1)
    cleaned_count = np.count_nonzero(cleaned_mask)
    print(f"         Cleaned shadow pixels: {cleaned_count:,} ({(cleaned_count / (h*w))*100:.2f}%)")

    # Stage 4: Geometry Feature Extraction
    print("\n[Stage 4] Extracting connected component region geometries...")
    unranked_regions = extract_region_geometries(cleaned_mask, min_area=20.0)
    print(f"         Extracted {len(unranked_regions)} valid shadow regions (min_area >= 20.0 px).")

    # Stage 5: Candidate Ranking & Shape / Complexity Analysis
    print("\n[Stage 5] Computing ShadowShapeScore, ComplexityScore, MergeSuspicion, and FinalConfidence...")
    ranked_regions = rank_shadow_regions(image, unranked_regions)

    # Required Output Table for ALL candidates with all 12 specified fields (Sorted by Region ID)
    print("\n" + "=" * 175)
    print(" DIAGNOSTIC TABLE: ALL 22 CANDIDATE REGIONS (SORTED BY REGION ID) ")
    print("=" * 175)
    print(f"{'ID':<4} | {'Area':<7} | {'AspectR':<7} | {'Elong':<7} | {'Solid':<7} | {'Extent':<7} | {'GeoScore':<9} | {'ShapeScore':<10} | {'ComplxScore':<10} | {'MergeSusp':<9} | {'StructRel':<9} | {'FinalConf':<9}")
    print("-" * 175)
    
    sorted_by_id = sorted(ranked_regions, key=lambda item: item["id"])
    for r in sorted_by_id:
        s = r["scores"]
        print(
            f"{r['id']:<4} | "
            f"{r['area']:<7.1f} | "
            f"{r['aspect_ratio']:<7.2f} | "
            f"{r['elongation']:<7.3f} | "
            f"{r['solidity']:<7.3f} | "
            f"{r['extent']:<7.3f} | "
            f"{s['geometry_score']:<9.3f} | "
            f"{s['shadow_shape_score']:<10.3f} | "
            f"{s['complexity_score']:<10.3f} | "
            f"{s['merge_suspicion']:<9.3f} | "
            f"{s['structural_reliability']:<9.3f} | "
            f"{s['confidence_score']:<9.3f}"
        )
    print("=" * 175)

    # Validation Table: Final Ranked Candidates (Ordered by Rank)
    print("\n" + "=" * 185)
    print(" FINAL RANKED SHADOW CANDIDATE TABLE (TOP 10 RANKED CANDIDATES) ")
    print("=" * 185)
    print(f"{'Rank':<5} | {'ID':<4} | {'FinalConf':<10} | {'ShapeScore':<11} | {'ContrScore':<11} | {'StructRel':<10} | {'AreaRel':<8} | {'Area':<7} | {'AspectR':<8} | {'Elong':<7} | {'Solid':<7}")
    print("=" * 185)
    
    for r in ranked_regions[:10]:
        s = r["scores"]
        print(
            f"#{r['confidence_rank']:<4} | "
            f"{r['id']:<4} | "
            f"{s['confidence_score']:<10.3f} | "
            f"{s['shadow_shape_score']:<11.3f} | "
            f"{s['contrast_score']:<11.3f} | "
            f"{s['structural_reliability']:<10.3f} | "
            f"{s['area_reliability']:<8.3f} | "
            f"{r['area']:<7.1f} | "
            f"{r['aspect_ratio']:<8.2f} | "
            f"{r['elongation']:<7.3f} | "
            f"{r['solidity']:<7.3f}"
        )
    print("=" * 185)

    # Export measurements and all sub-scores to JSON files
    json_output_path = os.path.join(output_dir, "shadow_regions_confidence.json")
    json_regions_data = []
    
    for r in ranked_regions:
        json_regions_data.append({
            "confidence_rank": int(r["confidence_rank"]),
            "id": int(r["id"]),
            "scores": {
                "confidence_score": float(r["scores"]["confidence_score"]),
                "c_raw": float(r["scores"]["c_raw"]),
                "area_reliability": float(r["scores"]["area_reliability"]),
                "structural_reliability": float(r["scores"]["structural_reliability"]),
                "complexity_score": float(r["scores"]["complexity_score"]),
                "merge_suspicion": float(r["scores"]["merge_suspicion"]),
                "shadow_shape_score": float(r["scores"]["shadow_shape_score"]),
                "geometry_score": float(r["scores"]["geometry_score"]),
                "contrast_score": float(r["scores"]["contrast_score"]),
                "raw_contrast_ratio": float(r["scores"]["raw_contrast_ratio"]),
                "mean_inner_intensity": float(r["scores"]["mean_inner_intensity"]),
                "mean_outer_intensity": float(r["scores"]["mean_outer_intensity"]),
                "orientation_score": float(r["scores"]["orientation_score"]),
                "dominant_angle_deg": float(r["scores"]["dominant_angle_deg"])
            },
            "area": float(r["area"]),
            "centroid": [float(r["centroid"][0]), float(r["centroid"][1])],
            "bounding_box": [int(b) for b in r["bounding_box"]],
            "perimeter": float(r["perimeter"]),
            "major_axis_length": float(r["major_axis_length"]),
            "minor_axis_length": float(r["minor_axis_length"]),
            "aspect_ratio": float(r["aspect_ratio"]),
            "elongation": float(r["elongation"]),
            "solidity": float(r["solidity"]),
            "extent": float(r["extent"]),
            "orientation_deg": float(r["orientation_deg"]),
            "oriented_bbox": r["oriented_bbox"].tolist()
        })

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_regions_data, f, indent=2)

    with open(os.path.join(output_dir, "shadow_regions.json"), "w", encoding="utf-8") as f:
        json.dump(json_regions_data, f, indent=2)
    
    print(f"\nSaved all region measurements and confidence scores to JSON: {json_output_path}")

    # Stage 6: Visualizations
    top5_regions = ranked_regions[:5]
    overlay_top5 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    for r in top5_regions:
        x, y, bw, bh = r["bounding_box"]
        conf = r["scores"]["confidence_score"]
        cv.rectangle(overlay_top5, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv.drawContours(overlay_top5, [r["oriented_bbox"]], 0, (0, 120, 255), 1)
        cx, cy = int(r["centroid"][0]), int(r["centroid"][1])
        cv.circle(overlay_top5, (cx, cy), 2, (255, 0, 0), -1)
        cv.putText(
            overlay_top5,
            f"#{r['id']} C:{conf:.2f}",
            (x, max(12, y - 3)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1
        )

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("1. Original Satellite RGB Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(raw_mask, cmap="gray")
    plt.title(f"2. Candidate Mask (HSV)\n({raw_count:,} px)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cleaned_mask, cmap="gray")
    plt.title(f"3. Cleaned Mask (Morphology)\n({cleaned_count:,} px)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(overlay_top5)
    plt.title("4. Ranked Shadow Candidates (TOP 5 ONLY)\n[Green=BBox, Blue=RotatedBox, Label=#ID C:Score]")
    plt.axis("off")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "shadow_pipeline_stage_diagnostics.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Dedicated Top 10 Visualization
    top10_regions = ranked_regions[:10]
    overlay_top10 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    for r in top10_regions:
        x, y, bw, bh = r["bounding_box"]
        conf = r["scores"]["confidence_score"]
        cv.rectangle(overlay_top10, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv.drawContours(overlay_top10, [r["oriented_bbox"]], 0, (0, 120, 255), 1)
        cx, cy = int(r["centroid"][0]), int(r["centroid"][1])
        cv.circle(overlay_top10, (cx, cy), 2, (255, 0, 0), -1)
        cv.putText(
            overlay_top10,
            f"#{r['id']} C:{conf:.2f}",
            (x, max(10, y - 2)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 0),
            1
        )

    plt.figure(figsize=(10, 7))
    plt.imshow(overlay_top10)
    plt.title("TOP 10 Candidate Shadow Regions (Final Candidate Ranking)")
    plt.axis("off")
    plt.tight_layout()
    top10_output_path = os.path.join(output_dir, "shadow_top10_candidates.png")
    plt.savefig(top10_output_path, dpi=150)
    plt.close()

    print(f"Saved 4-panel diagnostic plot (Top 5 panel): {output_path}")
    print(f"Saved dedicated Top 10 candidate visualization: {top10_output_path}")
    print("=" * 185)


if __name__ == "__main__":
    run_pipeline_demo()
