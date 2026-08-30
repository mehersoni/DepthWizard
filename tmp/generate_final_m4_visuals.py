"""
DeepthWizard — Final Visual Demonstration Generator

Generates high-quality visual demonstration overlays for the frozen M4 physical
shadow-based building height estimation algorithm across 6 representative cases:
1. Small Building Success (< 4.0m)
2. Medium Building Success (4.0m - 12.0m)
3. Large Building Success (>= 12.0m)
4. Representative FALSE_SHORT_SHADOW Failure Mode
5. Representative FALSE_LONG_SHADOW Failure Mode
6. Representative LOW_CONFIDENCE / Difficult Case

GROUND-TRUTH HEIGHT IS INCLUDED ONLY AS AN EVALUATION ANNOTATION, NEVER AS AN INFERENCE INPUT.
"""

import os
import sys
import glob
import math
import cv2 as cv
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from shadow.detector import detect_shadow_candidates
from shadow.cleaner import clean_candidate_mask
from shadow.m4_physical_raycast_experiment import measure_building_shadow_m4_physical
from tmp.final_m4_inference import parse_tfw, compute_pca_shadow_direction, predict_building_height_m4


def render_visual_case(
    rgb_img: np.ndarray,
    cleaned_mask: np.ndarray,
    building_contour: np.ndarray,
    gt_height_m: float,
    case_title: str,
    output_filename: str,
    gsd: float = 0.05,
    sun_elev: float = 41.8
) -> np.ndarray:
    """Renders a 2-panel diagnostic visualization for a single building case."""
    
    # Run FROZEN M4 inference (NO GT INPUT)
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2]
    u_x, u_y = compute_pca_shadow_direction(cleaned_mask)
    
    m4_res = measure_building_shadow_m4_physical(
        building_contour, cleaned_mask, (u_x, u_y), gsd, sun_elev, v_channel
    )

    pred_h = float(m4_res["shadow_length_m"] * math.tan(math.radians(sun_elev)))
    status = m4_res["status"]
    conf = m4_res["confidence"]
    term_reason = m4_res["termination_reason"]
    l_shadow_m = m4_res["shadow_length_m"]
    abs_err = abs(pred_h - gt_height_m)

    # Compute bounding box with margin
    pts = building_contour.reshape(-1, 2)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 2].max() if pts.shape[1] > 2 else pts[:, 0].max(), pts[:, 1].max()
    
    # Include tip point in bounding box
    p0_x, p0_y = m4_res["base_point"]
    p1_x, p1_y = m4_res["tip_point"]

    x_min_crop = int(max(0, min(x_min, p0_x, p1_x) - 40))
    y_min_crop = int(max(0, min(y_min, p0_y, p1_y) - 40))
    x_max_crop = int(min(rgb_img.shape[1], max(x_max, p0_x, p1_x) + 40))
    y_max_crop = int(min(rgb_img.shape[0], max(y_max, p0_y, p1_y) + 40))

    # Crop RGB and Mask
    crop_rgb = rgb_img[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()
    crop_mask = cleaned_mask[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    crop_mask_bgr = cv.cvtColor(crop_mask, cv.COLOR_GRAY2BGR)

    # Adjust coordinates for crop
    cnt_crop = building_contour.copy()
    cnt_crop[:, 0, 0] -= x_min_crop
    cnt_crop[:, 0, 1] -= y_min_crop
    
    cp0_x, cp0_y = int(round(p0_x - x_min_crop)), int(round(p0_y - y_min_crop))
    cp1_x, cp1_y = int(round(p1_x - x_min_crop)), int(round(p1_y - y_min_crop))

    # Color palette
    outline_col = (0, 255, 0) if status == "VALID" and abs_err < 5.0 else (0, 0, 255)
    base_col = (0, 255, 255)   # Yellow
    tip_col = (255, 255, 0)    # Cyan
    ray_col = (255, 0, 255)    # Magenta

    # Draw Panel 1: RGB Overlay
    cv.polylines(crop_rgb, [cnt_crop], isClosed=True, color=outline_col, thickness=2)
    cv.line(crop_rgb, (cp0_x, cp0_y), (cp1_x, cp1_y), ray_col, 2)
    cv.circle(crop_rgb, (cp0_x, cp0_y), 4, base_col, -1)
    cv.circle(crop_rgb, (cp1_x, cp1_y), 5, tip_col, -1)

    # Draw Panel 2: Shadow Mask Overlay
    cv.polylines(crop_mask_bgr, [cnt_crop], isClosed=True, color=outline_col, thickness=1)
    cv.line(crop_mask_bgr, (cp0_x, cp0_y), (cp1_x, cp1_y), ray_col, 2)
    cv.circle(crop_mask_bgr, (cp0_x, cp0_y), 4, base_col, -1)
    cv.circle(crop_mask_bgr, (cp1_x, cp1_y), 5, tip_col, -1)

    # Resize panels for display
    target_h = 320
    w1 = int(round(crop_rgb.shape[1] * (target_h / crop_rgb.shape[0])))
    p1_resized = cv.resize(crop_rgb, (w1, target_h))
    p2_resized = cv.resize(crop_mask_bgr, (w1, target_h))

    # Combine panels side-by-side
    panel_combined = np.hstack((p1_resized, p2_resized))

    # Create top header and bottom info text banner
    header_h, banner_h = 50, 110
    canvas_w = panel_combined.shape[1]
    canvas = np.ones((header_h + target_h + banner_h, canvas_w, 3), dtype=np.uint8) * 245

    # Insert panel
    canvas[header_h:header_h + target_h, :canvas_w] = panel_combined

    # Draw Title Header
    cv.putText(canvas, f"Case: {case_title}", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv.line(canvas, (0, header_h - 2), (canvas_w, header_h - 2), (200, 200, 200), 1)

    # Draw Bottom Info Banner
    y_start = header_h + target_h + 20
    info_line1 = f"Pred Height: {pred_h:.2f} m  |  Shadow Len: {l_shadow_m:.2f} m  |  Status: {status} (Conf: {conf:.2f})"
    info_line2 = f"Eval Annotation (Post-Hoc ONLY): GT Height = {gt_height_m:.2f} m  |  Abs Error = {abs_err:.2f} m"
    info_line3 = f"Termination Reason: {term_reason}  |  PCA Vector: ({u_x:.3f}, {u_y:.3f})"

    cv.putText(canvas, info_line1, (15, y_start), cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 150), 1)
    cv.putText(canvas, info_line2, (15, y_start + 25), cv.FONT_HERSHEY_SIMPLEX, 0.48, (150, 0, 0), 1)
    cv.putText(canvas, info_line3, (15, y_start + 50), cv.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    # Save output image
    out_path = os.path.join(root_dir, "output", output_filename)
    cv.imwrite(out_path, canvas)
    print(f"Saved visual overlay: {out_path}")
    return canvas


def generate_all_final_visuals():
    print("=========================================================")
    print("      GENERATING FINAL M4 VISUAL DEMONSTRATIONS         ")
    print("=========================================================")

    dataset_dir = os.path.join(root_dir, "Dataset", "Potsdam")
    tile_id = "2_10"
    
    rgb_matches = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_{tile_id}_RGB.tif"), recursive=True)
    label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_{tile_id}_label_noBoundary.tif"), recursive=True)

    if not rgb_matches or not label_matches:
        print("Error: Potsdam Tile 2_10 files not found!")
        return

    rgb_img = cv.imread(rgb_matches[0])
    label_img = cv.imread(label_matches[0])
    
    raw_mask = detect_shadow_candidates(rgb_img)
    cleaned_mask = clean_candidate_mask(raw_mask)

    # Extract building contours
    if len(label_img.shape) == 3:
        building_mask = ((label_img[:, :, 2] > 200) & (label_img[:, :, 1] > 200) & (label_img[:, :, 0] > 200)).astype(np.uint8) * 255
    else:
        building_mask = (label_img == 255).astype(np.uint8) * 255

    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(building_mask, connectivity=8)
    
    contour_map = {}
    for lbl in range(1, num_labels):
        if stats[lbl, cv.CC_STAT_AREA] < 100:
            continue
        single_mask = (labels_im == lbl).astype(np.uint8) * 255
        cnts, _ = cv.findContours(single_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if cnts:
            contour_map[lbl] = max(cnts, key=cv.contourArea)

    # Defined representative test cases (Tile 2_10)
    test_cases = [
        {"b_id": 4, "gt_h": 3.65, "title": "1. Small Building Success (< 4.0m)", "out": "final_demo_1_small_success.png"},
        {"b_id": 2, "gt_h": 8.52, "title": "2. Medium Building Success (4.0m - 12.0m)", "out": "final_demo_2_medium_success.png"},
        {"b_id": 1, "gt_h": 15.22, "title": "3. Large Building Success (>= 12.0m)", "out": "final_demo_3_large_success.png"},
        {"b_id": 3, "gt_h": 12.87, "title": "4. Representative FALSE_SHORT_SHADOW Failure Mode", "out": "final_demo_4_false_short.png"},
        {"b_id": 5, "gt_h": 2.27, "title": "5. Representative FALSE_LONG_SHADOW Failure Mode", "out": "final_demo_5_false_long.png"},
        {"b_id": 10, "gt_h": 10.21, "title": "6. Representative LOW_CONFIDENCE / Difficult Case", "out": "final_demo_6_low_confidence.png"}
    ]

    rendered_canvases = []
    for tc in test_cases:
        b_id = tc["b_id"]
        if b_id in contour_map:
            cnt = contour_map[b_id]
            canvas = render_visual_case(
                rgb_img=rgb_img,
                cleaned_mask=cleaned_mask,
                building_contour=cnt,
                gt_height_m=tc["gt_h"],
                case_title=tc["title"],
                output_filename=tc["out"]
            )
            rendered_canvases.append(canvas)

    print("\nVisual demonstration overlays successfully generated!")


if __name__ == "__main__":
    generate_all_final_visuals()
