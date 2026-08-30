"""
DeepthWizard — Final M4 Inference Entry Point

This module provides a clean, production-ready inference API for building height
estimation using physical shadow geometry.

FROZEN ALGORITHM: Production M4 (measure_building_shadow_m4_physical)
NO GROUND-TRUTH HEIGHT IS EVER USED AS AN INPUT TO THIS INFERENCE PIPELINE.
"""

import os
import sys
import math
import argparse
from typing import Dict, Any, Tuple, Optional, List
import cv2 as cv
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from shadow.detector import detect_shadow_candidates
from shadow.cleaner import clean_candidate_mask
from shadow.m4_physical_raycast_experiment import measure_building_shadow_m4_physical


def parse_tfw(tfw_path: str) -> float:
    """Extracts Ground Sample Distance (GSD, meters per pixel) from world file."""
    try:
        with open(tfw_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= 1:
            dx = abs(float(lines[0]))
            if 0.001 <= dx <= 10.0:
                return dx
    except Exception:
        pass
    return 0.05  # Default Potsdam GSD


def compute_pca_shadow_direction(shadow_mask: np.ndarray) -> Tuple[float, float]:
    """Computes principal shadow direction vector using PCA on shadow mask coordinates."""
    y_coords, x_coords = np.where(shadow_mask > 0)
    if len(x_coords) < 50:
        return (0.7071, 0.7071)  # Default fallback direction (~45 degrees)

    pts = np.vstack((x_coords, y_coords)).T.astype(np.float64)
    mean, eigenvectors = cv.PCACompute(pts, mean=None)
    u_x, u_y = eigenvectors[0, 0], eigenvectors[0, 1]

    # Orient vector towards bottom-right quadrant (standard shadow orientation in Potsdam)
    if u_y < 0:
        u_x, u_y = -u_x, -u_y
    if u_x < 0 and abs(u_x) > abs(u_y):
        u_x, u_y = -u_x, -u_y

    norm = math.hypot(u_x, u_y)
    return (u_x / norm, u_y / norm) if norm > 1e-5 else (0.7071, 0.7071)


def predict_building_height_m4(
    rgb_img: np.ndarray,
    building_contour: np.ndarray,
    meters_per_pixel: float = 0.05,
    sun_elevation_deg: float = 41.8,
    precomputed_shadow_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Predicts building height for a single building contour using frozen M4 physical raycasting.

    Parameters:
        rgb_img: 3-channel RGB image (uint8).
        building_contour: NumPy array representing OpenCV building contour points.
        meters_per_pixel: Ground Sample Distance (GSD) in m/px.
        sun_elevation_deg: Solar elevation angle in degrees.
        precomputed_shadow_mask: Optional binary shadow mask.

    Returns:
        Dict containing prediction metadata, shadow length, building height, base/tip points, and status.
    """
    if rgb_img is None or building_contour is None or len(building_contour) == 0:
        return {
            "predicted_height_m": 0.0,
            "shadow_length_m": 0.0,
            "shadow_length_px": 0.0,
            "base_point": (0.0, 0.0),
            "tip_point": (0.0, 0.0),
            "status": "REJECTED",
            "confidence": 0.0,
            "termination_reason": "INVALID_INPUT",
            "meters_per_pixel": meters_per_pixel,
            "sun_elevation_deg": sun_elevation_deg
        }

    # Extract V-channel (brightness/value) from HSV color space
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2]

    # Shadow candidate detection and Morphological cleaning
    if precomputed_shadow_mask is None:
        raw_mask = detect_shadow_candidates(rgb_img)
        cleaned_mask = clean_candidate_mask(raw_mask)
    else:
        cleaned_mask = precomputed_shadow_mask

    # Principal shadow vector computation
    u_x, u_y = compute_pca_shadow_direction(cleaned_mask)

    # Execute FROZEN M4 physical raycast
    m4_result = measure_building_shadow_m4_physical(
        building_contour=building_contour,
        cleaned_mask=cleaned_mask,
        shadow_direction=(u_x, u_y),
        meters_per_pixel=meters_per_pixel,
        sun_elevation_deg=sun_elevation_deg,
        image_v_channel=v_channel,
        max_plausible_height_m=40.0,
        max_gap_allowed_px=2
    )

    # Physical height conversion: H = L_shadow * tan(sun_elevation)
    shadow_length_m = m4_result["shadow_length_m"]
    tan_elev = math.tan(math.radians(sun_elevation_deg))
    predicted_height_m = shadow_length_m * tan_elev

    return {
        "predicted_height_m": float(predicted_height_m),
        "shadow_length_m": float(shadow_length_m),
        "shadow_length_px": float(m4_result["shadow_length_px"]),
        "base_point": (float(m4_result["base_point"][0]), float(m4_result["base_point"][1])),
        "tip_point": (float(m4_result["tip_point"][0]), float(m4_result["tip_point"][1])),
        "shadow_direction": (float(u_x), float(u_y)),
        "status": m4_result["status"],
        "confidence": float(m4_result["confidence"]),
        "termination_reason": m4_result["termination_reason"],
        "rejection_reason": m4_result.get("rejection_reason"),
        "density": float(m4_result.get("density", 0.0)),
        "supported_pixels": int(m4_result.get("supported_pixels", 0)),
        "meters_per_pixel": float(meters_per_pixel),
        "sun_elevation_deg": float(sun_elevation_deg)
    }


def demo_inference_on_potsdam_tile(tile_id: str = "2_10"):
    """Demonstrates inference on a Potsdam tile sample without GT input."""
    import glob
    print(f"\n--- Demonstrating M4 Final Inference on Potsdam Tile {tile_id} ---")
    
    dataset_dir = os.path.join(root_dir, "Dataset", "Potsdam")
    rgb_matches = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_{tile_id}_RGB.tif"), recursive=True)
    if not rgb_matches:
        rgb_matches = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_0{tile_id}_RGB.tif"), recursive=True)

    if not rgb_matches:
        print(f"Error: Tile RGB file for {tile_id} not found in {dataset_dir}.")
        return

    rgb_path = rgb_matches[0]
    parent_dir = os.path.dirname(rgb_path)
    
    tfw_path = os.path.join(parent_dir, f"top_potsdam_{tile_id}_RGB.tfw")
    if not os.path.exists(tfw_path):
        tfw_path = os.path.join(parent_dir, f"top_potsdam_0{tile_id}_RGB.tfw")

    label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_{tile_id}_label_noBoundary.tif"), recursive=True)
    if not label_matches:
        label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_0{tile_id}_label_noBoundary.tif"), recursive=True)
    if not label_matches:
        label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_{tile_id}_label.tif"), recursive=True)
    if not label_matches:
        label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_0{tile_id}_label.tif"), recursive=True)

    if not label_matches:
        print(f"Error: Label file for {tile_id} not found in {dataset_dir}.")
        return

    label_path = label_matches[0]

    rgb_img = cv.imread(rgb_path)
    label_img = cv.imread(label_path)
    gsd = parse_tfw(tfw_path) if os.path.exists(tfw_path) else 0.05

    # Extract building contours from ground-truth mask ONLY to define building footprints
    if len(label_img.shape) == 3:
        building_mask = ((label_img[:, :, 2] > 200) & (label_img[:, :, 1] > 200) & (label_img[:, :, 0] > 200)).astype(np.uint8) * 255
    else:
        building_mask = (label_img == 255).astype(np.uint8) * 255

    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(building_mask, connectivity=8)
    
    evaluated_count = 0
    print(f"Discovered {num_labels - 1} raw building components in label mask.")

    for lbl in range(1, min(6, num_labels)):
        area_px = stats[lbl, cv.CC_STAT_AREA]
        if area_px < 100:
            continue

        single_mask = (labels_im == lbl).astype(np.uint8) * 255
        contours, _ = cv.findContours(single_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv.contourArea)

        # Run M4 inference (NO GT HEIGHT INPUT)
        res = predict_building_height_m4(rgb_img, cnt, meters_per_pixel=gsd, sun_elevation_deg=41.8)

        print(f"\nBuilding #{lbl}:")
        print(f"  Footprint Area: {area_px} px ({area_px * gsd * gsd:.1f} m^2)")
        print(f"  Base Point: ({res['base_point'][0]:.1f}, {res['base_point'][1]:.1f})")
        print(f"  Tip Point: ({res['tip_point'][0]:.1f}, {res['tip_point'][1]:.1f})")
        print(f"  Shadow Direction: ({res['shadow_direction'][0]:.4f}, {res['shadow_direction'][1]:.4f})")
        print(f"  Predicted Shadow Length: {res['shadow_length_m']:.2f} m ({res['shadow_length_px']:.1f} px)")
        print(f"  PREDICTED HEIGHT: {res['predicted_height_m']:.2f} m")
        print(f"  Status: {res['status']} | Confidence: {res['confidence']:.2f}")
        print(f"  Termination Reason: {res['termination_reason']}")
        evaluated_count += 1

    print(f"\nSuccessfully demonstrated inference on {evaluated_count} buildings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 Final Inference Entry Point")
    parser.add_argument("--tile", type=str, default="2_10", help="Potsdam tile ID (e.g. 2_10)")
    args = parser.parse_args()
    demo_inference_on_potsdam_tile(args.tile)
