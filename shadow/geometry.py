"""
M4 Shadow Cue Module - Shadow Geometry Extraction Component

This module extracts geometric and morphological features from cleaned binary shadow masks,
and provides directional geometry, object-shadow adjacency, strict object-shadow pairing,
and geometric pixel length calculations.
"""

from typing import List, Dict, Any, Tuple
import cv2 as cv
import numpy as np


def extract_region_geometries(
    cleaned_mask: np.ndarray,
    min_area: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Extract geometric features and structural properties for all connected candidate shadow regions.

    Features Extracted Per Region:
    ------------------------------
    - id / label: Unique region identifier (1-indexed).
    - area: Region area in pixels (float).
    - bounding_box: Axis-aligned bounding rectangle (x, y, width, height).
    - centroid: Sub-pixel center of mass (cx, cy).
    - perimeter: Contour boundary length.
    - major_axis_length: Length of major axis from minimum area enclosing rectangle.
    - minor_axis_length: Length of minor axis from minimum area enclosing rectangle.
    - aspect_ratio: Ratio of major axis to minor axis (major / minor).
    - elongation: Measure of region stretch: 1 - (minor / major). Range [0, 1).
    - solidity: Ratio of region contour area to its convex hull area (area / convex_hull_area).
    - extent: Ratio of region contour area to its axis-aligned bounding box area (area / (w * h)).
    - orientation_deg: Mathematical orientation angle in degrees from minAreaRect.
    - contour: OpenCV contour array (N, 1, 2).
    - oriented_bbox: Corner points of the rotated minimum area bounding box.

    Parameters:
    -----------
    cleaned_mask : np.ndarray
        Cleaned 2D binary mask (H, W), dtype uint8 (255 = shadow candidate, 0 = background).
    min_area : float, default=20.0
        Minimum area threshold (in pixels). Regions smaller than min_area are discarded.

    Returns:
    --------
    regions : List[Dict[str, Any]]
        List of dictionaries containing structured geometric properties for each detected region.
    """
    if cleaned_mask is None or not isinstance(cleaned_mask, np.ndarray):
        raise ValueError("Input mask must be a valid NumPy array.")

    if cleaned_mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, received shape {cleaned_mask.shape}")

    # Find external contours of all shadow regions
    contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    regions: List[Dict[str, Any]] = []
    region_id = 1

    for contour in contours:
        area = float(cv.contourArea(contour))
        if area < min_area:
            continue

        # Axis-aligned bounding box (x, y, w, h)
        x, y, w, h = cv.boundingRect(contour)

        # Image moments for centroid calculation
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)

        # Boundary perimeter
        perimeter = float(cv.arcLength(contour, True))

        # Minimum area enclosing rotated rectangle
        rect = cv.minAreaRect(contour)
        (rect_cx, rect_cy), (rect_w, rect_h), angle = rect

        # Major and minor axis lengths
        major_axis = float(max(rect_w, rect_h))
        minor_axis = float(min(rect_w, rect_h))

        # Avoid division by zero
        if minor_axis > 1e-5:
            aspect_ratio = float(major_axis / minor_axis)
        else:
            aspect_ratio = float(major_axis)

        if major_axis > 1e-5:
            elongation = float(1.0 - (minor_axis / major_axis))
        else:
            elongation = 0.0

        # Convex Hull & Solidity
        hull = cv.convexHull(contour)
        hull_area = float(cv.contourArea(hull))
        if hull_area > 1e-5:
            solidity = float(area / hull_area)
        else:
            solidity = 1.0

        # Extent (area / bounding box area)
        bbox_area = float(w * h)
        if bbox_area > 1e-5:
            extent = float(area / bbox_area)
        else:
            extent = 1.0

        # Corners of oriented bounding box
        box_points = cv.boxPoints(rect)
        box_points = np.int32(box_points)

        region_info = {
            "id": region_id,
            "area": area,
            "bounding_box": (x, y, w, h),
            "centroid": (cx, cy),
            "perimeter": perimeter,
            "major_axis_length": major_axis,
            "minor_axis_length": minor_axis,
            "aspect_ratio": aspect_ratio,
            "elongation": elongation,
            "solidity": solidity,
            "extent": extent,
            "orientation_deg": float(angle),
            "contour": contour,
            "oriented_bbox": box_points
        }

        regions.append(region_info)
        region_id += 1

    return regions


def compute_shadow_length_px(
    base_point: Tuple[float, float],
    tip_point: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean geometric shadow length in pixels between BASE and TIP endpoints.

    Formula: L_px = sqrt((x_tip - x_base)^2 + (y_tip - y_base)^2)
    """
    dx = float(tip_point[0] - base_point[0])
    dy = float(tip_point[1] - base_point[1])
    return float(np.hypot(dx, dy))


def compute_shadow_directional_geometry(
    image: np.ndarray,
    region: Dict[str, Any],
    sampling_distance: int = 5
) -> Dict[str, Any]:
    """
    Calculate basic major-axis directional geometry for a candidate shadow region.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid NumPy array.")

    h, w = image.shape[:2]
    if image.ndim == 3:
        v_channel = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 2]
    else:
        v_channel = image

    contour = region["contour"]
    cx, cy = region["centroid"]
    angle_deg = region["orientation_deg"]

    angle_rad = np.radians(angle_deg)
    u_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)

    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    centroid_pt = np.array([cx, cy], dtype=np.float64)
    projections = np.dot(contour_pts - centroid_pt, u_vector)

    idx_a = int(np.argmin(projections))
    idx_b = int(np.argmax(projections))

    endpoint_a = (float(contour_pts[idx_a][0]), float(contour_pts[idx_a][1]))
    endpoint_b = (float(contour_pts[idx_b][0]), float(contour_pts[idx_b][1]))

    pa = np.array(endpoint_a, dtype=np.float64)
    pb = np.array(endpoint_b, dtype=np.float64)

    axis_vec = pb - pa
    axis_len = np.linalg.norm(axis_vec)

    if axis_len < 1e-5:
        dir_ab = u_vector
    else:
        dir_ab = axis_vec / axis_len

    sample_pt_a = pa - dir_ab * sampling_distance
    sample_pt_b = pb + dir_ab * sampling_distance

    def extract_patch_stats(center_pt: np.ndarray, radius: int = 3) -> Tuple[float, float]:
        cx_p, cy_p = int(round(center_pt[0])), int(round(center_pt[1]))
        x0, y0 = max(0, cx_p - radius), max(0, cy_p - radius)
        x1, y1 = min(w, cx_p + radius + 1), min(h, cy_p + radius + 1)

        patch = v_channel[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0, 0.0
        mean_val = float(np.mean(patch))
        std_val = float(np.std(patch))
        return mean_val, std_val

    mean_a, std_a = extract_patch_stats(sample_pt_a)
    mean_b, std_b = extract_patch_stats(sample_pt_b)

    evidence_score_a = mean_a + 0.5 * std_a
    evidence_score_b = mean_b + 0.5 * std_b

    if evidence_score_a >= evidence_score_b:
        estimated_base = endpoint_a
        estimated_tip = endpoint_b
        shadow_dir_vec = (float(dir_ab[0]), float(dir_ab[1]))
        object_dir_vec = (float(-dir_ab[0]), float(-dir_ab[1]))
    else:
        estimated_base = endpoint_b
        estimated_tip = endpoint_a
        shadow_dir_vec = (float(-dir_ab[0]), float(-dir_ab[1]))
        object_dir_vec = (float(dir_ab[0]), float(dir_ab[1]))

    denom = evidence_score_a + evidence_score_b + 1e-5
    direction_confidence = float(abs(evidence_score_a - evidence_score_b) / denom)
    direction_confidence = max(0.0, min(1.0, direction_confidence))
    is_ambiguous = bool(direction_confidence < 0.12)

    return {
        "candidate_id": int(region["id"]),
        "centroid": (float(cx), float(cy)),
        "orientation_deg": float(angle_deg),
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "estimated_base_point": estimated_base,
        "estimated_tip_point": estimated_tip,
        "shadow_direction_vector": shadow_dir_vec,
        "object_search_direction_vector": object_dir_vec,
        "direction_confidence": direction_confidence,
        "is_ambiguous": is_ambiguous
    }


def compute_object_shadow_adjacency(
    image: np.ndarray,
    region: Dict[str, Any],
    corridor_width_factor: float = 0.75,
    corridor_length_factor: float = 1.5,
    min_object_threshold: float = 0.25,
    ambiguity_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Phase 3 Object-Shadow Adjacency / Pairing Baseline.
    """
    pairing = compute_object_shadow_pairing(
        image,
        region,
        corridor_width_factor=corridor_width_factor,
        corridor_length_factor=corridor_length_factor
    )
    return {
        "candidate_id": pairing["candidate_id"],
        "centroid": pairing["centroid"],
        "orientation_deg": pairing["orientation_deg"],
        "estimated_object_side": pairing["estimated_object_side"],
        "object_score": pairing["object_score"],
        "object_shadow_adjacency_score": pairing["adjacency_score"],
        "direction_confidence": pairing["adjacency_score"],
        "status": "[GOOD]" if pairing["status"] == "[STRONG PAIR]" else ("[AMBIGUOUS]" if pairing["status"] == "[WEAK PAIR]" else "[WEAK]"),
        "estimated_base_point": pairing["estimated_base_point"],
        "estimated_tip_point": pairing["estimated_tip_point"],
        "shadow_direction_vector": pairing["shadow_direction_vector"],
        "object_search_direction_vector": pairing["object_search_direction_vector"],
        "corridor_a_corners": pairing["corridor_a_corners"],
        "corridor_b_corners": pairing["corridor_b_corners"]
    }


def compute_object_shadow_pairing(
    image: np.ndarray,
    region: Dict[str, Any],
    corridor_width_factor: float = 0.75,
    corridor_length_factor: float = 1.5
) -> Dict[str, Any]:
    """
    Phase 3 Refined OBJECT-SHADOW PAIRING TEST.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid NumPy array.")

    h, w = image.shape[:2]
    if image.ndim == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        v_channel = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 2]
    else:
        gray = image
        v_channel = image

    canny_edges = cv.Canny(gray, 50, 150)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    contour = region["contour"]
    cx, cy = region["centroid"]
    angle_deg = region["orientation_deg"]
    major_axis = region["major_axis_length"]
    minor_axis = region["minor_axis_length"]

    # 1. Determine major-axis endpoints P_A, P_B and unit vectors
    angle_rad = np.radians(angle_deg)
    u_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)
    n_vec = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float64)

    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    centroid_pt = np.array([cx, cy], dtype=np.float64)
    projections = np.dot(contour_pts - centroid_pt, u_vec)

    idx_a = int(np.argmin(projections))
    idx_b = int(np.argmax(projections))

    pa = contour_pts[idx_a]
    pb = contour_pts[idx_b]

    axis_vec = pb - pa
    axis_len = np.linalg.norm(axis_vec)
    dir_ab = u_vec if axis_len < 1e-5 else axis_vec / axis_len

    half_width = float(max(4.0, 0.5 * corridor_width_factor * minor_axis))
    corr_length = float(max(12.0, corridor_length_factor * major_axis))

    # Corridor A (outwards along -dir_ab)
    c_a1 = pa + half_width * n_vec
    c_a2 = pa - half_width * n_vec
    c_a3 = pa - corr_length * dir_ab - half_width * n_vec
    c_a4 = pa - corr_length * dir_ab + half_width * n_vec
    poly_a = np.array([c_a1, c_a2, c_a3, c_a4], dtype=np.int32)

    # Corridor B (outwards along +dir_ab)
    c_b1 = pb + half_width * n_vec
    c_b2 = pb - half_width * n_vec
    c_b3 = pb + corr_length * dir_ab - half_width * n_vec
    c_b4 = pb + corr_length * dir_ab + half_width * n_vec
    poly_b = np.array([c_b1, c_b2, c_b3, c_b4], dtype=np.int32)

    def analyze_corridor(poly_pts: np.ndarray, base_pt: np.ndarray, out_dir: np.ndarray) -> Dict[str, Any]:
        corr_mask = np.zeros((h, w), dtype=np.uint8)
        cv.fillPoly(corr_mask, [poly_pts], 255)
        total_px = int(np.count_nonzero(corr_mask))

        if total_px == 0:
            return {
                "object_score": 0.0,
                "dist_score": 0.0,
                "dir_score": 0.0,
                "bound_score": 0.0,
                "gap_score": 0.0,
                "struct_score": 0.0,
                "obj_location": base_pt
            }

        corr_v = v_channel[corr_mask == 255]
        corr_edges = canny_edges[corr_mask == 255]

        mean_v = float(np.mean(corr_v))
        edge_ratio = float(np.count_nonzero(corr_edges)) / float(total_px)
        max_v = float(np.max(corr_v))

        # Independent ObjectScore
        obj_score = float(0.40 * (mean_v / 255.0) + 0.35 * min(1.0, edge_ratio * 4.0) + 0.25 * max(0.0, (max_v - 50.0) / 205.0))
        obj_score = max(0.0, min(1.0, obj_score))

        # Find strongest Canny / bright edge location inside corridor
        edge_coords = np.argwhere((corr_mask == 255) & (canny_edges > 0))
        if len(edge_coords) > 0:
            edge_pts_xy = edge_coords[:, [1, 0]].astype(np.float64)
            dists = np.linalg.norm(edge_pts_xy - base_pt, axis=1)
            best_idx = int(np.argmin(dists))
            obj_loc = edge_pts_xy[best_idx]
            d_val = float(dists[best_idx])
        else:
            obj_loc = base_pt + out_dir * (corr_length * 0.5)
            d_val = float(corr_length * 0.5)

        # 1. Distance Score (S_dist)
        dist_score = float(max(0.0, 1.0 - (d_val / (1.5 * major_axis + 1e-5))))

        # 2. Direction Alignment Score (S_dir)
        vec_to_obj = obj_loc - base_pt
        norm_to_obj = np.linalg.norm(vec_to_obj)
        if norm_to_obj > 1e-5:
            dir_to_obj = vec_to_obj / norm_to_obj
            dir_score = float(max(0.0, np.dot(dir_to_obj, out_dir)))
        else:
            dir_score = 1.0

        # 3. Boundary Orientation Score (S_bound)
        int_x, int_y = int(round(obj_loc[0])), int(round(obj_loc[1]))
        int_x, int_y = max(0, min(w - 1, int_x)), max(0, min(h - 1, int_y))
        gx = sobel_x[int_y, int_x]
        gy = sobel_y[int_y, int_x]
        grad_norm = np.hypot(gx, gy)
        if grad_norm > 1e-5:
            edge_normal = np.array([gx, gy], dtype=np.float64) / grad_norm
            bound_score = float(abs(np.dot(edge_normal, out_dir)))
        else:
            bound_score = 0.5

        # 4. Gap Continuity Score (S_gap)
        num_samples = 10
        gap_pts_x = np.linspace(base_pt[0], obj_loc[0], num_samples)
        gap_pts_y = np.linspace(base_pt[1], obj_loc[1], num_samples)
        dark_count = 0
        for gx_i, gy_i in zip(gap_pts_x, gap_pts_y):
            ix, iy = max(0, min(w - 1, int(round(gx_i)))), max(0, min(h - 1, int(round(gy_i))))
            if v_channel[iy, ix] < 40:
                dark_count += 1
        gap_score = float(1.0 - (dark_count / float(num_samples)))

        # 5. Structure Size Score (S_struct)
        struct_score = float(min(1.0, (total_px * edge_ratio) / (0.5 * major_axis * minor_axis + 1.0)))

        return {
            "object_score": obj_score,
            "dist_score": dist_score,
            "dir_score": dir_score,
            "bound_score": bound_score,
            "gap_score": gap_score,
            "struct_score": struct_score,
            "obj_location": obj_loc
        }

    res_a = analyze_corridor(poly_a, pa, -dir_ab)
    res_b = analyze_corridor(poly_b, pb, +dir_ab)

    def calc_adj_score(r: Dict[str, Any]) -> float:
        return float(
            0.25 * r["dist_score"] +
            0.25 * r["dir_score"] +
            0.20 * r["bound_score"] +
            0.15 * r["gap_score"] +
            0.15 * r["struct_score"]
        )

    adj_a = calc_adj_score(res_a)
    adj_b = calc_adj_score(res_b)

    pair_score_a = float(np.sqrt(res_a["object_score"] * adj_a))
    pair_score_b = float(np.sqrt(res_b["object_score"] * adj_b))

    if pair_score_a >= pair_score_b:
        estimated_base = (float(pa[0]), float(pa[1]))
        estimated_tip = (float(pb[0]), float(pb[1]))
        object_side = "Side A"
        best_res = res_a
        best_adj = adj_a
        best_pair = pair_score_a
        shadow_dir_vec = (float(dir_ab[0]), float(dir_ab[1]))
        object_dir_vec = (float(-dir_ab[0]), float(-dir_ab[1]))
    else:
        estimated_base = (float(pb[0]), float(pb[1]))
        estimated_tip = (float(pa[0]), float(pa[1]))
        object_side = "Side B"
        best_res = res_b
        best_adj = adj_b
        best_pair = pair_score_b
        shadow_dir_vec = (float(-dir_ab[0]), float(-dir_ab[1]))
        object_dir_vec = (float(dir_ab[0]), float(dir_ab[1]))

    obj_score = best_res["object_score"]
    if obj_score >= 0.50 and best_adj >= 0.50:
        status = "[STRONG PAIR]"
    elif best_pair >= 0.35:
        status = "[WEAK PAIR]"
    else:
        status = "[NO PAIR]"

    # Calculate Euclidean shadow length in pixels
    L_shadow_px = compute_shadow_length_px(estimated_base, estimated_tip)

    return {
        "candidate_id": int(region["id"]),
        "centroid": (float(cx), float(cy)),
        "orientation_deg": float(angle_deg),
        "estimated_object_side": object_side,
        "object_score": obj_score,
        "adjacency_score": best_adj,
        "distance_score": best_res["dist_score"],
        "direction_score": best_res["dir_score"],
        "boundary_score": best_res["bound_score"],
        "gap_score": best_res["gap_score"],
        "structure_score": best_res["struct_score"],
        "final_pair_score": best_pair,
        "status": status,
        "estimated_base_point": estimated_base,
        "estimated_tip_point": estimated_tip,
        "shadow_length_px": L_shadow_px,
        "object_location": (float(best_res["obj_location"][0]), float(best_res["obj_location"][1])),
        "shadow_direction_vector": shadow_dir_vec,
        "object_search_direction_vector": object_dir_vec,
        "corridor_a_corners": poly_a,
        "corridor_b_corners": poly_b
    }
