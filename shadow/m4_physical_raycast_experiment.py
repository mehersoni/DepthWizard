"""
M4 Isolated Physical Shadow-Tip Raycast Module (Refined Version)

Performs physically-constrained raycast shadow measurement for building height estimation.
Key features:
1. Base point selection strictly at building boundary points contacting valid shadow label pixels.
2. Direct outward raycasting along PCA shadow vector with strict local continuity (max 2px gap).
3. Local relative intensity step & positive gradient transition detection (terminates at local contrast step).
4. Physical maximum search distance derived from solar elevation and maximum plausible height.
5. Rejection/low-confidence assignment when clear physical shadow tip transition is absent.
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import cv2 as cv
import numpy as np


def measure_building_shadow_m4_physical(
    building_contour: np.ndarray,
    cleaned_mask: np.ndarray,
    shadow_direction: Tuple[float, float],
    meters_per_pixel: float = 0.05,
    sun_elevation_deg: float = 41.8,
    image_v_channel: Optional[np.ndarray] = None,
    max_plausible_height_m: float = 40.0,
    max_gap_allowed_px: int = 2  # Strict: max 2 contiguous non-shadow pixels allowed (~0.10m)
) -> Dict[str, Any]:
    """
    Measures shadow length for a building contour using isolated M4 physical raycasting.
    """
    if building_contour is None or len(building_contour) == 0:
        return {
            "shadow_length_px": 0.0,
            "shadow_length_m": 0.0,
            "base_point": (0.0, 0.0),
            "tip_point": (0.0, 0.0),
            "status": "REJECTED",
            "confidence": 0.0,
            "termination_reason": "INVALID_CONTOUR",
            "rejection_reason": "Invalid building contour"
        }

    u_x, u_y = shadow_direction
    norm = math.hypot(u_x, u_y)
    if norm < 1e-5:
        u_x, u_y = 1.0, 0.0
    else:
        u_x, u_y = u_x / norm, u_y / norm

    # Perpendicular unit vector
    v_x, v_y = -u_y, u_x

    h_img, w_img = cleaned_mask.shape
    b_pts = building_contour.reshape(-1, 2).astype(np.float64)

    # Label connected components of cleaned shadow mask
    num_labels, labels = cv.connectedComponents(cleaned_mask, connectivity=8)

    # -------------------------------------------------------------------------
    # 1. PHYSICAL MAXIMUM SEARCH LIMIT
    # Max shadow length L_max = H_max / tan(elevation)
    # -------------------------------------------------------------------------
    if sun_elevation_deg > 1.0:
        tan_elev = math.tan(math.radians(sun_elevation_deg))
        l_max_m = max_plausible_height_m / max(0.01, tan_elev)
    else:
        l_max_m = 40.0

    l_max_px = int(round(l_max_m / max(0.001, meters_per_pixel)))
    l_max_px = min(600, max(10, l_max_px))

    # -------------------------------------------------------------------------
    # 2. BUILDING-SHADOW CONTACT & BASE POINT SELECTION
    # Select base points ONLY on building contour points directly contacting shadow pixels
    # -------------------------------------------------------------------------
    contact_candidates = []
    for idx, pt in enumerate(b_pts):
        px, py = pt[0], pt[1]
        contact_labels = []
        for d in range(1, 6):
            qx = int(round(px + d * u_x))
            qy = int(round(py + d * u_y))
            if 0 <= qx < w_img and 0 <= qy < h_img:
                lbl = labels[qy, qx]
                if lbl > 0:
                    contact_labels.append(lbl)
        if len(contact_labels) > 0:
            unique_lbls, counts = np.unique(contact_labels, return_counts=True)
            dom_lbl = unique_lbls[np.argmax(counts)]
            contact_candidates.append((px, py, int(dom_lbl), len(contact_labels)))

    if not contact_candidates:
        b_par = b_pts[:, 0] * u_x + b_pts[:, 1] * u_y
        base_idx = int(np.argmax(b_par))
        p0_x, p0_y = float(b_pts[base_idx, 0]), float(b_pts[base_idx, 1])
        return {
            "shadow_length_px": 0.0,
            "shadow_length_m": 0.0,
            "base_point": (p0_x, p0_y),
            "tip_point": (p0_x, p0_y),
            "status": "REJECTED",
            "confidence": 0.0,
            "termination_reason": "NO_BUILDING_SHADOW_CONTACT",
            "rejection_reason": "No building-shadow contact"
        }

    # -------------------------------------------------------------------------
    # 3. DIRECT OUTWARD RAYCASTING WITH LOCAL GRADIENT & RELATIVE STEP TERMINATION
    # -------------------------------------------------------------------------
    best_ray = None
    best_score = -1.0

    for p0_x, p0_y, dom_lbl, c_hits in contact_candidates:
        consecutive_gaps = 0
        last_supported_t = 0
        supported_pixels = 0
        termination_reason = "REACHED_PHYSICAL_MAX_LIMIT"

        # Sample initial base shadow V-channel values near base (t=1..5)
        v_base_samples = []
        if image_v_channel is not None:
            for dt in range(1, 6):
                rx_s = int(round(p0_x + dt * u_x))
                ry_s = int(round(p0_y + dt * u_y))
                if 0 <= rx_s < w_img and 0 <= ry_s < h_img:
                    v_base_samples.append(float(image_v_channel[ry_s, rx_s]))
        
        v_base_min = float(np.percentile(v_base_samples, 20)) if v_base_samples else 50.0
        # Relative step threshold (+15 units above shadow minimum)
        v_step_thresh = v_base_min + 15.0

        for t in range(1, l_max_px + 1):
            is_supported = False
            for k in [-1, 0, 1]:
                rx = int(round(p0_x + t * u_x + k * v_x))
                ry = int(round(p0_y + t * u_y + k * v_y))
                if 0 <= rx < w_img and 0 <= ry < h_img:
                    lbl_here = labels[ry, rx]
                    if lbl_here == dom_lbl:
                        is_supported = True
                        break

            if is_supported:
                consecutive_gaps = 0
                last_supported_t = t
                supported_pixels += 1
            else:
                consecutive_gaps += 1
                if consecutive_gaps > max_gap_allowed_px:
                    termination_reason = "SHADOW_SUPPORT_LOST"
                    break

            # Local Intensity Transition & Gradient Detection
            if image_v_channel is not None and t > 2:
                rx_c = int(round(p0_x + t * u_x))
                ry_c = int(round(p0_y + t * u_y))
                if 0 <= rx_c < w_img and 0 <= ry_c < h_img:
                    v_curr = float(image_v_channel[ry_c, rx_c])
                    
                    # 1) Relative intensity step (+15 above base shadow)
                    if v_curr >= v_step_thresh:
                        termination_reason = "RELATIVE_INTENSITY_STEP"
                        last_supported_t = t - 1
                        break

                    # 2) Positive local forward gradient step (v_curr_plus2 - v_curr >= 12)
                    rx_p2 = int(round(p0_x + (t + 2) * u_x))
                    ry_p2 = int(round(p0_y + (t + 2) * u_y))
                    if 0 <= rx_p2 < w_img and 0 <= ry_p2 < h_img:
                        v_next2 = float(image_v_channel[ry_p2, rx_p2])
                        if (v_next2 - v_curr) >= 12.0:
                            termination_reason = "LOCAL_GRADIENT_STEP"
                            last_supported_t = t
                            break

        l_px = float(max(0, last_supported_t))
        l_m = l_px * meters_per_pixel

        # Score ray quality
        density = supported_pixels / max(1.0, l_px)
        ray_score = (l_px * density) + (c_hits * 1.5)

        if ray_score > best_score:
            best_score = ray_score
            p_tip_x = float(p0_x + l_px * u_x)
            p_tip_y = float(p0_y + l_px * u_y)

            # Determine confidence & status
            if l_m < 0.5:
                status = "REJECTED"
                conf = 0.0
                rej_reason = "Shadow length below 0.5m minimum physical threshold"
            elif termination_reason == "REACHED_PHYSICAL_MAX_LIMIT" and l_m > 15.0:
                # Reached physical search cap without finding a clear local transition
                status = "LOW CONFIDENCE"
                conf = 0.3
                rej_reason = "Reached search limit without clear local shadow-tip transition"
            elif density < 0.6 or supported_pixels < 3:
                status = "LOW CONFIDENCE"
                conf = 0.4
                rej_reason = "Low shadow density support along ray"
            else:
                status = "VALID"
                conf = float(min(1.0, density))
                rej_reason = None

            best_ray = {
                "shadow_length_px": l_px,
                "shadow_length_m": l_m,
                "base_point": (p0_x, p0_y),
                "tip_point": (p_tip_x, p_tip_y),
                "supporting_pixels": supported_pixels,
                "density": float(density),
                "status": status,
                "confidence": conf,
                "termination_reason": termination_reason,
                "rejection_reason": rej_reason
            }

    if best_ray is None:
        p0_x, p0_y = contact_candidates[0][0], contact_candidates[0][1]
        return {
            "shadow_length_px": 0.0,
            "shadow_length_m": 0.0,
            "base_point": (p0_x, p0_y),
            "tip_point": (p0_x, p0_y),
            "status": "REJECTED",
            "confidence": 0.0,
            "termination_reason": "NO_CONTINUOUS_SHADOW_SUPPORT",
            "rejection_reason": "No continuous shadow support"
        }

    return best_ray
