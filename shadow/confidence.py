"""
M4 Shadow Cue Module - Refined Candidate Confidence Ranking

This module evaluates candidate shadow regions by decoupling two structural properties:
A. Shadow Shape Suitability (S_shape): Prefers coherent, elongated, orientation-aligned shapes.
B. Structural Cleanliness vs. Merge Suspicion (R_struct): Penalizes sprawling, low-solidity merged background regions.

Key Sub-Scores:
1. Shadow Shape Score (S_shape): Combines elongation, orientation consistency, and solidity.
2. Geometry Sub-Score (S_geo): Aspect ratio, elongation, solidity, area weighting.
3. Boundary Contrast Sub-Score (S_contrast): Normalized Contrast Index (NCI) between inner region & outer boundary ring.

Reliability Modifiers:
1. Area Reliability Factor (R_area = 1 - exp(-Area / A_ref)): Scales down noise-prone tiny regions (<50 px).
2. Structural Reliability / Merge Factor (R_struct = 1 - merge_suspicion):
   Discriminates true elongated shadows (where low extent relative to BBox is natural)
   from sprawling low-solidity merged regions (where low extent + low solidity = high complexity).

Experimental Confidence Formula:
  C_raw = w_shape * S_shape + w_contrast * S_contrast + w_geo * S_geo
  Confidence Score C = R_area * R_struct * C_raw

IMPORTANT DISCLAIMER:
- This confidence score represents an experimental relative ranking metric for candidate prioritization.
- It DOES NOT represent ground-truth shadow probability.
- Regions are NOT deleted or hard-filtered based on this score.
"""

from typing import List, Dict, Any, Tuple
import cv2 as cv
import numpy as np


def compute_boundary_contrast(
    gray_or_v_image: np.ndarray,
    contour: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    ring_dilation: int = 3
) -> Tuple[float, float, float, float]:
    """
    Compute inner mean intensity, outer ring mean intensity, raw contrast ratio,
    and a bounded Normalized Contrast Index (NCI) score.
    """
    h, w = gray_or_v_image.shape[:2]
    x, y, bw, bh = bounding_box

    margin = ring_dilation + 4
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)

    crop_img = gray_or_v_image[y0:y1, x0:x1]
    shifted_contour = contour - np.array([x0, y0])

    inner_mask = np.zeros(crop_img.shape, dtype=np.uint8)
    cv.drawContours(inner_mask, [shifted_contour], -1, 255, -1)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * ring_dilation + 1, 2 * ring_dilation + 1))
    dilated_mask = cv.dilate(inner_mask, kernel, iterations=1)
    outer_ring_mask = cv.bitwise_and(dilated_mask, cv.bitwise_not(inner_mask))

    inner_pixels = crop_img[inner_mask == 255]
    outer_pixels = crop_img[outer_ring_mask == 255]

    if len(inner_pixels) == 0 or len(outer_pixels) == 0:
        return 0.0, 0.0, 1.0, 0.0

    mean_inner = float(np.mean(inner_pixels))
    mean_outer = float(np.mean(outer_pixels))

    raw_contrast_ratio = float(mean_outer / (mean_inner + 1e-5))

    if mean_outer + mean_inner > 1e-5:
        nci = (mean_outer - mean_inner) / (mean_outer + mean_inner)
        contrast_score = float(max(0.0, min(1.0, nci)))
    else:
        contrast_score = 0.0

    return mean_inner, mean_outer, raw_contrast_ratio, contrast_score


def compute_area_reliability(area: float, a_ref: float = 100.0) -> float:
    """
    Compute area measurement reliability factor R_area in range (0, 1].
    R_area = 1 - exp(-Area / A_ref)
    """
    r_area = 1.0 - np.exp(-float(area) / float(a_ref))
    return float(max(0.01, min(1.0, r_area)))


def compute_shadow_shape_score(
    elongation: float,
    aspect_ratio: float,
    solidity: float,
    orientation_score: float
) -> float:
    """
    Compute bounded ShadowShapeScore reflecting directional shadow suitability.
    Prefers coherent, elongated, orientation-aligned shapes.
    """
    elong_factor = min(1.0, max(0.0, elongation / 0.60))
    solid_factor = min(1.0, max(0.0, solidity))
    shape_score = 0.40 * elong_factor + 0.30 * orientation_score + 0.30 * solid_factor
    return float(max(0.0, min(1.0, shape_score)))


def compute_complexity_and_merge_suspicion(
    solidity: float,
    extent: float,
    aspect_ratio: float,
    elongation: float,
    area: float
) -> Tuple[float, float, float]:
    """
    Compute region complexity score, merge suspicion factor, and structural reliability multiplier.
    
    Decouples low extent of clean elongated shadows from true merged background regions:
    For thin elongated regions, bounding box extent is naturally low. We compute
    effective_extent = max(extent, elongation * 0.8) so long thin shadows are not false-flagged.
    """
    # 1. Effective extent taking elongation into account
    effective_extent = max(extent, min(1.0, elongation * 0.80))

    # 2. Structural compactness factor
    extent_scaling = min(1.0, max(0.0, effective_extent / 0.40))
    solidity_factor = min(1.0, max(0.0, solidity))
    compactness = solidity_factor * extent_scaling

    # 3. Complexity score (0 = clean coherent region, 1 = complex sprawling region)
    complexity_score = float(max(0.0, min(1.0, 1.0 - compactness)))

    # 4. Merge suspicion scales with area for complex regions
    area_scale = min(1.0, max(0.0, area / 300.0))
    merge_suspicion = float(max(0.0, min(1.0, complexity_score * (0.4 + 0.6 * area_scale))))

    # 5. Structural reliability modifier
    structural_reliability = float(max(0.1, 1.0 - merge_suspicion))

    return complexity_score, merge_suspicion, structural_reliability


def estimate_dominant_orientation(regions: List[Dict[str, Any]]) -> float:
    """
    Estimate dominant orientation among candidate regions using an area-weighted histogram mode.
    """
    valid_angles = []
    weights = []
    for r in regions:
        if r["area"] >= 20.0:
            valid_angles.append(r["orientation_deg"])
            weights.append(r["area"])

    if not valid_angles:
        return 0.0

    hist, bin_edges = np.histogram(valid_angles, bins=18, range=(-90, 90), weights=weights)
    dominant_bin = np.argmax(hist)
    dominant_angle = float((bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0)
    return dominant_angle


def compute_geometry_score(
    area: float,
    elongation: float,
    solidity: float,
    aspect_ratio: float
) -> float:
    """
    Compute smooth geometry score balancing elongation, solidity, and area.
    """
    elongation_subscore = min(1.0, max(0.0, elongation))
    solidity_subscore = min(1.0, max(0.0, solidity))
    area_subscore = min(1.0, max(0.0, area / 200.0))

    geo_score = 0.4 * elongation_subscore + 0.4 * solidity_subscore + 0.2 * area_subscore
    return float(geo_score)


def compute_orientation_score(
    region_orientation_deg: float,
    dominant_orientation_deg: float,
    max_dev_deg: float = 45.0
) -> float:
    """
    Compute angular similarity score to the scene's dominant candidate orientation mode.
    """
    diff = abs(region_orientation_deg - dominant_orientation_deg)
    if diff > 90.0:
        diff = 180.0 - diff
    orient_score = max(0.0, 1.0 - (diff / max_dev_deg))
    return float(orient_score)


def rank_shadow_regions(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
    w_shape: float = 0.40,
    w_contrast: float = 0.40,
    w_geo: float = 0.20,
    a_ref: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Compute refined experimental confidence scores for all candidate regions and rank them.

    Formula:
      C_raw = w_shape * S_shape + w_contrast * S_contrast + w_geo * S_geo
      Confidence Score C = R_area * R_struct * C_raw
      where R_struct = 1 - merge_suspicion

    Does NOT delete or filter any candidate regions.
    """
    if not regions:
        return []

    if image.ndim == 3:
        v_channel = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 2]
    else:
        v_channel = image

    dominant_angle = estimate_dominant_orientation(regions)

    ranked_regions = []

    for r in regions:
        geo_score = compute_geometry_score(
            area=r["area"],
            elongation=r["elongation"],
            solidity=r["solidity"],
            aspect_ratio=r["aspect_ratio"]
        )

        mean_in, mean_out, raw_contrast_ratio, contrast_score = compute_boundary_contrast(
            gray_or_v_image=v_channel,
            contour=r["contour"],
            bounding_box=r["bounding_box"]
        )

        orient_score = compute_orientation_score(
            region_orientation_deg=r["orientation_deg"],
            dominant_orientation_deg=dominant_angle
        )

        shape_score = compute_shadow_shape_score(
            elongation=r["elongation"],
            aspect_ratio=r["aspect_ratio"],
            solidity=r["solidity"],
            orientation_score=orient_score
        )

        area_reliability = compute_area_reliability(r["area"], a_ref=a_ref)

        complexity_score, merge_suspicion, r_struct = compute_complexity_and_merge_suspicion(
            solidity=r["solidity"],
            extent=r["extent"],
            aspect_ratio=r["aspect_ratio"],
            elongation=r["elongation"],
            area=r["area"]
        )

        c_raw = (
            w_shape * shape_score +
            w_contrast * contrast_score +
            w_geo * geo_score
        )

        confidence_score = float(min(1.0, max(0.0, area_reliability * r_struct * c_raw)))

        r_copy = dict(r)
        r_copy["scores"] = {
            "confidence_score": confidence_score,
            "c_raw": float(c_raw),
            "area_reliability": float(area_reliability),
            "structural_reliability": float(r_struct),
            "complexity_score": float(complexity_score),
            "merge_suspicion": float(merge_suspicion),
            "shadow_shape_score": float(shape_score),
            "geometry_score": float(geo_score),
            "contrast_score": float(contrast_score),
            "raw_contrast_ratio": float(raw_contrast_ratio),
            "mean_inner_intensity": float(mean_in),
            "mean_outer_intensity": float(mean_out),
            "orientation_score": float(orient_score),
            "dominant_angle_deg": float(dominant_angle)
        }
        ranked_regions.append(r_copy)

    ranked_regions.sort(key=lambda item: item["scores"]["confidence_score"], reverse=True)

    for rank, item in enumerate(ranked_regions, start=1):
        item["confidence_rank"] = rank

    return ranked_regions
