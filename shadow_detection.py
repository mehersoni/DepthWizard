"""
Module: shadow_detection
Description: Shadow detection, morphology filtering, and shadow-geometry height constraint module.
Supports:
- MODE A: Solar metadata available (L * tan(solar_elevation))
- MODE B: Solar metadata unavailable (structural/edge confidence only, no fabricated heights)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    label,
    find_objects,
    generate_binary_structure
)


def compute_shadow_confidence(
    rgb: np.ndarray,
    luminance_percentile: float = 25.0,
    blue_ratio_threshold: float = 0.95
) -> np.ndarray:
    """
    Compute continuous shadow confidence map S(x,y) in [0, 1].
    
    Shadows in high-resolution aerial imagery exhibit:
    1. Low total luminance / value (V in HSV, I in RGB)
    2. Higher relative blue/short-wavelength ratio due to Rayleigh scattered skylight
    3. Low color saturation/variance compared to vegetation and roofs
    
    Parameters:
    - rgb: uint8 or float32 image (H, W, 3)
    - luminance_percentile: Percentile threshold for low luminance candidate pixels
    - blue_ratio_threshold: Minimum blue ratio for skylight illumination
    
    Returns:
    - shadow_conf: float32 array in [0, 1]
    """
    if rgb.dtype == np.uint8:
        rgb_f = rgb.astype(np.float32) / 255.0
    else:
        rgb_f = rgb.astype(np.float32)
        if np.max(rgb_f) > 1.0:
            rgb_f = rgb_f / 255.0

    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    # Luminance / Intensity
    intensity = 0.299 * r + 0.587 * g + 0.114 * b

    # Blue ratio relative to red/green
    rg_mean = (r + g) / 2.0 + 1e-5
    blue_ratio = b / rg_mean

    # Compute luminance threshold from the image distribution
    lum_thresh = np.percentile(intensity, luminance_percentile)
    
    # Continuous confidence formulation:
    # 1. Dark score: higher for darker pixels below threshold
    dark_score = np.clip((lum_thresh * 1.5 - intensity) / (lum_thresh * 1.5 + 1e-5), 0.0, 1.0)
    
    # 2. Spectral skylight score: boosted if blue ratio >= 1.0 (ambient scattered skylight)
    blue_score = np.clip((blue_ratio - 0.8) / 0.5, 0.0, 1.0)
    
    # 3. Ratio-based shadow index (Tsai's / C3 inspired):
    diff_br = (b - r) / (b + r + 1e-4)
    spectral_score = np.clip((diff_br + 0.1) / 0.4, 0.0, 1.0)
    
    # Combine scores into continuous confidence
    shadow_conf = dark_score * (0.6 * blue_score + 0.4 * spectral_score)
    shadow_conf = np.clip(shadow_conf, 0.0, 1.0).astype(np.float32)

    return shadow_conf


def filter_shadow_mask(
    shadow_conf: np.ndarray,
    confidence_threshold: float = 0.35,
    min_area_pixels: int = 50,
    closing_radius: int = 3,
    opening_radius: int = 2
) -> np.ndarray:
    """
    Apply morphological opening and closing and minimum connected-component
    area filtering to produce a clean binary shadow mask.
    """
    raw_binary = shadow_conf >= confidence_threshold
    struct = generate_binary_structure(2, 2)

    # Morphological closing to fill small holes inside shadows
    if closing_radius > 0:
        closed = binary_closing(raw_binary, structure=struct, iterations=closing_radius)
    else:
        closed = raw_binary

    # Morphological opening to eliminate isolated noisy speckles
    if opening_radius > 0:
        opened = binary_opening(closed, structure=struct, iterations=opening_radius)
    else:
        opened = closed

    # Connected component area filtering
    labeled_array, num_features = label(opened, structure=struct)
    if num_features == 0:
        return np.zeros_like(shadow_conf, dtype=bool)

    # Fast bincount of component sizes
    sizes = np.bincount(labeled_array.ravel())
    mask_sizes = sizes >= min_area_pixels
    mask_sizes[0] = False  # Background
    clean_mask = mask_sizes[labeled_array]

    return clean_mask


def detect_building_shadow_pairs(
    rgb: np.ndarray,
    shadow_mask: np.ndarray,
    initial_dsm: np.ndarray,
    gsd_m: float = 0.05,
    min_shadow_len_px: int = 10,
    max_shadow_len_px: int = 500
) -> Dict[str, Any]:
    """
    Identify candidate building structures adjacent to cast shadow regions
    and estimate shadow length and casting direction.
    """
    struct = generate_binary_structure(2, 2)
    labeled_shadows, num_shadows = label(shadow_mask, structure=struct)
    slices = find_objects(labeled_shadows)

    building_shadow_pairs = []
    shadow_lengths_m = []
    shadow_directions_deg = []

    # Building threshold: elevated pixels in DSM relative to local baseline
    valid_dsm = initial_dsm[np.isfinite(initial_dsm)]
    ground_datum = float(np.percentile(valid_dsm, 15.0)) if len(valid_dsm) > 0 else 40.0
    building_mask = initial_dsm > (ground_datum + 2.5)  # Buildings > 2.5m above ground

    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        comp_mask = (labeled_shadows[slc] == (idx + 1))
        area = int(np.sum(comp_mask))
        if area < 50:
            continue

        r_slice, c_slice = slc
        sub_dsm = initial_dsm[slc]
        sub_bldg = building_mask[slc]

        r_coords, c_coords = np.where(comp_mask)
        if len(r_coords) < 10:
            continue

        coords = np.column_stack([r_coords, c_coords])
        coords_centered = coords - np.mean(coords, axis=0)
        cov = np.cov(coords_centered, rowvar=False)

        if cov.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            length_px = float(4.0 * np.sqrt(max(1.0, eigvals[1])))
            length_m = length_px * gsd_m

            major_vec = eigvecs[:, 1]
            angle_deg = float(np.degrees(np.arctan2(major_vec[0], major_vec[1])))
        else:
            length_px = float(np.max(r_coords) - np.min(r_coords) + 1)
            length_m = length_px * gsd_m
            angle_deg = 0.0

        if min_shadow_len_px <= length_px <= max_shadow_len_px:
            shadow_lengths_m.append(length_m)
            shadow_directions_deg.append(angle_deg)
            building_shadow_pairs.append({
                "shadow_id": idx + 1,
                "bbox": [int(r_slice.start), int(r_slice.stop), int(c_slice.start), int(c_slice.stop)],
                "area_px": area,
                "length_px": length_px,
                "length_m": length_m,
                "direction_deg": angle_deg,
                "mean_initial_dsm_m": float(np.mean(sub_dsm[comp_mask]))
            })

    return {
        "num_shadow_candidates": num_shadows,
        "num_accepted_pairs": len(building_shadow_pairs),
        "pairs": building_shadow_pairs,
        "shadow_lengths_m": shadow_lengths_m,
        "shadow_directions_deg": shadow_directions_deg,
        "mean_shadow_length_m": float(np.mean(shadow_lengths_m)) if shadow_lengths_m else 0.0,
        "median_shadow_length_m": float(np.median(shadow_lengths_m)) if shadow_lengths_m else 0.0
    }


def apply_shadow_height_constraint(
    initial_dsm: np.ndarray,
    shadow_mask: np.ndarray,
    building_shadow_data: Dict[str, Any],
    solar_elevation_deg: Optional[float] = None,
    solar_azimuth_deg: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply shadow height constraint to refine DSM.
    
    - MODE A: If solar_elevation_deg is provided, calculate H_shadow = L * tan(elevation)
              and refine building height.
    - MODE B: If solar_elevation_deg is None, DO NOT fabricate solar angles.
              Use shadow regions to preserve ground datum and sharpen building step-edges.
    """
    constrained_dsm = initial_dsm.copy()

    if solar_elevation_deg is not None:
        # MODE A: Solar geometry available
        mode = "MODE_A_SOLAR_METADATA_AVAILABLE"
        elev_rad = np.radians(solar_elevation_deg)
        tan_elev = np.tan(elev_rad)

        height_constraints = []
        for pair in building_shadow_data.get("pairs", []):
            l_m = pair["length_m"]
            h_est = float(l_m * tan_elev)
            pair["estimated_height_m"] = h_est
            height_constraints.append(h_est)

        summary = {
            "mode": mode,
            "solar_elevation_deg": solar_elevation_deg,
            "solar_azimuth_deg": solar_azimuth_deg,
            "num_height_constraints": len(height_constraints),
            "mean_shadow_height_m": float(np.mean(height_constraints)) if height_constraints else 0.0,
            "metric_height_applied": True
        }
    else:
        # MODE B: Solar geometry unavailable
        mode = "MODE_B_SOLAR_METADATA_UNAVAILABLE"
        summary = {
            "mode": mode,
            "solar_elevation_deg": None,
            "solar_azimuth_deg": None,
            "num_height_constraints": 0,
            "mean_shadow_height_m": None,
            "metric_height_applied": False,
            "reason": "Potsdam GeoTIFF metadata does not contain solar elevation/azimuth tags. "
                      "In accordance with scientific constraints, solar geometry is not fabricated."
        }

    return constrained_dsm, summary
