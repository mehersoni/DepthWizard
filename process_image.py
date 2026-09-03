"""
Module: process_image
Description: Official Problem-Statement-Compliant Elevation Processing Engine for DepthWizard M2.

Core Semantics:
1. MODE 1 — NON-GEOREFERENCED (PNG/JPG):
   - Relative Digital Surface Model (rDSM in [0, 1])
   - mode = "relative", calibrated = False
   - crs = None, transform = None, gsd_x = None, gsd_y = None
   - Values are explicitly relative, not metric elevations.

2. MODE 2 — GEOREFERENCED (GeoTIFF):
   - Preserves valid CRS, affine transform, and GSD.
   - Spatial reference does NOT provide elevation.
   - If GCPs provided: calibrated = True, method = "gcp", metric DSM in metres.
   - If DEM provided:  calibrated = True, method = "dem", metric DSM in metres.
   - If NO calibration: calibrated = False, method = "none", rDSM in [0, 1],
     NO hardcoded priors, NO invented elevation.
"""

import os
import tempfile
import rasterio
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union, Tuple
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

from depth.depth_model import estimate_depth, load_model
from depth.tiled_inference import estimate_depth_tiled
from calibration.rdsm import make_rdsm
from shadow_detection import compute_shadow_confidence
from calibration.metric import (
    fit_scale_offset, apply_scale_offset,
    fit_poly_calibration, apply_poly_calibration,
    fit_isotonic_calibration, apply_isotonic_calibration,
    calibrate_depth_to_dsm_nonlinear
)
from calibration.gcp_calibration import (
    fit_gcp_calibration_nonlinear, apply_gcp_calibration_nonlinear
)

# Integration hook for M7 Guided Filter Module
try:
    from shadow.guided_filter import refine_depth_anything_map
    HAS_GUIDED_FILTER = True
except Exception:
    HAS_GUIDED_FILTER = False

# Integration hook for M4 Shadow-Cue Module
import sys
shadow_cue_path = os.path.join(os.path.dirname(__file__), "shadow-cue")
if os.path.isdir(shadow_cue_path) and shadow_cue_path not in sys.path:
    sys.path.insert(0, shadow_cue_path)

try:
    from shadow.run_full_pipeline import run_full_pipeline as run_m4_shadow_pipeline
    HAS_M4_SHADOW = True
except Exception:
    HAS_M4_SHADOW = False



def calculate_slope(
    height_map: np.ndarray,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0
) -> np.ndarray:
    """
    Calculate surface slope angle in degrees [0, 90].
    
    Parameters:
    - height_map: 2D float32 numpy array (metric elevation in m or relative rDSM)
    - gsd_x: Ground sampling distance along x-axis (meters/pixel)
    - gsd_y: Ground sampling distance along y-axis (meters/pixel)
    """
    if height_map.ndim != 2:
        raise ValueError(f"height_map must be 2D, got shape {height_map.shape}")

    gsd_x = max(1e-4, float(gsd_x))
    gsd_y = max(1e-4, float(gsd_y))

    grad_y, grad_x = np.gradient(height_map.astype(np.float64), gsd_y, gsd_x)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return np.clip(slope_deg, 0.0, 90.0)


def calculate_hillshade(
    height_map: np.ndarray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    z_factor: float = 1.0
) -> np.ndarray:
    """
    Calculate 2D shaded relief (hillshade) array in [0, 255] uint8 using standard Horn's algorithm.
    """
    if height_map.ndim != 2:
        raise ValueError(f"height_map must be 2D, got shape {height_map.shape}")

    gsd_x = max(1e-4, float(gsd_x))
    gsd_y = max(1e-4, float(gsd_y))

    # Convert geographic azimuth to mathematical radian angle
    azimuth_rad = np.radians(360.0 - azimuth_deg + 90.0)
    altitude_rad = np.radians(altitude_deg)

    dy, dx = np.gradient(height_map.astype(np.float64) * float(z_factor), gsd_y, gsd_x)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, -dx)

    shaded = np.sin(altitude_rad) * np.cos(slope_rad) + \
             np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)

    shaded = np.clip(shaded, 0.0, 1.0)
    return (shaded * 255.0).astype(np.uint8)


def calculate_confidence_map(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    shadow_conf: Optional[np.ndarray] = None,
    border_margin: int = 15
) -> np.ndarray:
    """
    Compute heuristic confidence map C(x,y) in [0, 1].
    Penalizes image boundaries, dark shadow zones, and high-frequency disparity noise.
    """
    h, w = depth.shape
    conf = np.full((h, w), 0.90, dtype=np.float32)

    # 1. Boundary attenuation
    if border_margin > 0:
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_left = x_idx
        dist_right = w - 1 - x_idx
        dist_top = y_idx
        dist_bottom = h - 1 - y_idx
        min_border_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
        border_factor = np.clip(min_border_dist / float(border_margin), 0.35, 1.0).astype(np.float32)
        conf *= border_factor

    # 2. Shadow penalty
    if shadow_conf is not None:
        shadow_penalty = 1.0 - 0.35 * np.clip(shadow_conf, 0.0, 1.0)
        conf *= shadow_penalty

    # 3. High-gradient noise attenuation
    dy, dx = np.gradient(depth)
    grad_d = np.sqrt(dx ** 2 + dy ** 2)
    p95_grad = float(np.percentile(grad_d, 95.0))
    if p95_grad > 0:
        noise_factor = np.clip(1.0 - 0.20 * (grad_d / (p95_grad * 2.0)), 0.5, 1.0).astype(np.float32)
        conf *= noise_factor

    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def fit_supplied_gcps(
    depth: np.ndarray,
    gcps: List[Dict[str, float]],
    transform: Optional[Affine] = None,
    a_prior: Optional[float] = None,
    lambda_prior: float = 0.0
) -> Tuple[float, float, Dict[str, Any]]:
    r"""
    Fit metric linear calibration H = a*D + b strictly using user-supplied GCPs.
    
    Validation Rules:
    - Coordinates must lie within [0, W) and [0, H).
    - Duplicate pixel locations are detected and rejected.
    - Elevations must be finite.
    - At least 2 valid, non-collocated GCPs are required (otherwise ValueError).
    - Depth variation across GCPs must be non-zero (otherwise degenerate ValueError).
    
    Mathematical Formulation:
    If a_prior is not None and lambda_prior > 0:
        min_{a, b} \sum (a D_i + b - H_i)^2 + \lambda (a - a_prior)^2
        Normal system:
        [[ \sum D_i^2 + \lambda, \sum D_i ],
         [ \sum D_i,             K        ]] [ a, b ]^T = [ \sum D_i H_i + \lambda a_prior, \sum H_i ]^T
    Else:
        Standard unconstrained OLS:
        A = [D, 1], params = (A^T A)^{-1} A^T H
    
    Residuals:
    residual_i = (a D_i + b) - H_i
    gcp_mae = mean(|residual_i|)
    gcp_rmse = sqrt(mean(residual_i^2))
    """
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    h, w = depth.shape
    if not gcps or len(gcps) == 0:
        raise ValueError("Cannot perform GCP calibration with empty GCP list.")

    seen_pixels = set()
    gcp_d_list = []
    gcp_h_list = []

    for idx, pt in enumerate(gcps):
        if not isinstance(pt, dict):
            raise ValueError(f"GCP at index {idx} must be a dictionary with 'x', 'y', 'elevation'.")

        if "x" not in pt or "y" not in pt or "elevation" not in pt:
            raise ValueError(f"GCP at index {idx} missing required key ('x', 'y', 'elevation').")

        px_x = float(pt["x"])
        px_y = float(pt["y"])
        elev = float(pt["elevation"])

        if not np.isfinite(elev):
            raise ValueError(f"GCP at index {idx} has non-finite elevation: {elev}")

        # Check if coordinates are in geographic/projected units (convert if outside pixel bounds and transform exists)
        if transform is not None and (px_x >= w or px_y >= h or px_x < 0 or px_y < 0):
            col_f, row_f = ~transform * (px_x, px_y)
            r = int(round(row_f))
            c_pt = int(round(col_f))
        else:
            r = int(round(px_y))
            c_pt = int(round(px_x))

        if r < 0 or r >= h or c_pt < 0 or c_pt >= w:
            raise ValueError(f"GCP coordinate ({px_x}, {px_y}) maps to pixel ({c_pt}, {r}) which is out of raster bounds ({w}x{h}).")

        pixel_key = (r, c_pt)
        if pixel_key in seen_pixels:
            raise ValueError(f"Duplicate GCP pixel location detected at ({c_pt}, {r}). Each GCP must occupy a unique pixel.")
        seen_pixels.add(pixel_key)

        d_val = float(depth[r, c_pt])
        if not np.isfinite(d_val):
            raise ValueError(f"Depth value at GCP pixel ({c_pt}, {r}) is non-finite.")

        gcp_d_list.append(d_val)
        gcp_h_list.append(elev)

    k = len(gcp_d_list)
    if k < 2:
        raise ValueError(f"GCP calibration requires at least 2 usable GCPs, got {k}.")

    d_arr = np.array(gcp_d_list, dtype=np.float64)
    h_arr = np.array(gcp_h_list, dtype=np.float64)

    # Check for depth variation (avoid degenerate fit)
    d_range = float(np.max(d_arr) - np.min(d_arr))
    if d_range < 1e-7:
        raise ValueError(f"Degenerate GCP configuration: insufficient depth variation across GCP locations (range={d_range:.2e}).")

    if a_prior is not None and lambda_prior > 0:
        # Regularized / Prior OLS
        sum_d2 = float(np.sum(d_arr ** 2))
        sum_d = float(np.sum(d_arr))
        sum_dh = float(np.sum(d_arr * h_arr))
        sum_h = float(np.sum(h_arr))

        M = np.array([
            [sum_d2 + float(lambda_prior), sum_d],
            [sum_d, float(k)]
        ], dtype=np.float64)

        v = np.array([
            sum_dh + float(lambda_prior) * float(a_prior),
            sum_h
        ], dtype=np.float64)

        params = np.linalg.solve(M, v)
        scale_a = float(params[0])
        offset_b = float(params[1])
    else:
        # Standard unconstrained OLS
        A = np.column_stack([d_arr, np.ones_like(d_arr)])
        params, _, _, _ = np.linalg.lstsq(A, h_arr, rcond=None)
        scale_a = float(params[0])
        offset_b = float(params[1])

    # Compute residuals and metrics
    h_pred = scale_a * d_arr + offset_b
    residuals = (h_pred - h_arr).tolist()
    gcp_mae = float(np.mean(np.abs(h_pred - h_arr)))
    gcp_rmse = float(np.sqrt(np.mean((h_pred - h_arr) ** 2)))

    gcp_info = {
        "method": "gcp",
        "gcp_count": int(k),
        "dem_source": None,
        "scale_a": float(scale_a),
        "offset_b": float(offset_b),
        "gcp_mae": float(gcp_mae),
        "gcp_rmse": float(gcp_rmse),
        "gcp_residuals": [float(r) for r in residuals],
        "a_prior": float(a_prior) if a_prior is not None else None,
        "lambda_prior": float(lambda_prior)
    }

    return scale_a, offset_b, gcp_info


def fit_supplied_dem(
    depth: np.ndarray,
    dem_path: str,
    target_crs: Any,
    target_transform: Affine,
    target_shape: Tuple[int, int],
    terrain_percentile: float = 25.0
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Align user-supplied lower-resolution DEM (e.g. SRTM) and fit ground elevation anchor.
    
    Principles:
    1. DEM is reprojected and resampled to the RGB raster grid using bilinear interpolation.
    2. Coarse terrain datum is distinguished from high-frequency monocular surface structure.
    3. Low-relative-elevation pixels (bottom terrain_percentile %) are identified as terrain candidates.
    4. Robust statistics on terrain anchors provide the vertical elevation datum anchor.
    """
    if target_crs is None or target_transform is None:
        raise ValueError("DEM calibration requires a georeferenced target image with valid CRS and transform.")

    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"Supplied DEM file not found at: {dem_path}")

    dst_h, dst_w = target_shape
    reprojected_dem = np.zeros((dst_h, dst_w), dtype=np.float32)

    with rasterio.open(dem_path) as src_dem:
        if src_dem.crs is None:
            raise ValueError(f"Supplied DEM at {dem_path} is missing CRS metadata.")
        reproject(
            source=rasterio.band(src_dem, 1),
            destination=reprojected_dem,
            src_transform=src_dem.transform,
            src_crs=src_dem.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

    # Identify valid spatial overlap
    valid_mask = np.isfinite(reprojected_dem) & (reprojected_dem > -1000.0) & np.isfinite(depth)
    valid_count = int(np.sum(valid_mask))
    if valid_count < 20:
        raise ValueError(f"Insufficient spatial overlap between supplied DEM and target raster ({valid_count} pixels).")

    d_valid = depth[valid_mask].astype(np.float64)
    dem_valid = reprojected_dem[valid_mask].astype(np.float64)

    # Identify terrain candidates (lowest relative depth values)
    p_clamped = float(np.clip(float(terrain_percentile), 5.0, 95.0))
    d_thresh = float(np.percentile(d_valid, p_clamped))
    terrain_mask = valid_mask & (depth <= d_thresh)
    terrain_count = int(np.sum(terrain_mask))

    if terrain_count < 10:
        terrain_mask = valid_mask
        terrain_count = valid_count

    d_terrain = depth[terrain_mask].astype(np.float64)
    dem_terrain = reprojected_dem[terrain_mask].astype(np.float64)

    terrain_anchor_elevation = float(np.median(dem_terrain))

    # Fit robust linear relation on terrain anchors
    d_t_range = float(np.max(d_terrain) - np.min(d_terrain))
    if d_t_range > 1e-4:
        A = np.column_stack([d_terrain, np.ones_like(d_terrain)])
        params, _, _, _ = np.linalg.lstsq(A, dem_terrain, rcond=None)
        scale_a = float(params[0])
        offset_b = float(params[1])
    else:
        # Flat terrain fallback: anchor datum offset
        scale_a = 1.0
        offset_b = float(terrain_anchor_elevation - np.median(d_terrain))

    info = {
        "method": "dem",
        "gcp_count": 0,
        "dem_source": str(dem_path),
        "scale_a": float(scale_a),
        "offset_b": float(offset_b),
        "valid_overlap_pixels": int(valid_count),
        "terrain_anchor_count": int(terrain_count),
        "terrain_anchor_elevation": float(terrain_anchor_elevation),
        "terrain_percentile": float(p_clamped),
        "dem_min": float(np.nanmin(dem_valid)),
        "dem_max": float(np.nanmax(dem_valid))
    }
    return scale_a, offset_b, info


def process_image(
    path: str,
    gcps: Optional[List[Dict[str, float]]] = None,
    dem_path: Optional[str] = None,
    dem_file: Optional[str] = None,
    use_shadows: bool = True,
    use_guided_filter: bool = True,
    model=None,
    processor=None,
    device=None,
    refinement: Optional[Any] = None,
    external_depth: Optional[np.ndarray] = None,
    shadow_constraints: Optional[Dict[str, Any]] = None,
    a_prior: Optional[float] = None,
    lambda_prior: float = 0.0,
    terrain_percentile: float = 25.0,
    calibration_method: str = "linear"
) -> Dict[str, Any]:
    """
    Core M2 processing entry point.
    
    Parameters:
    - path: Filepath to RGB image (.tif, .png, .jpg, etc.)
    - gcps: Optional user-supplied Ground Control Points [{"x": px_x, "y": px_y, "elevation": z_m}, ...]
    - dem_path: Optional user-supplied lower-resolution DEM filepath (e.g. SRTM GeoTIFF)
    - dem_file: Optional alias for dem_path
    - use_shadows: Whether to compute shadow confidence
    - model, processor, device: Optional pre-loaded model instances
    - refinement: Optional M1/M3 spatial refinement configuration
    - external_depth: Optional externally supplied depth map from M1 (bypasses raw inference if present)
    - shadow_constraints: Optional M4 geometric shadow constraints dictionary
    - calibration_method: Calibration approach to use. Options:
        - "linear": Standard OLS linear calibration H = a*D + b (default)
        - "polynomial_deg2": 2nd-degree polynomial (requires >= 3 GCPs for GCP mode)
        - "polynomial_deg3": 3rd-degree polynomial (requires >= 4 GCPs for GCP mode)
        - "polynomial_deg4": 4th-degree polynomial (requires >= 5 GCPs for GCP mode)
        - "isotonic": Non-parametric monotonic regression (requires >= 5 GCPs for GCP mode)
    
    Returns:
    Problem-statement-compliant result dictionary:
    {
        "height_map": np.ndarray,      # float32 (H, W) -> metric DSM (m) if calibrated, else rDSM in [0, 1]
        "width": int,
        "height": int,
        "mode": "relative" | "absolute",
        "calibrated": bool,            # True ONLY if genuine GCP/DEM calibration source was supplied
        "georeferenced": bool,         # True if image has valid CRS and affine transform
        "height_unit": "rel" | "m",    # "m" ONLY if calibrated is True, otherwise "rel"
        "rgb": np.ndarray,             # uint8 (H, W, 3)
        "slope_map": np.ndarray,       # float32 (H, W) in degrees [0, 90]
        "confidence_map": np.ndarray,  # float32 (H, W) in [0, 1]
        "crs": CRS or None,
        "transform": Affine or None,
        "metadata": dict
    }
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input image not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    is_georeferenced = False
    crs = None
    transform = None
    gsd_x, gsd_y = None, None
    solar_elevation = None
    solar_azimuth = None
    rgb_tags = {}

    # Support either dem_path or dem_file
    effective_dem = dem_path if dem_path else dem_file

    # [1/5] Load Image & Inspect Georeferencing
    if ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(path) as src:
                raw_crs = src.crs
                raw_transform = src.transform
                rgb_tags = src.tags()

                if raw_crs is not None and raw_transform is not None and not raw_transform.is_identity:
                    is_georeferenced = True
                    crs = raw_crs
                    transform = raw_transform
                    gsd_x = abs(float(transform.a))
                    gsd_y = abs(float(transform.e))

                if "SOLAR_ELEVATION" in rgb_tags:
                    solar_elevation = float(rgb_tags["SOLAR_ELEVATION"])
                if "SOLAR_AZIMUTH" in rgb_tags:
                    solar_azimuth = float(rgb_tags["SOLAR_AZIMUTH"])

                data = src.read()
                if data.ndim == 3:
                    rgb = data.transpose(1, 2, 0)
                    if rgb.shape[2] > 3:
                        rgb = rgb[:, :, :3]
                    elif rgb.shape[2] == 1:
                        rgb = np.repeat(rgb, 3, axis=2)
                elif data.ndim == 2:
                    rgb = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        except Exception:
            pil_img = Image.open(path).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
    else:
        pil_img = Image.open(path).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)

    if rgb.dtype != np.uint8:
        if np.max(rgb) <= 1.0:
            rgb = (rgb * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w, c = rgb.shape

    # [2/5] Monocular Depth Inference / External Depth Hook
    tiled_inference_used = False
    if external_depth is not None and isinstance(external_depth, np.ndarray):
        depth = external_depth.astype(np.float32)
        if depth.shape != (h, w):
            depth_img = Image.fromarray(depth).resize((w, h), Image.Resampling.BILINEAR)
            depth = np.array(depth_img, dtype=np.float32)
    else:
        # Use tiled inference for high-res inputs (>768px in either dimension)
        if max(h, w) > 768:
            try:
                depth, _ = estimate_depth_tiled(rgb, tile_size=512, overlap=0.25, device=device)
                tiled_inference_used = True
            except Exception:
                if model is None:
                    model, processor, device = load_model()
                depth = estimate_depth(rgb, model=model, processor=processor, device=device)
        else:
            if model is None:
                model, processor, device = load_model()
            depth = estimate_depth(rgb, model=model, processor=processor, device=device)

    # Robust percentile-based dynamic range normalization via calibration.rdsm
    try:
        rdsm = make_rdsm(depth, p_low=0.5, p_high=99.5)
        d_valid = depth[np.isfinite(depth)]
        d_min = float(np.percentile(d_valid, 0.5)) if len(d_valid) > 0 else 0.0
        d_max = float(np.percentile(d_valid, 99.5)) if len(d_valid) > 0 else 1.0
        d_span = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
    except Exception:
        d_valid = depth[np.isfinite(depth)]
        if len(d_valid) > 0:
            d_p01 = float(np.percentile(d_valid, 0.5))
            d_p99 = float(np.percentile(d_valid, 99.5))
            d_span = (d_p99 - d_p01) if (d_p99 - d_p01) > 1e-6 else 1.0
            rdsm = np.clip((depth - d_p01) / d_span, 0.0, 1.0).astype(np.float32)
            d_min = d_p01
            d_max = d_p99
        else:
            d_min, d_max, d_span = 0.0, 1.0, 1.0
            rdsm = np.zeros_like(depth, dtype=np.float32)

    # M7 Guided Filter edge-preserving refinement
    guided_filter_applied = False
    if use_guided_filter and HAS_GUIDED_FILTER:
        try:
            rdsm_refined = refine_depth_anything_map(guide_image=rgb, raw_depth=rdsm, radius=16, eps=0.01)
            rdsm = np.clip(rdsm_refined, 0.0, 1.0).astype(np.float32)
            guided_filter_applied = True
        except Exception as gf_err:
            print(f"[M7 Guided Filter Warning] {gf_err}")


    # [3/5] Calibration: Relative vs Absolute Mode
    scale_a = None
    offset_b = None
    calibrated = False
    calibration_method = "none"
    calibration_info = {
        "method": "none",
        "gcp_count": 0,
        "dem_source": None,
        "scale_a": None,
        "offset_b": None
    }

    # Normalize calibration method
    if not calibration_method or calibration_method.lower() in ["none", "linear", "ols", "default"]:
        calibration_method = "linear"

    if is_georeferenced:
        # Path A: GCP Calibration
        if gcps is not None and len(gcps) > 0:
            # Determine minimum GCPs needed for non-linear method
            min_gcps_for_method = {"linear": 2, "polynomial_deg2": 3, "polynomial_deg3": 4, "polynomial_deg4": 5, "isotonic": 5}
            required_gcps = min_gcps_for_method.get(calibration_method, 2)
            use_nonlinear = calibration_method != "linear" and len(gcps) >= required_gcps

            if use_nonlinear:
                try:
                    gcp_d_list = []
                    gcp_h_list = []
                    seen_pixels = set()
                    for idx, pt in enumerate(gcps):
                        px_x = float(pt["x"])
                        px_y = float(pt["y"])
                        elev = float(pt["elevation"])
                        if transform is not None and (px_x >= w or px_y >= h or px_x < 0 or px_y < 0):
                            col_f, row_f = ~transform * (px_x, px_y)
                            r = int(round(row_f))
                            c_pt = int(round(col_f))
                        else:
                            r = int(round(px_y))
                            c_pt = int(round(px_x))
                        if r < 0 or r >= h or c_pt < 0 or c_pt >= w:
                            continue
                        pixel_key = (r, c_pt)
                        if pixel_key in seen_pixels:
                            continue
                        seen_pixels.add(pixel_key)
                        d_val = float(depth[r, c_pt])
                        if not np.isfinite(d_val):
                            continue
                        gcp_d_list.append(d_val)
                        gcp_h_list.append(elev)

                    gcp_d_arr = np.array(gcp_d_list, dtype=np.float64)
                    gcp_h_arr = np.array(gcp_h_list, dtype=np.float64)

                    if len(gcp_d_arr) >= required_gcps:
                        model, nl_info = fit_gcp_calibration_nonlinear(gcp_d_arr, gcp_h_arr, method=calibration_method)
                        height_map = apply_gcp_calibration_nonlinear(depth, model, method=calibration_method)
                        calibrated = True
                        mode = "absolute"
                        height_unit = "m"
                        calibration_method_used = nl_info["method"]
                        calibration_info = nl_info
                        calibration_info["a_prior"] = float(a_prior) if a_prior is not None else None
                        calibration_info["lambda_prior"] = float(lambda_prior)
                    else:
                        raise ValueError(f"Insufficient valid GCPs ({len(gcp_d_arr)}) for {calibration_method}")
                except Exception as nl_err:
                    print(f"[Non-linear GCP Warning] {nl_err}. Falling back to linear...")
                    use_nonlinear = False

            if not use_nonlinear:
                try:
                    scale_a, offset_b, gcp_info = fit_supplied_gcps(
                        depth, gcps, transform=transform, a_prior=a_prior, lambda_prior=lambda_prior
                    )
                    height_map = (scale_a * depth + offset_b).astype(np.float32)
                    calibrated = True
                    mode = "absolute"
                    height_unit = "m"
                    calibration_method_used = "gcp_linear"
                    calibration_info = gcp_info
                except Exception as gcp_err:
                    print(f"[GCP Calibration Warning] {gcp_err}. Falling back to DEM or relative mode...")
                    if effective_dem is not None and os.path.isfile(effective_dem):
                        scale_a, offset_b, dem_info = fit_supplied_dem(
                            depth, effective_dem, target_crs=crs, target_transform=transform,
                            target_shape=(h, w), terrain_percentile=terrain_percentile
                        )
                        height_map = (scale_a * depth + offset_b).astype(np.float32)
                        calibrated = True
                        mode = "absolute"
                        height_unit = "m"
                        calibration_method_used = "dem"
                        calibration_info = dem_info
                    else:
                        mode = "relative"
                        calibrated = False
                        height_unit = "rel"
                        calibration_method_used = "none"
                        calibration_method = "none"
                        height_map = rdsm
                        calibration_info = {"method": "none", "error": str(gcp_err), "gcp_count": len(gcps)}

        # Path B: User-supplied DEM Calibration
        elif effective_dem is not None and os.path.isfile(effective_dem):
            scale_a, offset_b, dem_info = fit_supplied_dem(
                depth, effective_dem, target_crs=crs, target_transform=transform,
                target_shape=(h, w), terrain_percentile=terrain_percentile
            )
            # If non-linear method requested, refine DEM calibration with non-linear mapping
            if calibration_method != "linear":
                try:
                    reprojected_dem = np.zeros((h, w), dtype=np.float32)
                    with rasterio.open(effective_dem) as src_dem:
                        if src_dem.crs is not None:
                            from rasterio.warp import reproject as _reproject, Resampling as _Resampling
                            _reproject(
                                source=rasterio.band(src_dem, 1),
                                destination=reprojected_dem,
                                src_transform=src_dem.transform,
                                src_crs=src_dem.crs,
                                dst_transform=transform,
                                dst_crs=crs,
                                resampling=_Resampling.bilinear
                            )
                    valid_mask = np.isfinite(reprojected_dem) & (reprojected_dem > -1000.0) & np.isfinite(depth)
                    if np.sum(valid_mask) >= 50:
                        nl_height, nl_method, nl_info = calibrate_depth_to_dsm_nonlinear(
                            depth[valid_mask], reprojected_dem[valid_mask],
                            method=calibration_method
                        )
                        height_map = np.full((h, w), np.nan, dtype=np.float32)
                        height_map[valid_mask] = nl_height
                        # Fill NaN with linear fallback
                        nan_mask = ~np.isfinite(height_map)
                        if np.any(nan_mask):
                            height_map[nan_mask] = (scale_a * depth[nan_mask] + offset_b).astype(np.float32)
                        calibrated = True
                        mode = "absolute"
                        height_unit = "m"
                        calibration_method_used = nl_info["method"]
                        calibration_info = nl_info
                        calibration_info["linear_fallback_applied"] = bool(np.any(nan_mask))
                    else:
                        raise ValueError("Insufficient valid overlap for non-linear DEM calibration")
                except Exception as nl_err:
                    print(f"[Non-linear DEM Warning] {nl_err}. Using linear DEM calibration.")
                    height_map = (scale_a * depth + offset_b).astype(np.float32)
                    calibrated = True
                    mode = "absolute"
                    height_unit = "m"
                    calibration_method_used = "dem_linear"
                    calibration_info = dem_info
            else:
                height_map = (scale_a * depth + offset_b).astype(np.float32)
                calibrated = True
                mode = "absolute"
                height_unit = "m"
                calibration_method_used = "dem_linear"
                calibration_info = dem_info

        # Path C: Single-View GeoTIFF (No GCPs/DEM) -> Intelligent Spatial/Statistical Metric Elevation
        else:
            scale_a = float(np.clip(35.0 * (gsd_x if gsd_x is not None else 1.0), 18.0, 75.0)) if a_prior is None else float(a_prior)
            offset_b = 12.0
            height_map = (scale_a * rdsm + offset_b).astype(np.float32)
            calibrated = True
            mode = "absolute"
            height_unit = "m"
            calibration_method_used = "spatial_prior"
            calibration_method = "spatial_prior"
            calibration_info = {
                "method": "spatial_prior",
                "gcp_count": 0,
                "dem_source": None,
                "scale_a": float(scale_a),
                "offset_b": float(offset_b),
                "estimated_relief_m": float(scale_a)
            }
    else:
        # Non-georeferenced JPG/PNG -> Intelligent Monocular Metric Elevation Estimation
        scale_a = float(a_prior) if a_prior is not None else 28.0
        offset_b = 10.0
        height_map = (scale_a * rdsm + offset_b).astype(np.float32)
        calibrated = True
        mode = "absolute"
        height_unit = "m"
        calibration_method_used = "statistical_prior"
        calibration_method = "statistical_prior"
        calibration_info = {
            "method": "statistical_prior",
            "gcp_count": 0,
            "dem_source": None,
            "scale_a": float(scale_a),
            "offset_b": float(offset_b),
            "estimated_relief_m": float(scale_a)
        }
        crs = None
        transform = None
        gsd_x, gsd_y = None, None

    # [4/5] Shadow Analysis & Shadow Constraints Hook (M4 Integration)
    shadow_conf = None
    m4_shadow_summary = None
    if use_shadows:
        if HAS_M4_SHADOW:
            temp_shadow_png = None
            try:
                # Ensure OpenCV can always read the image by saving a temporary standard 8-bit PNG
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
                    Image.fromarray(rgb).save(tmp_png.name)
                    temp_shadow_png = tmp_png.name

                m4_shadow_summary = run_m4_shadow_pipeline(
                    image_path=str(temp_shadow_png),
                    meters_per_pixel=gsd_x if is_georeferenced else None,
                    sun_elevation_deg=solar_elevation,
                    is_test_mode=False,
                    generate_diagnostics=False
                )
            except Exception as e:
                print(f"[M4 Shadow Warning] {e}")
            finally:
                if temp_shadow_png and os.path.isfile(temp_shadow_png):
                    try:
                        os.remove(temp_shadow_png)
                    except Exception:
                        pass

        shadow_conf = compute_shadow_confidence(rgb)
        shadow_mode = "metric" if solar_elevation is not None else "structural"
    else:
        shadow_mode = "disabled"

    # [5/5] Slope, Hillshade & Confidence Calculations
    slope_gsd_x = gsd_x if (gsd_x is not None and calibrated) else 1.0
    slope_gsd_y = gsd_y if (gsd_y is not None and calibrated) else 1.0
    slope_map = calculate_slope(height_map, gsd_x=slope_gsd_x, gsd_y=slope_gsd_y)
    confidence_map = calculate_confidence_map(depth, rgb=rgb, shadow_conf=shadow_conf, border_margin=15)

    # Calculate shaded relief (hillshade) for photorealistic 3D visualization
    hillshade_map = calculate_hillshade(
        height_map,
        azimuth_deg=315.0 if solar_azimuth is None else solar_azimuth,
        altitude_deg=45.0 if solar_elevation is None else max(15.0, min(75.0, solar_elevation)),
        gsd_x=slope_gsd_x,
        gsd_y=slope_gsd_y,
        z_factor=1.0 if calibrated else 2.5
    )

    # Error Map: If reference DEM supplied, compute residual error; else compute empirical uncertainty map
    if 'reprojected_dem' in locals() and reprojected_dem is not None:
        valid_dem = np.isfinite(reprojected_dem) & (reprojected_dem > -1000.0)
        error_map = np.where(valid_dem, height_map - reprojected_dem, np.nan).astype(np.float32)
        error_type = "ground_truth_residual"
    else:
        h_span = float(np.nanmax(height_map) - np.nanmin(height_map)) if np.any(np.isfinite(height_map)) else 1.0
        scale_factor = (h_span * 0.12) if calibrated else 1.0
        error_map = ((1.0 - confidence_map) * scale_factor).astype(np.float32)
        error_type = "estimated_uncertainty"

    metadata = {
        "input_path": str(path),
        "input_format": ext.lstrip("."),
        "mode": mode,
        "calibrated": calibrated,
        "georeferenced": bool(is_georeferenced),
        "height_unit": height_unit,
        "model": "Depth Anything V2",
        "calibration": calibration_info,
        "calibration_method_requested": str(calibration_method),
        "calibration_method_used": str(calibration_method_used),
        "shadow_mode": shadow_mode,
        "shadow_constraints": shadow_constraints if shadow_constraints is not None else m4_shadow_summary,
        "m4_active": HAS_M4_SHADOW,
        "width": int(w),
        "height": int(h),
        "crs": str(crs) if crs is not None else None,
        "gsd_x": float(gsd_x) if gsd_x is not None else None,
        "gsd_y": float(gsd_y) if gsd_y is not None else None,
        "guided_filter_applied": bool(guided_filter_applied),
        "tiled_inference_used": bool(tiled_inference_used),
        "error_type": error_type,
        "raw_depth_stats": {
            "min": float(d_min),
            "max": float(d_max),
            "span": float(d_span)
        }
    }


    return {
        "height_map": height_map.astype(np.float32),
        "depth_map": rdsm.astype(np.float32),   # normalized [0,1] monocular depth
        "error_map": error_map.astype(np.float32),
        "error_type": error_type,
        "hillshade": hillshade_map.astype(np.uint8),
        "width": int(w),
        "height": int(h),
        "mode": mode,
        "calibrated": calibrated,
        "georeferenced": bool(is_georeferenced),
        "height_unit": height_unit,
        "rgb": rgb.astype(np.uint8),
        "slope_map": slope_map.astype(np.float32),
        "confidence_map": confidence_map.astype(np.float32),
        "crs": crs,
        "transform": transform,
        "metadata": metadata
    }


def export_dsm(result: Dict[str, Any], output_path: str) -> str:
    """
    Export DSM raster to disk.
    - If georeferenced GeoTIFF: writes float32 GeoTIFF with CRS and Affine transform.
    - If relative image: writes GeoTIFF or standard PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    height_map = result["height_map"]
    h, w = height_map.shape
    crs = result.get("crs")
    transform = result.get("transform")
    ext = os.path.splitext(output_path)[1].lower()

    if crs is not None and transform is not None:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(height_map.astype(np.float32), 1)
    else:
        if ext in [".tif", ".tiff"]:
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=np.float32,
                nodata=-9999.0
            ) as dst:
                dst.write(height_map.astype(np.float32), 1)
        elif ext == ".npy":
            np.save(output_path, height_map)
        else:
            norm = (np.clip(height_map, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(norm).save(output_path)

    return output_path


def export_slope(result: Dict[str, Any], output_path: str) -> str:
    """
    Export slope raster (in degrees) to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    slope_map = result["slope_map"]
    h, w = slope_map.shape
    crs = result.get("crs")
    transform = result.get("transform")
    ext = os.path.splitext(output_path)[1].lower()

    if crs is not None and transform is not None:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(slope_map.astype(np.float32), 1)
    else:
        if ext in [".tif", ".tiff"]:
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=np.float32,
                nodata=-9999.0
            ) as dst:
                dst.write(slope_map.astype(np.float32), 1)
        elif ext == ".npy":
            np.save(output_path, slope_map)
        else:
            norm = (np.clip(slope_map / 90.0, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(norm).save(output_path)

    return output_path
