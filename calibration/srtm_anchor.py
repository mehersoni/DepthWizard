"""
Module: calibration.srtm_anchor
Description: Scene-Specific SRTM / Global DEM Coarse Terrain Elevation Anchoring.

The SIH26175 specification allows lower-resolution DEM sources (e.g. SRTM / Copernicus DEM)
to be used as coarse ground control elevation anchors to resolve scene-specific scale and offset
without requiring high-resolution LiDAR reference DSMs.
"""

import os
import io
import math
import urllib.request
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds
from sklearn.linear_model import HuberRegressor, RANSACRegressor


def fetch_srtm_dem(
    raster_path: str,
    zoom: int = 14,
    cache_dir: str = "data/dem_cache"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fetch and reproject coarse SRTM / Copernicus terrain elevation to match
    the exact geospatial grid (CRS, affine transform, dimensions) of the target raster.
    """
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(raster_path))[0]
    cached_dem_path = os.path.join(cache_dir, f"{basename}_srtm_dem.tif")

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        dst_h, dst_w = src.height, src.width
        dst_transform = src.transform
        dst_crs = src.crs

    # Check local cache first
    if os.path.exists(cached_dem_path):
        with rasterio.open(cached_dem_path) as src_cache:
            if src_cache.shape == (dst_h, dst_w) and src_cache.crs == dst_crs:
                dem_data = src_cache.read(1).astype(np.float32)
                meta = {
                    "source": "AWS Terrain Open Data (SRTM 30m / USGS / Copernicus DEM)",
                    "cached": True,
                    "cache_path": cached_dem_path,
                    "resolution_m": 9.5,
                    "crs": str(dst_crs),
                    "dimensions": [dst_h, dst_w],
                    "valid_pixels": int(np.sum(np.isfinite(dem_data))),
                    "min_elevation_m": float(np.nanmin(dem_data)),
                    "max_elevation_m": float(np.nanmax(dem_data)),
                    "mean_elevation_m": float(np.nanmean(dem_data))
                }
                return dem_data, meta

    # Transform bounding box to WGS84 for tile coordinate calculation
    wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
    min_lon, min_lat, max_lon, max_lat = wgs_bounds

    n = 2.0 ** zoom

    def deg2num(lat_deg, lon_deg):
        lat_rad = math.radians(lat_deg)
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    x0, y0 = deg2num(max_lat, min_lon)
    x1, y1 = deg2num(min_lat, max_lon)

    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    tiles = {}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{zoom}/{x}/{y}.png"
            req = urllib.request.Request(url, headers={"User-Agent": "DepthWizard/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                arr = np.array(img, dtype=np.float32)
                # Decode Terrarium format: elevation = (R * 256 + G + B / 256) - 32768
                elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                tiles[(x, y)] = elev

    num_x = x_max - x_min + 1
    num_y = y_max - y_min + 1
    stitched = np.zeros((num_y * 256, num_x * 256), dtype=np.float32)
    for (x, y), data in tiles.items():
        iy = (y - y_min) * 256
        ix = (x - x_min) * 256
        stitched[iy : iy + 256, ix : ix + 256] = data

    def num2merc(x, y, z):
        lon_deg = x / (2.0 ** z) * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / (2.0 ** z))))
        lat_deg = math.degrees(lat_rad)
        x_m = lon_deg * 20037508.34 / 180.0
        y_m = math.log(math.tan((90.0 + lat_deg) * math.pi / 360.0)) / (math.pi / 180.0) * 20037508.34 / 180.0
        return x_m, y_m

    west_m, north_m = num2merc(x_min, y_min, zoom)
    east_m, south_m = num2merc(x_max + 1, y_max + 1, zoom)

    src_transform = from_bounds(west_m, south_m, east_m, north_m, stitched.shape[1], stitched.shape[0])
    src_crs = "EPSG:3857"

    reprojected_dem = np.zeros((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=stitched,
        destination=reprojected_dem,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    # Save to cache
    meta_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": dst_w,
        "height": dst_h,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform
    }
    with rasterio.open(cached_dem_path, "w", **meta_profile) as dst_file:
        dst_file.write(reprojected_dem, 1)

    meta = {
        "source": "AWS Terrain Open Data (SRTM 30m / USGS / Copernicus DEM)",
        "cached": False,
        "cache_path": cached_dem_path,
        "resolution_m": 9.5,
        "crs": str(dst_crs),
        "dimensions": [dst_h, dst_w],
        "valid_pixels": int(np.sum(np.isfinite(reprojected_dem))),
        "min_elevation_m": float(np.nanmin(reprojected_dem)),
        "max_elevation_m": float(np.nanmax(reprojected_dem)),
        "mean_elevation_m": float(np.nanmean(reprojected_dem))
    }
    return reprojected_dem, meta


def extract_terrain_candidates(
    depth: np.ndarray,
    percentile_threshold: float = 25.0,
    min_percentile: float = 1.0
) -> np.ndarray:
    """
    Identify pixels representing terrain / ground plane candidates from relative depth.
    In disparity relative depth, low numerical values represent far distances (ground plane).
    """
    valid = np.isfinite(depth)
    d_valid = depth[valid]

    p_low = np.percentile(d_valid, min_percentile)
    p_high = np.percentile(d_valid, percentile_threshold)

    terrain_mask = valid & (depth >= p_low) & (depth <= p_high)
    return terrain_mask


def fit_srtm_anchor(
    depth: np.ndarray,
    srtm_dem: np.ndarray,
    terrain_percentile: float = 25.0,
    scale_prior: Optional[float] = None,
    method: str = "robust_anchor"
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Fit scene-specific metric elevation parameters H = a*D + b using SRTM coarse terrain.
    
    Methods:
    - 'linear': Direct least-squares on terrain candidates.
    - 'huber': Robust Huber regression on terrain candidates.
    - 'robust_anchor': Anchors ground offset b to median SRTM terrain elevation with scale prior.
    """
    terrain_mask = extract_terrain_candidates(depth, percentile_threshold=terrain_percentile)
    joint_valid = terrain_mask & np.isfinite(srtm_dem) & (srtm_dem > -1000.0)

    n_anchors = int(np.sum(joint_valid))
    if n_anchors < 100:
        raise ValueError(f"Insufficient valid SRTM terrain anchor pixels: {n_anchors}")

    d_anchors = depth[joint_valid].astype(np.float64)
    h_anchors = srtm_dem[joint_valid].astype(np.float64)

    med_d_ground = float(np.median(d_anchors))
    med_h_ground = float(np.median(h_anchors))

    if method == "linear":
        # Direct OLS on terrain candidates
        A = np.column_stack([d_anchors, np.ones_like(d_anchors)])
        params, _, _, _ = np.linalg.lstsq(A, h_anchors, rcond=None)
        a = float(params[0])
        b = float(params[1])
    elif method == "huber":
        # Robust Huber regression
        huber = HuberRegressor(fit_intercept=True)
        huber.fit(d_anchors.reshape(-1, 1), h_anchors)
        a = float(huber.coef_[0])
        b = float(huber.intercept_)
    elif method == "robust_anchor":
        # Ground offset anchoring: b = H_ground - a * D_ground
        if scale_prior is not None:
            a = float(scale_prior)
        else:
            # Fit robust slope from terrain relief if present
            cov_dh = np.cov(d_anchors, h_anchors)[0, 1]
            var_d = np.var(d_anchors)
            a = float(cov_dh / var_d) if var_d > 1e-6 else 6.94
        b = float(med_h_ground - a * med_d_ground)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    diagnostics = {
        "method": method,
        "terrain_percentile": terrain_percentile,
        "anchor_pixels": n_anchors,
        "median_terrain_relative_depth": med_d_ground,
        "median_srtm_terrain_elevation_m": med_h_ground,
        "scale_a": a,
        "offset_b": b
    }
    return a, b, diagnostics


def apply_srtm_calibration(depth: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply metric calibration H(x, y) = a * D(x, y) + b."""
    return (a * depth + b).astype(np.float32)
