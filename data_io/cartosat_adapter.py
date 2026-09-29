"""
DepthWizard - ISRO Cartosat Satellite Data Adapter
Supports:
  - Cartosat-1 (IRS-P5): 2.5m Panchromatic stereo
  - Cartosat-2 Series (2A-2F): 0.65m - 0.8m PAN / 1.6m - 2.0m 4-band Multispectral
  - Cartosat-3: 0.28m PAN / 1.12m 4-band Multispectral
  - CartoDEM: Digital Elevation Models (10m / 30m grid)

Key Capabilities:
  1. Automatic dynamic range stretching for 10-bit, 11-bit, 12-bit, and 16-bit satellite imagery.
  2. Single-band Panchromatic to 3-channel grayscale conversion for DINOv2 / Depth Anything V2.
  3. 4-band VNIR (NIR, Red, Green, Blue) channel re-ordering to standard RGB.
  4. Solar geometry extraction (Sun Elevation, Sun Azimuth) for shadow-based height estimation.
  5. Coordinate Reference System (CRS), Affine transform, and GSD preservation.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from PIL import Image


def stretch_satellite_dynamic_range(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Applies robust cumulative linear contrast stretch to convert high dynamic range
    satellite imagery (10-bit, 11-bit, 12-bit, 16-bit uint16 or float32) into standard 8-bit [0, 255] uint8.
    Prevents saturation and haze washout.
    """
    arr = image.astype(np.float32)

    if arr.ndim == 2:
        valid_mask = np.isfinite(arr) & (arr > 0)
        if not np.any(valid_mask):
            return np.zeros_like(arr, dtype=np.uint8)

        p_low = np.percentile(arr[valid_mask], lower_percentile)
        p_high = np.percentile(arr[valid_mask], upper_percentile)

        if p_high - p_low < eps:
            stretched = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            stretched = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
        return stretched

    elif arr.ndim == 3:
        # Stretch each channel independently or jointly
        stretched_channels = []
        for c in range(arr.shape[2]):
            ch = arr[:, :, c]
            valid_mask = np.isfinite(ch) & (ch > 0)
            if not np.any(valid_mask):
                stretched_channels.append(np.zeros_like(ch, dtype=np.uint8))
                continue

            p_low = np.percentile(ch[valid_mask], lower_percentile)
            p_high = np.percentile(ch[valid_mask], upper_percentile)

            if p_high - p_low < eps:
                ch_s = np.clip(ch, 0, 255).astype(np.uint8)
            else:
                ch_s = np.clip((ch - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
            stretched_channels.append(ch_s)

        return np.stack(stretched_channels, axis=-1)

    else:
        raise ValueError(f"Unsupported image array dimensions: {arr.shape}")


def extract_satellite_metadata(tags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts standard ISRO / NRSC / Cartosat metadata attributes from raster tags or XML headers.
    """
    meta: Dict[str, Any] = {
        "solar_elevation": None,
        "solar_azimuth": None,
        "satellite_name": None,
        "sensor": None,
        "incidence_angle": None,
        "roll_angle": None,
        "pitch_angle": None,
        "raw_tags": tags
    }

    # Normalize keys for case-insensitive lookup
    upper_tags = {k.upper(): v for k, v in tags.items()}

    # Solar Elevation / Sun Elevation
    for key in ["SOLAR_ELEVATION", "SUNELEVATION", "SUN_ELEVATION", "SUN_ELEV", "ELEVATION"]:
        if key in upper_tags:
            try:
                meta["solar_elevation"] = float(upper_tags[key])
                break
            except (ValueError, TypeError):
                pass

    # Solar Azimuth / Sun Azimuth
    for key in ["SOLAR_AZIMUTH", "SUNAZIMUTH", "SUN_AZIMUTH", "SUN_AZIM", "AZIMUTH"]:
        if key in upper_tags:
            try:
                meta["solar_azimuth"] = float(upper_tags[key])
                break
            except (ValueError, TypeError):
                pass

    # Satellite Name / Mission
    for key in ["SATELLITE", "SPACECRAFT_NAME", "MISSION", "SATELLITE_NAME", "PLATFORM"]:
        if key in upper_tags:
            meta["satellite_name"] = str(upper_tags[key])
            break

    # Sensor
    for key in ["SENSOR", "SENSOR_NAME", "INSTRUMENT", "CAMERA"]:
        if key in upper_tags:
            meta["sensor"] = str(upper_tags[key])
            break

    # Incidence / Tilt angles
    for key in ["INCIDENCE_ANGLE", "INCIDENCEANGLE", "TILT_ANGLE"]:
        if key in upper_tags:
            try:
                meta["incidence_angle"] = float(upper_tags[key])
                break
            except (ValueError, TypeError):
                pass

    return meta


def load_cartosat_image(
    path: str,
    stretch_dynamic_range: bool = True,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0
) -> Dict[str, Any]:
    """
    Loads and preprocesses an ISRO Cartosat satellite image (PAN or Multispectral).
    
    Returns:
    {
        "rgb": np.ndarray (H, W, 3) uint8 ready for Depth Anything V2,
        "raw_data": np.ndarray original bands,
        "is_panchromatic": bool,
        "band_count": int,
        "crs": rasterio.crs.CRS or None,
        "transform": rasterio.transform.Affine or None,
        "gsd_x": float or None (meters),
        "gsd_y": float or None (meters),
        "solar_elevation": float or None (degrees),
        "solar_azimuth": float or None (degrees),
        "metadata": dict
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cartosat image file not found: '{path}'")

    with rasterio.open(path) as src:
        raw_crs = src.crs
        raw_transform = src.transform
        tags = src.tags()
        band_count = src.count
        data = src.read()  # (C, H, W)
        dtype = src.dtypes[0]

    gsd_x, gsd_y = None, None
    if raw_transform is not None and not raw_transform.is_identity:
        gsd_x = abs(float(raw_transform.a))
        gsd_y = abs(float(raw_transform.e))

    meta = extract_satellite_metadata(tags)
    is_panchromatic = False

    # Process bands
    if band_count == 1:
        # Cartosat-1, Cartosat-2 PAN, or Cartosat-3 PAN (Single grayscale band)
        is_panchromatic = True
        pan = data[0]  # (H, W)

        if stretch_dynamic_range or dtype != "uint8" or np.max(pan) > 255:
            pan_8u = stretch_satellite_dynamic_range(
                pan, lower_percentile=lower_percentile, upper_percentile=upper_percentile
            )
        else:
            pan_8u = pan.astype(np.uint8)

        # Replicate grayscale into 3-channel RGB for ViT encoder
        rgb = np.repeat(pan_8u[:, :, np.newaxis], 3, axis=2)

    elif band_count == 3:
        # Standard 3-band RGB / False Color
        img = data.transpose(1, 2, 0)  # (H, W, 3)
        if stretch_dynamic_range or dtype != "uint8" or np.max(img) > 255:
            rgb = stretch_satellite_dynamic_range(
                img, lower_percentile=lower_percentile, upper_percentile=upper_percentile
            )
        else:
            rgb = img.astype(np.uint8)

    elif band_count >= 4:
        # Cartosat 4-band Multispectral (Standard Cartosat order: Band 1=Blue, Band 2=Green, Band 3=Red, Band 4=NIR)
        # We extract Red (idx 2), Green (idx 1), Blue (idx 0)
        red = data[2]
        green = data[1]
        blue = data[0]
        raw_rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)

        if stretch_dynamic_range or dtype != "uint8" or np.max(raw_rgb) > 255:
            rgb = stretch_satellite_dynamic_range(
                raw_rgb, lower_percentile=lower_percentile, upper_percentile=upper_percentile
            )
        else:
            rgb = raw_rgb.astype(np.uint8)

    else:
        raise ValueError(f"Invalid band count ({band_count}) for Cartosat scene '{path}'.")

    return {
        "rgb": rgb,
        "raw_data": data,
        "is_panchromatic": is_panchromatic,
        "band_count": band_count,
        "crs": raw_crs,
        "transform": raw_transform,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "solar_elevation": meta["solar_elevation"],
        "solar_azimuth": meta["solar_azimuth"],
        "metadata": meta,
        "path": path
    }
