import os
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image, UnidentifiedImageError

import rasterio
from rasterio.warp import reproject, Resampling

def load_aligned_dsm(rgb_path, dsm_path):
    """
    Geospatially aligns a DSM to an RGB reference image.
    Ensures identical CRS, bounding box, and pixel dimensions.
    """
    with rasterio.open(rgb_path) as src_rgb:
        rgb_transform = src_rgb.transform
        rgb_crs = src_rgb.crs
        h, w = src_rgb.height, src_rgb.width
        
    with rasterio.open(dsm_path) as src_dsm:
        # Create an empty array perfectly matching the RGB dimensions
        aligned_dsm = np.zeros((h, w), dtype=np.float32)
        
        # Reproject using actual geographic metadata, not just array stretching
        reproject(
            source=rasterio.band(src_dsm, 1),
            destination=aligned_dsm,
            src_transform=src_dsm.transform,
            src_crs=src_dsm.crs,
            dst_transform=rgb_transform,
            dst_crs=rgb_crs,
            resampling=Resampling.nearest # Nearest neighbor prevents interpolating fake elevations
        )
        
    return aligned_dsm

def load_and_validate_image(image_path: str) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
    if not os.path.exists(image_path):
        logging.error(f"[M1 PREPROCESS] File not found at path: {image_path}")
        return None

    # Default metadata dictionary
    metadata = {"is_georeferenced": False, "crs": None, "transform": None}

    try:
        # Check if it is a georeferenced TIFF
        if image_path.lower().endswith(('.tif', '.tiff')) and rasterio is not None:
            with rasterio.open(image_path) as src:
                # Read first 3 bands (RGB)
                r, g, b = src.read(1), src.read(2), src.read(3)
                
                # Stack and convert to PIL Image for the HuggingFace pipeline
                rgb_array = np.dstack((r, g, b))
                image = Image.fromarray(rgb_array)
                
                metadata["is_georeferenced"] = True
                metadata["crs"] = src.crs.to_string() if src.crs else None
                metadata["transform"] = src.transform
                
                logging.info("[M1 PREPROCESS] GeoTIFF loaded with spatial metadata.")
                return image, metadata
                
        # Fallback for standard PNG/JPG
        image = Image.open(image_path).convert("RGB")
        logging.info("[M1 PREPROCESS] Standard optical image loaded (No spatial metadata).")
        return image, metadata

    except Exception as e:
        logging.error(f"[M1 PREPROCESS] Error: {str(e)}")
        return None