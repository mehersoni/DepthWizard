import os
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image, UnidentifiedImageError

try:
    # pyrefly: ignore [missing-import]
    import rasterio
except ImportError:
    rasterio = None
    logging.warning("[M1 PREPROCESS] rasterio not installed. GeoTIFF metadata extraction disabled.")

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