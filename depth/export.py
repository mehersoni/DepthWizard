"""
M1 Perception Module - M5 Viewer JSON Bridge
"""
import json
import logging
import numpy as np
from PIL import Image


def export_to_m5_json(raw_depth: np.ndarray, output_path: str, grid_size: int = 128) -> bool:
    """
    Compresses and normalizes the depth array to a 1D JSON list for the M5 Three.js viewer.
    """
    try:
        # Normalize to 0.0 - 1.0
        d_min, d_max = np.min(raw_depth), np.max(raw_depth)
        normalized = (raw_depth - d_min) / (d_max - d_min + 1e-8)
        
        # Resize to match Three.js grid
        img = Image.fromarray(normalized)
        img_resized = img.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
        resized_array = np.array(img_resized, dtype=np.float32)
        
        # Flatten and export
        flat_list = resized_array.flatten().tolist()
        with open(output_path, 'w') as f:
            json.dump(flat_list, f)
            
        return True
    except Exception as e:
        logging.error(f"[M1 EXPORT] Failed to export M5 JSON: {str(e)}")
        return False