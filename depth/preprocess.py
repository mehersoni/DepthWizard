"""
M1 Perception Module - Image Preprocessing and Validation
"""
import os
import logging
from typing import Optional
from PIL import Image, UnidentifiedImageError

def load_and_validate_image(image_path: str) -> Optional[Image.Image]:
    """
    Safely loads a satellite image, verifying existence and format.

    Parameters:
    -----------
    image_path : str
        The file path to the input RGB satellite image.

    Returns:
    --------
    image : PIL.Image.Image or None
        The loaded RGB image, or None if validation fails.
    """
    if not os.path.exists(image_path):
        logging.error(f"[M1 PREPROCESS] File not found at path: {image_path}")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except UnidentifiedImageError:
        logging.error(f"[M1 PREPROCESS] Invalid image format at: {image_path}")
        return None
    except Exception as e:
        logging.error(f"[M1 PREPROCESS] Unexpected error loading image: {str(e)}")
        return None