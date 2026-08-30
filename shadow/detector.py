"""
M4 Shadow Cue Module - Shadow Detector Component
"""

import cv2 as cv
import numpy as np


def detect_shadow_candidates(
    image: np.ndarray,
    v_max: int = 80,
    s_min: int = 0,
    s_max: int = 255,
    h_min: int = 0,
    h_max: int = 180,
    adaptive_v: bool = False,
    v_percentile: float = 25.0
) -> np.ndarray:
    """
    Detect candidate shadow regions from a BGR satellite image using HSV thresholding.

    For detailed physical background, optical reasoning, and parameter documentation,
    refer to `shadow/README.md`.

    Parameters:
    -----------
    image : np.ndarray
        Input BGR satellite image, shape (H, W, 3), dtype uint8.
    v_max : int, default=80
        Upper bound threshold for Value (brightness) channel [0, 255].
    s_min : int, default=0
        Lower bound for Saturation channel [0, 255].
    s_max : int, default=255
        Upper bound for Saturation channel [0, 255].
    h_min : int, default=0
        Lower bound for Hue channel [0, 180].
    h_max : int, default=180
        Upper bound for Hue channel [0, 180].
    adaptive_v : bool, default=False
        If True, dynamically computes v_max based on brightness distribution percentile.
    v_percentile : float, default=25.0
        V-channel percentile used when adaptive_v=True.

    Returns:
    --------
    candidate_mask : np.ndarray
        Binary mask (H, W), dtype uint8 (255 = candidate shadow, 0 = non-shadow).
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, received shape {image.shape if image is not None else None}")

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    if adaptive_v:
        v_channel = hsv_image[:, :, 2]
        effective_v_max = int(np.percentile(v_channel, v_percentile))
        effective_v_max = max(0, min(255, effective_v_max))
    else:
        effective_v_max = v_max

    lower_hsv = np.array([h_min, s_min, 0], dtype=np.uint8)
    upper_hsv = np.array([h_max, s_max, effective_v_max], dtype=np.uint8)

    candidate_mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)

    return candidate_mask
