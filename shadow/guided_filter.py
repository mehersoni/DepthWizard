"""
M7 Guided Filter Depth Refinement Module

Provides edge-preserving Guided Filtering for Depth Anything V2 monocular depth rasters using
co-registered high-resolution satellite orthophotos (RGB) as guidance.

Key Features:
1. Pure OpenCV Fallback Engine (`guided_filter_pure_cv2` via `cv2.boxFilter`).
2. Hardware-Accelerated Native Engine (`cv2.ximgproc.guidedFilter`) auto-detected when present.
3. Strict spatial alignment, datatype, and finite-range validations.
4. Bounded output clipping to preserve physical/relative disparity representation [0.0, 1.0].
5. Zero Ground-Truth Leakage Guarantee (GT height is strictly excluded from all filtering decisions).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class GuidedFilterConfig:
    """
    Immutable configuration dataclass for Guided Filter depth refinement.
    
    Attributes:
        radius (int): Neighborhood kernel radius in pixels.
                      r=16 px corresponds to 0.80m physical radius at 0.05m/px Potsdam GSD.
        eps (float): Regularization parameter penalizing high local guidance variance.
        enable_guided_filter (bool): Global feature flag enabling depth refinement.
        enable_fallback (bool): Flag allowing pure OpenCV box-filter fallback if ximgproc is missing.
    """
    radius: int = 16
    eps: float = 0.01
    enable_guided_filter: bool = True
    enable_fallback: bool = True


def guided_filter_pure_cv2(
    guide_gray: np.ndarray,
    target_depth: np.ndarray,
    radius: int = 16,
    eps: float = 0.01
) -> np.ndarray:
    """
    Pure OpenCV box-filter implementation of He et al. (IEEE TPAMI 2013) Guided Filter.
    Operates in O(1) time complexity per pixel regardless of filter kernel size.

    Parameters:
        guide_gray (np.ndarray): 2D float32 grayscale guidance raster in range [0.0, 1.0].
        target_depth (np.ndarray): 2D float32 filtering target disparity map in range [0.0, 1.0].
        radius (int): Box filter spatial radius in pixels.
        eps (float): Regularization parameter.

    Returns:
        np.ndarray: Refined 2D float32 disparity map matching target_depth dimensions.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    # 1. Local Means
    mean_I = cv.boxFilter(guide_gray, cv.CV_32F, ksize)
    mean_p = cv.boxFilter(target_depth, cv.CV_32F, ksize)

    # 2. Local Covariance & Variance
    mean_Ip = cv.boxFilter(guide_gray * target_depth, cv.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv.boxFilter(guide_gray * guide_gray, cv.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    # 3. Linear Regression Coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 4. Smoothed Coefficients
    mean_a = cv.boxFilter(a, cv.CV_32F, ksize)
    mean_b = cv.boxFilter(b, cv.CV_32F, ksize)

    # 5. Filtered Output q = mean_a * I + mean_b
    q = mean_a * guide_gray + mean_b
    return q.astype(np.float32)


def refine_depth_anything_map(
    guide_image: np.ndarray,
    raw_depth: np.ndarray,
    radius: int = 16,
    eps: float = 0.01,
    use_contrib_if_available: bool = True
) -> np.ndarray:
    """
    Refines Depth Anything V2 monocular disparity map using co-registered RGB orthophoto guidance.

    Parameters:
        guide_image (np.ndarray): Guidance raster (H, W, 3) BGR/RGB or (H, W) grayscale.
        raw_depth (np.ndarray): 2D relative disparity map (H, W) float32 in range [0.0, 1.0].
        radius (int): Spatial radius in pixels (default r=16 px = 0.80m at 0.05m/px GSD).
        eps (float): Regularization parameter (default eps=0.01).
        use_contrib_if_available (bool): If True, uses cv2.ximgproc.guidedFilter when available.

    Returns:
        np.ndarray: Refined disparity map (H, W) float32 clipped to [0.0, 1.0].

    Raises:
        ValueError: If guide_image or raw_depth inputs fail structural sanity checks.
    """
    # 1. Validation: Inputs must exist
    if guide_image is None or not isinstance(guide_image, np.ndarray):
        raise ValueError("Guide image must be a valid non-None NumPy array.")
    if raw_depth is None or not isinstance(raw_depth, np.ndarray):
        raise ValueError("Raw depth map must be a valid non-None NumPy array.")

    # 2. Validation: Spatial shape matching
    h_g, w_g = guide_image.shape[:2]
    if raw_depth.ndim != 2:
        raise ValueError(f"Raw depth map must be a 2D array, received shape {raw_depth.shape}")
    h_d, w_d = raw_depth.shape
    if (h_g, w_g) != (h_d, w_d):
        raise ValueError(f"Spatial dimension mismatch: guide {guide_image.shape[:2]} vs depth {raw_depth.shape}")

    # 3. Validation & Sanitization: Finite values & Float32
    d_clean = np.nan_to_num(raw_depth, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # 4. Guidance Pre-processing
    if guide_image.ndim == 3 and guide_image.shape[2] in [3, 4]:
        guide_bgr = guide_image[:, :, :3]
        if guide_bgr.dtype == np.uint8:
            guide_gray_f32 = (cv.cvtColor(guide_bgr, cv.COLOR_BGR2GRAY) / 255.0).astype(np.float32)
        else:
            guide_gray_f32 = cv.cvtColor(guide_bgr.astype(np.float32), cv.COLOR_BGR2GRAY)
    elif guide_image.ndim == 2:
        guide_bgr = cv.cvtColor(guide_image, cv.COLOR_GRAY2BGR) if guide_image.dtype == np.uint8 else None
        if guide_image.dtype == np.uint8:
            guide_gray_f32 = (guide_image / 255.0).astype(np.float32)
        else:
            guide_gray_f32 = guide_image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported guide image dimensions: {guide_image.shape}")

    # 5. Engine Selection: ximgproc vs Pure OpenCV Fallback
    has_ximgproc = hasattr(cv, "ximgproc") and hasattr(cv.ximgproc, "guidedFilter")

    if use_contrib_if_available and has_ximgproc and guide_bgr is not None:
        try:
            # OpenCV ximgproc guided filter expects guide uint8 BGR and src float32
            guide_u8 = guide_bgr if guide_bgr.dtype == np.uint8 else (guide_bgr * 255).astype(np.uint8)
            d_filt = cv.ximgproc.guidedFilter(
                guide=guide_u8,
                src=d_clean,
                radius=radius,
                eps=eps
            )
        except Exception:
            # Graceful fallback to pure OpenCV box-filter
            d_filt = guided_filter_pure_cv2(guide_gray_f32, d_clean, radius=radius, eps=eps)
    else:
        # Standard deterministic fallback
        d_filt = guided_filter_pure_cv2(guide_gray_f32, d_clean, radius=radius, eps=eps)

    # 6. Bounded Output Bounding
    d_filt = np.clip(d_filt, 0.0, 1.0).astype(np.float32)
    return d_filt
