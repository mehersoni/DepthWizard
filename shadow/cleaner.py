"""
M4 Shadow Cue Module - Morphological Cleaner Component

This module cleans raw candidate shadow masks using OpenCV morphological operations
(opening and closing) to remove speckle noise and seal interior region voids.
"""

from typing import Tuple, Union
import cv2 as cv
import numpy as np


def clean_candidate_mask(
    candidate_mask: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    shape: int = cv.MORPH_ELLIPSE,
    open_iterations: int = 1,
    close_iterations: int = 1
) -> np.ndarray:
    """
    Clean a binary candidate shadow mask using morphological opening and closing operations.

    Reasoning:
    ----------
    1. Morphological Opening (Erosion followed by Dilation):
       Removes isolated pixel speckles, thin spurious noise lines, and tiny non-shadow artifacts
       without shrinking the overall boundaries of valid shadow regions.
    2. Morphological Closing (Dilation followed by Erosion):
       Fills small interior holes (e.g., bright roof projections, sensor noise) inside shadow blobs,
       yielding solid, contiguous candidate shadow polygons.

    Parameters:
    -----------
    candidate_mask : np.ndarray
        Binary mask (H, W), dtype uint8, where 255 represents candidate shadow pixels.
    kernel_size : int or tuple of (int, int), default=3
        Size of the structuring element kernel (e.g., 3 for a 3x3 kernel).
    shape : int, default=cv.MORPH_ELLIPSE
        OpenCV structuring element shape (cv.MORPH_ELLIPSE, cv.MORPH_RECT, or cv.MORPH_CROSS).
    open_iterations : int, default=1
        Number of morphological opening iterations. Set to 0 to skip opening.
    close_iterations : int, default=1
        Number of morphological closing iterations. Set to 0 to skip closing.

    Returns:
    --------
    cleaned_mask : np.ndarray
        Cleaned binary mask (H, W), dtype uint8 (255 = shadow, 0 = background).
    """
    if candidate_mask is None or not isinstance(candidate_mask, np.ndarray):
        raise ValueError("Candidate mask must be a valid NumPy array.")

    if candidate_mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, received shape {candidate_mask.shape}")

    # Standardize kernel size tuple
    if isinstance(kernel_size, int):
        k_size = (kernel_size, kernel_size)
    else:
        k_size = kernel_size

    # Create structuring element
    kernel = cv.getStructuringElement(shape, k_size)

    cleaned = candidate_mask.copy()

    # 1. Morphological Opening: Erase small noise speckles
    if open_iterations > 0:
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel, iterations=open_iterations)

    # 2. Morphological Closing: Fill internal voids and connect close fragments
    if close_iterations > 0:
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=close_iterations)

    return cleaned
