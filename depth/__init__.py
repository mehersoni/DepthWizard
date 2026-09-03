"""
Depth Module for DepthWizard
Provides depth estimation models, tiled inference, preprocessing, and export utilities.
"""

from .depth_model import estimate_depth, load_model, get_device
from .tiled_inference import estimate_depth_tiled, run_tiled_inference

try:
    from .models import DepthEstimator
    from .preprocess import load_and_validate_image
    from .tiles import generate_overlapping_tiles, solve_global_alignment
    from .visualize import generate_diagnostic_plot
    from .export import export_to_m5_json
except ImportError:
    pass

__all__ = [
    "estimate_depth",
    "load_model",
    "get_device",
    "estimate_depth_tiled",
    "run_tiled_inference",
    "DepthEstimator",
    "load_and_validate_image",
    "generate_overlapping_tiles",
    "solve_global_alignment",
    "generate_diagnostic_plot",
    "export_to_m5_json",
]
