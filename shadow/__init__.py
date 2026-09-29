from .detector import detect_shadow_candidates
from .cleaner import clean_candidate_mask
from .geometry import (
    extract_region_geometries,
    compute_shadow_directional_geometry,
    compute_object_shadow_adjacency,
    compute_object_shadow_pairing,
    compute_shadow_length_px
)
from .confidence import rank_shadow_regions
from .scale import PhysicalScaleManager
from .height import (
    pixel_to_physical_shadow_length,
    compute_building_height,
    estimate_building_height,
    propagate_height_uncertainty
)
from .run_full_pipeline import run_full_pipeline
from .guided_filter import (
    GuidedFilterConfig,
    guided_filter_pure_cv2,
    color_guided_filter_pure_cv2,
    sharpen_color_edges,
    refine_depth_anything_map
)

__all__ = [
    "detect_shadow_candidates",
    "clean_candidate_mask",
    "extract_region_geometries",
    "compute_shadow_directional_geometry",
    "compute_object_shadow_adjacency",
    "compute_object_shadow_pairing",
    "compute_shadow_length_px",
    "rank_shadow_regions",
    "PhysicalScaleManager",
    "pixel_to_physical_shadow_length",
    "compute_building_height",
    "estimate_building_height",
    "propagate_height_uncertainty",
    "run_full_pipeline",
    "GuidedFilterConfig",
    "guided_filter_pure_cv2",
    "color_guided_filter_pure_cv2",
    "sharpen_color_edges",
    "refine_depth_anything_map"
]


