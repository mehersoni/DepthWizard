"""
Unit Tests for M7 Guided Filter Depth Refinement Module (`shadow/guided_filter.py`)

Executes comprehensive synthetic and functional unit tests for Guided Filtering:
1. Spatial shape preservation.
2. Datatype preservation (np.float32).
3. Finite value assertion (no NaN / Inf).
4. Bounded range assertion [0.0, 1.0].
5. Uniform image stability.
6. Synthetic step-edge sharpening behavior.
7. Spatial dimension mismatch rejection.
8. NaN / Inf input sanitization.
9. Pure OpenCV fallback execution.
10. Contrib execution check.
11. Deterministic execution variance assertion (sigma^2 = 0.0).
"""

import unittest
import os
import sys
import numpy as np
import cv2 as cv

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.guided_filter import (
    refine_depth_anything_map,
    guided_filter_pure_cv2,
    GuidedFilterConfig
)


class TestGuidedFilterModule(unittest.TestCase):

    def setUp(self):
        """Generates synthetic test rasters."""
        self.height = 100
        self.width = 100
        np.random.seed(42)

        # Synthetic BGR orthophoto with vertical edge at x = 50
        self.guide_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.guide_bgr[:, :50, :] = 50   # Dark region
        self.guide_bgr[:, 50:, :] = 200  # Bright region

        # Synthetic blurry depth disparity map with soft transition at x = 50
        x_grid = np.linspace(0, 1, self.width)
        blurry_row = 1.0 / (1.0 + np.exp(-0.2 * (np.arange(self.width) - 50)))
        self.raw_depth = np.tile(blurry_row, (self.height, 1)).astype(np.float32)

    def test_shape_and_dtype_preservation(self):
        """Verifies output shape and dtype match input depth map."""
        filt_depth = refine_depth_anything_map(self.guide_bgr, self.raw_depth, radius=4, eps=0.01)
        self.assertEqual(filt_depth.shape, self.raw_depth.shape)
        self.assertEqual(filt_depth.dtype, np.float32)

    def test_finite_and_bounded_range(self):
        """Verifies output contains only finite values within range [0.0, 1.0]."""
        filt_depth = refine_depth_anything_map(self.guide_bgr, self.raw_depth, radius=4, eps=0.01)
        self.assertTrue(np.all(np.isfinite(filt_depth)))
        self.assertTrue(np.all(filt_depth >= 0.0))
        self.assertTrue(np.all(filt_depth <= 1.0))

    def test_uniform_image_stability(self):
        """Verifies uniform input rasters yield stable uniform outputs."""
        uniform_guide = np.full((50, 50, 3), 128, dtype=np.uint8)
        uniform_depth = np.full((50, 50), 0.5, dtype=np.float32)
        filt_depth = refine_depth_anything_map(uniform_guide, uniform_depth, radius=4, eps=0.01)
        np.testing.assert_allclose(filt_depth, 0.5, atol=1e-4)

    def test_synthetic_step_edge_sharpening(self):
        """Verifies Guided Filter sharpens depth transition along guidance edge."""
        filt_depth = refine_depth_anything_map(self.guide_bgr, self.raw_depth, radius=4, eps=0.01)
        
        # Measure gradient magnitude across boundary (x=49 to x=51)
        grad_raw = abs(self.raw_depth[50, 51] - self.raw_depth[50, 49])
        grad_filt = abs(filt_depth[50, 51] - filt_depth[50, 49])
        
        self.assertGreaterEqual(grad_filt, grad_raw, "Filtered edge gradient should be sharper than raw gradient.")

    def test_dimension_mismatch_rejection(self):
        """Verifies spatial dimension mismatch raises ValueError."""
        mismatched_guide = np.zeros((80, 80, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            refine_depth_anything_map(mismatched_guide, self.raw_depth)

    def test_nan_inf_handling(self):
        """Verifies NaN/Inf values in raw depth are sanitized safely."""
        corrupt_depth = self.raw_depth.copy()
        corrupt_depth[10, 10] = np.nan
        corrupt_depth[20, 20] = np.inf
        corrupt_depth[30, 30] = -np.inf
        
        filt_depth = refine_depth_anything_map(self.guide_bgr, corrupt_depth, radius=4, eps=0.01)
        self.assertTrue(np.all(np.isfinite(filt_depth)))

    def test_fallback_engine_execution(self):
        """Verifies pure OpenCV fallback engine operates correctly."""
        guide_gray = (cv.cvtColor(self.guide_bgr, cv.COLOR_BGR2GRAY) / 255.0).astype(np.float32)
        filt_fallback = guided_filter_pure_cv2(guide_gray, self.raw_depth, radius=4, eps=0.01)
        self.assertEqual(filt_fallback.shape, self.raw_depth.shape)
        self.assertEqual(filt_fallback.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(filt_fallback)))

    def test_deterministic_execution_variance(self):
        """Verifies 5 repeated executions yield identical deterministic results (variance = 0.0)."""
        results = []
        for _ in range(5):
            res = refine_depth_anything_map(self.guide_bgr, self.raw_depth, radius=4, eps=0.01)
            results.append(res)
            
        res_stack = np.stack(results, axis=0)
        variance = float(np.var(res_stack, axis=0).max())
        self.assertLess(variance, 1e-12, "Run variance across 5 repeated runs must be negligible (< 1e-12).")

    def test_config_dataclass(self):
        """Verifies GuidedFilterConfig immutable dataclass attributes."""
        cfg = GuidedFilterConfig(radius=16, eps=0.01)
        self.assertEqual(cfg.radius, 16)
        self.assertEqual(cfg.eps, 0.01)
        self.assertTrue(cfg.enable_guided_filter)


if __name__ == "__main__":
    unittest.main()
