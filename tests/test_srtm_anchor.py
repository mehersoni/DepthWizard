import unittest
import numpy as np
from calibration.srtm_anchor import (
    extract_terrain_candidates,
    fit_srtm_anchor,
    apply_srtm_calibration
)


class TestSRTMAnchor(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create synthetic 100x100 relative depth and coarse DEM
        # Ground plane at ~45m with buildings up to 70m
        self.ground_elev = 45.0
        self.scale_true = 8.0
        self.offset_true = 37.0

        # Disparity depth in [0.5, 2.5]
        self.depth = np.random.uniform(0.5, 2.5, (100, 100)).astype(np.float32)
        # 30% of pixels are ground (lower depth values)
        self.depth[:30, :] = np.random.uniform(0.5, 1.0, (30, 100)).astype(np.float32)

        # Synthetic coarse DEM has terrain ~45m with low spatial variation
        self.srtm_dem = (self.ground_elev + np.random.normal(0, 0.5, (100, 100))).astype(np.float32)

    def test_extract_terrain_candidates(self):
        mask = extract_terrain_candidates(self.depth, percentile_threshold=25.0)
        self.assertEqual(mask.shape, self.depth.shape)
        self.assertTrue(np.any(mask))
        self.assertLessEqual(np.sum(mask), self.depth.size * 0.30)
        # Verify selected pixels are in the lower depth range
        self.assertLessEqual(np.max(self.depth[mask]), np.percentile(self.depth, 26.0))

    def test_fit_srtm_anchor_robust(self):
        scale_prior = 8.0
        a, b, diag = fit_srtm_anchor(
            self.depth,
            self.srtm_dem,
            terrain_percentile=25.0,
            scale_prior=scale_prior,
            method="robust_anchor"
        )
        self.assertAlmostEqual(a, 8.0, places=4)
        self.assertGreater(b, 30.0)
        self.assertLess(b, 50.0)
        self.assertIn("anchor_pixels", diag)
        self.assertGreater(diag["anchor_pixels"], 500)

    def test_apply_srtm_calibration(self):
        a, b = 7.5, 40.0
        pred = apply_srtm_calibration(self.depth, a, b)
        self.assertEqual(pred.shape, self.depth.shape)
        self.assertTrue(np.all(pred >= 40.0))
        self.assertAlmostEqual(float(pred[0, 0]), float(a * self.depth[0, 0] + b), places=5)

    def test_insufficient_anchors_raises_error(self):
        bad_dem = np.full((10, 10), np.nan, dtype=np.float32)
        small_depth = np.random.rand(10, 10).astype(np.float32)
        with self.assertRaises(ValueError):
            fit_srtm_anchor(small_depth, bad_dem, terrain_percentile=25.0)


if __name__ == "__main__":
    unittest.main()
