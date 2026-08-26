import unittest
import numpy as np
from calibration.gcp_calibration import (
    sample_random_gcps,
    sample_grid_stratified_gcps,
    sample_terrain_structure_gcps,
    fit_gcp_calibration,
    apply_gcp_calibration
)


class TestGCPCalibration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.mask = np.ones((50, 50), dtype=bool)
        self.depth = np.random.uniform(0.5, 2.5, (50, 50)).astype(np.float32)
        self.ref = 10.0 * self.depth + 30.0 + np.random.normal(0, 0.2, (50, 50)).astype(np.float32)

    def test_sample_random_gcps(self):
        for k in [1, 5, 20]:
            rows, cols = sample_random_gcps(self.mask, k, seed=42)
            self.assertEqual(len(rows), k)
            self.assertEqual(len(cols), k)
            self.assertTrue(np.all(self.mask[rows, cols]))

    def test_sample_grid_stratified_gcps(self):
        for k in [1, 5, 20]:
            rows, cols = sample_grid_stratified_gcps(self.mask, k, seed=42)
            self.assertEqual(len(rows), k)
            self.assertEqual(len(cols), k)
            self.assertTrue(np.all(self.mask[rows, cols]))

    def test_sample_terrain_structure_gcps(self):
        for k in [1, 5, 20]:
            rows, cols = sample_terrain_structure_gcps(self.depth, self.mask, k, seed=42)
            self.assertEqual(len(rows), k)
            self.assertEqual(len(cols), k)
            self.assertTrue(np.all(self.mask[rows, cols]))

    def test_fit_k1(self):
        rows, cols = sample_random_gcps(self.mask, 1, seed=42)
        d = self.depth[rows, cols]
        r = self.ref[rows, cols]
        a, b = fit_gcp_calibration(d, r, scale_prior=10.0)
        self.assertEqual(a, 10.0)
        self.assertAlmostEqual(b, float(r[0] - 10.0 * d[0]), places=4)

    def test_fit_k20(self):
        rows, cols = sample_random_gcps(self.mask, 20, seed=42)
        d = self.depth[rows, cols]
        r = self.ref[rows, cols]
        a, b = fit_gcp_calibration(d, r)
        self.assertGreater(a, 5.0)
        self.assertLess(a, 15.0)
        self.assertGreater(b, 20.0)
        self.assertLess(b, 40.0)

    def test_apply_gcp_calibration(self):
        pred = apply_gcp_calibration(self.depth, 10.0, 30.0)
        self.assertEqual(pred.shape, self.depth.shape)
        self.assertAlmostEqual(float(pred[0, 0]), float(10.0 * self.depth[0, 0] + 30.0), places=4)


if __name__ == "__main__":
    unittest.main()
