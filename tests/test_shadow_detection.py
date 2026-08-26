import unittest
import numpy as np
from shadow_detection import (
    compute_shadow_confidence,
    filter_shadow_mask,
    detect_building_shadow_pairs,
    apply_shadow_height_constraint
)


class TestShadowDetection(unittest.TestCase):
    def test_compute_shadow_confidence_shape_and_bounds(self):
        h, w = 100, 100
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Inject a dark shadow region with blue tint
        rgb[20:50, 20:50, 0] = 30   # Low Red
        rgb[20:50, 20:50, 1] = 40   # Low Green
        rgb[20:50, 20:50, 2] = 80   # Higher Blue

        conf = compute_shadow_confidence(rgb)
        self.assertEqual(conf.shape, (h, w))
        self.assertEqual(conf.dtype, np.float32)
        self.assertTrue(np.all(conf >= 0.0) and np.all(conf <= 1.0))
        self.assertGreater(np.mean(conf[20:50, 20:50]), np.mean(conf[:15, :15]))

    def test_filter_shadow_mask(self):
        conf = np.zeros((100, 100), dtype=np.float32)
        conf[20:60, 20:60] = 0.8
        conf[5, 5] = 0.9

        mask = filter_shadow_mask(conf, confidence_threshold=0.5, min_area_pixels=50)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, bool)
        self.assertGreater(np.sum(mask[20:60, 20:60]), 1000)
        self.assertFalse(mask[5, 5])

    def test_detect_building_shadow_pairs(self):
        h, w = 200, 200
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 150
        shadow_mask = np.zeros((h, w), dtype=bool)
        initial_dsm = np.ones((h, w), dtype=np.float32) * 40.0

        initial_dsm[50:100, 50:100] = 60.0
        shadow_mask[100:140, 50:100] = True

        res = detect_building_shadow_pairs(rgb, shadow_mask, initial_dsm, gsd_m=0.05)
        self.assertGreaterEqual(res["num_shadow_candidates"], 1)
        self.assertGreaterEqual(res["num_accepted_pairs"], 1)
        self.assertGreater(res["mean_shadow_length_m"], 0.0)

    def test_apply_shadow_height_constraint_mode_a_and_b(self):
        dsm = np.ones((100, 100), dtype=np.float32) * 45.0
        mask = np.zeros((100, 100), dtype=bool)
        data = {
            "pairs": [
                {"shadow_id": 1, "length_m": 10.0}
            ]
        }

        # Mode A: Solar elevation provided
        dsm_a, summary_a = apply_shadow_height_constraint(dsm, mask, data, solar_elevation_deg=45.0)
        self.assertEqual(summary_a["mode"], "MODE_A_SOLAR_METADATA_AVAILABLE")
        self.assertTrue(summary_a["metric_height_applied"])
        self.assertAlmostEqual(summary_a["mean_shadow_height_m"], 10.0 * np.tan(np.radians(45.0)))

        # Mode B: Solar elevation unavailable (None)
        dsm_b, summary_b = apply_shadow_height_constraint(dsm, mask, data, solar_elevation_deg=None)
        self.assertEqual(summary_b["mode"], "MODE_B_SOLAR_METADATA_UNAVAILABLE")
        self.assertFalse(summary_b["metric_height_applied"])
        self.assertIsNone(summary_b["mean_shadow_height_m"])


if __name__ == "__main__":
    unittest.main()
