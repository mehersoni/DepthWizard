"""
M4 Shadow Cue Module - Potsdam Empirical Validation Test Suite (STEP 10)

Automated test suite validating:
1. Potsdam tile discovery and tile ID parsing
2. RGB, DSM, nDSM, TFW, and building label relative pairing
3. Dynamic TFW world-file parser & GSD extraction (~0.05 m/px)
4. Dimension assertions (6000x6000 px)
5. Missing-file handling without crashing
6. Ground-truth height extraction from 32-bit float DSM & building labels
7. Spatial prediction-to-ground-truth building matching
8. Validation metric calculations (MAE, RMSE, Median Error, % Error)
9. Production solar-elevation safety blocking behavior ([HEIGHT UNAVAILABLE])
10. Absence of hardcoded C:\\ absolute paths
"""

import os
import sys
import unittest
import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.validate_potsdam_discovery import parse_potsdam_tile_id, discover_potsdam_dataset
from shadow.potsdam_validation import (
    parse_tfw_file,
    extract_potsdam_ground_truth_buildings,
    run_potsdam_tile_validation,
    compute_validation_metrics,
    run_full_potsdam_validation
)


class TestPotsdamValidationSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.demo_dir = os.path.join(cls.root_dir, "demoImages")

    def test_01_tile_id_extraction(self):
        """Test extraction of Potsdam tile IDs from various filename patterns."""
        self.assertEqual(parse_potsdam_tile_id("top_potsdam_2_10_RGB.tif"), "2_10")
        self.assertEqual(parse_potsdam_tile_id("dsm_potsdam_02_11.tif"), "2_11")
        self.assertEqual(parse_potsdam_tile_id("top_potsdam_02_12_RGB.tfw"), "2_12")
        self.assertIsNone(parse_potsdam_tile_id("sat2.png"))

    def test_02_potsdam_discovery_matrix(self):
        """Test dataset discovery in demoImages/ using relative paths."""
        records = discover_potsdam_dataset(root_dir=self.root_dir)
        self.assertGreaterEqual(len(records), 3, "Expected at least 3 discovered Potsdam tiles.")

        tile_ids = [r["tile_id"] for r in records]
        self.assertIn("2_10", tile_ids)
        self.assertIn("2_11", tile_ids)
        self.assertIn("2_12", tile_ids)

        for r in records:
            # Check relative paths only
            for p_val in [r["rgb_path"], r["dsm_path"], r["tfw_path"], r["label_path"]]:
                if p_val:
                    self.assertFalse(os.path.isabs(p_val), f"Absolute path detected: {p_val}")
                    self.assertNotIn(":", p_val, f"Drive letter detected in path: {p_val}")

            # Dimension assertions
            self.assertEqual(r["rgb_dim"], (6000, 6000))
            self.assertTrue(r["is_6000x6000"])

    def test_03_tfw_parsing_and_gsd_extraction(self):
        """Test parsing of ESRI TFW georeferencing world files."""
        tfw_path = os.path.join("Dataset", "Potsdam", "2_Ortho_RGB", "2_Ortho_RGB", "top_potsdam_2_10_RGB.tfw")
        if not os.path.exists(tfw_path):
            tfw_path = os.path.join("demoImages", "top_potsdam_2_10_RGB.tfw")

        tfw_res = parse_tfw_file(tfw_path)
        gsd = tfw_res["meters_per_pixel"]

        self.assertAlmostEqual(gsd, 0.05, delta=0.005)
        self.assertTrue(tfw_res["scale_manager"].is_calibrated)
        self.assertEqual(tfw_res["scale_manager"].meters_per_pixel, gsd)

    def test_04_missing_file_graceful_handling(self):
        """Test missing file reporting in discovery without crashing."""
        records = discover_potsdam_dataset(root_dir=self.root_dir, demo_dir_name="demoImages")
        for r in records:
            self.assertIsInstance(r["missing_files"], list)

    def test_05_ground_truth_height_extraction(self):
        """Test ground-truth elevation extraction from 32-bit DSM and building labels."""
        rec = discover_potsdam_dataset(root_dir=self.root_dir)[0]
        gt_data = extract_potsdam_ground_truth_buildings(rec["dsm_path"], rec["label_path"])

        self.assertIn("z_ground_m", gt_data)
        self.assertGreater(gt_data["total_buildings"], 0)
        self.assertGreater(gt_data["z_ground_m"], 0.0)

        first_b = gt_data["buildings"][0]
        self.assertIn("height_gt_m", first_b)
        self.assertIn("height_median_m", first_b)
        self.assertGreater(first_b["height_gt_m"], 0.0)
        self.assertGreaterEqual(first_b["pixel_count"], 50)

    def test_06_production_solar_blocking_safety(self):
        """Test production safety rule: return REJECTED/blocked when sun_elevation_deg=None."""
        rec = discover_potsdam_dataset(root_dir=self.root_dir)[0]
        tile_res = run_potsdam_tile_validation(rec, test_sun_elevation_deg=None)

        self.assertEqual(len(tile_res["matched_results"]), 0)
        self.assertGreater(len(tile_res["blocked_predictions"]), 0)

        for b in tile_res["blocked_predictions"]:
            self.assertIsNone(b["predicted_height_m"])
            self.assertEqual(b["status"], "REJECTED")

    def test_07_test_mode_height_and_spatial_matching(self):
        """Test experimental test mode execution and spatial prediction matching."""
        rec = discover_potsdam_dataset(root_dir=self.root_dir)[0]
        tile_res = run_potsdam_tile_validation(rec, test_sun_elevation_deg=45.0)

        self.assertGreater(len(tile_res["matched_results"]), 0)

        first_m = tile_res["matched_results"][0]
        self.assertGreater(first_m["predicted_height_m"], 0.0)
        self.assertGreater(first_m["ground_truth_height_m"], 0.0)
        self.assertGreaterEqual(first_m["absolute_error_m"], 0.0)
        self.assertGreaterEqual(first_m["percentage_error"], 0.0)

    def test_08_metric_calculations(self):
        """Test MAE, RMSE, and error percentage metric calculation logic."""
        dummy_tile_results = [{
            "tile_id": "2_10",
            "gsd_m": 0.05,
            "total_gt_buildings": 100,
            "total_predictions": 5,
            "matched_results": [
                {"absolute_error_m": 1.0, "percentage_error": 10.0},
                {"absolute_error_m": 3.0, "percentage_error": 30.0}
            ],
            "unmatched_predictions": [],
            "blocked_predictions": []
        }]

        metrics = compute_validation_metrics(dummy_tile_results)
        ov = metrics["overall"]

        self.assertEqual(ov["mae"], 2.0)
        self.assertAlmostEqual(ov["rmse"], float(np.sqrt(5.0)), delta=1e-3)
        self.assertEqual(ov["median_abs_error"], 2.0)
        self.assertEqual(ov["mean_pct_error"], 20.0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPotsdamValidationSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
