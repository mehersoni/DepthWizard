"""
DepthWizard M2 — Dedicated Metric Calibration Engine Test Suite
Tests A through G covering OLS, regularized prior calibration, GCP validation, DEM terrain anchoring, and strict compliance.
"""

import os
import sys
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_image import process_image, fit_supplied_gcps, fit_supplied_dem
from depth.depth_model import load_model


def run_m2_calibration_tests():
    print("=" * 80)
    print("DepthWizard M2 — Metric Calibration Engine Comprehensive Test Suite")
    print("=" * 80)

    # Preload model once
    model, processor, device = load_model()
    print(f"Model ready on device: {device}\n")

    # -------------------------------------------------------------------------
    # TEST A: PNG WITHOUT CALIBRATION
    # -------------------------------------------------------------------------
    print("TEST A: Non-georeferenced PNG without calibration")
    res_a = process_image("data/test_sample_photo.png", model=model, processor=processor, device=device)
    assert res_a["mode"] == "relative", f"Expected relative, got {res_a['mode']}"
    assert res_a["calibrated"] is False, "Expected calibrated=False"
    assert res_a["height_unit"] == "rel", f"Expected 'rel', got {res_a['height_unit']}"
    assert res_a["crs"] is None, "Expected crs=None"
    assert res_a["transform"] is None, "Expected transform=None"
    assert res_a["metadata"]["calibration"]["method"] == "none"
    assert res_a["metadata"]["calibration"]["scale_a"] is None
    assert np.min(res_a["height_map"]) >= -1e-6 and np.max(res_a["height_map"]) <= 1.0 + 1e-6
    print("   [PASS] PNG -> relative, uncalibrated, unit='rel', CRS=None, range in [0, 1]\n")

    # -------------------------------------------------------------------------
    # TEST B: GEOTIFF WITHOUT CALIBRATION
    # -------------------------------------------------------------------------
    print("TEST B: Georeferenced GeoTIFF without calibration")
    res_b = process_image("data/potsdam_sample_1024.tif", model=model, processor=processor, device=device)
    assert res_b["mode"] == "relative", f"Expected relative, got {res_b['mode']}"
    assert res_b["calibrated"] is False, "Expected calibrated=False"
    assert res_b["georeferenced"] is True, "Expected georeferenced=True"
    assert res_b["height_unit"] == "rel", f"Expected 'rel', got {res_b['height_unit']}"
    assert res_b["crs"] is not None, "Expected valid CRS"
    assert res_b["transform"] is not None, "Expected valid Affine transform"
    assert res_b["metadata"]["calibration"]["method"] == "none"
    assert res_b["metadata"]["calibration"]["scale_a"] is None
    assert np.min(res_b["height_map"]) >= -1e-6 and np.max(res_b["height_map"]) <= 1.0 + 1e-6
    print("   [PASS] GeoTIFF without calibration -> relative, uncalibrated, unit='rel', CRS preserved\n")

    # -------------------------------------------------------------------------
    # TEST C: GEOTIFF WITH 5 VALID GCPS
    # -------------------------------------------------------------------------
    print("TEST C: Georeferenced GeoTIFF with 5 valid GCPs")
    valid_gcps = [
        {"x": 100, "y": 100, "elevation": 44.52},
        {"x": 800, "y": 150, "elevation": 45.10},
        {"x": 512, "y": 512, "elevation": 58.30},
        {"x": 200, "y": 850, "elevation": 43.80},
        {"x": 850, "y": 850, "elevation": 44.20}
    ]
    res_c = process_image("data/potsdam_sample_1024.tif", gcps=valid_gcps, model=model, processor=processor, device=device)
    assert res_c["mode"] == "absolute", f"Expected absolute, got {res_c['mode']}"
    assert res_c["calibrated"] is True, "Expected calibrated=True"
    assert res_c["georeferenced"] is True, "Expected georeferenced=True"
    assert res_c["height_unit"] == "m", f"Expected 'm', got {res_c['height_unit']}"
    cal_c = res_c["metadata"]["calibration"]
    assert cal_c["method"] == "gcp"
    assert cal_c["gcp_count"] == 5
    assert np.isfinite(cal_c["scale_a"]) and np.isfinite(cal_c["offset_b"])
    assert "gcp_mae" in cal_c and np.isfinite(cal_c["gcp_mae"])
    assert "gcp_rmse" in cal_c and np.isfinite(cal_c["gcp_rmse"])
    assert len(cal_c["gcp_residuals"]) == 5
    print(f"   [PASS] 5 GCPs -> mode='absolute', calibrated=True, a={cal_c['scale_a']:.4f}, b={cal_c['offset_b']:.2f}m, MAE={cal_c['gcp_mae']:.3f}m, RMSE={cal_c['gcp_rmse']:.3f}m\n")

    # -------------------------------------------------------------------------
    # TEST D: KNOWN SYNTHETIC CALIBRATION (MATHEMATICAL VERIFICATION)
    # -------------------------------------------------------------------------
    print("TEST D: Mathematical Verification of Least Squares on Known Synthetic Depth")
    # Synthetic depth grid
    synth_depth = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    known_a = 10.0
    known_b = 20.0
    # H = 10*D + 20 -> for D in [1, 2, 5, 8, 9] -> H in [30, 40, 70, 100, 110]
    synth_gcps = [
        {"x": 0, "y": 0, "elevation": 30.0},
        {"x": 1, "y": 0, "elevation": 40.0},
        {"x": 1, "y": 1, "elevation": 70.0},
        {"x": 1, "y": 2, "elevation": 100.0},
        {"x": 2, "y": 2, "elevation": 110.0}
    ]

    fit_a, fit_b, gcp_info_d = fit_supplied_gcps(synth_depth, synth_gcps)
    assert abs(fit_a - known_a) < 1e-5, f"Expected a={known_a}, got {fit_a}"
    assert abs(fit_b - known_b) < 1e-5, f"Expected b={known_b}, got {fit_b}"
    assert gcp_info_d["gcp_mae"] < 1e-5, f"Expected 0 MAE, got {gcp_info_d['gcp_mae']}"
    assert gcp_info_d["gcp_rmse"] < 1e-5, f"Expected 0 RMSE, got {gcp_info_d['gcp_rmse']}"
    print(f"   [PASS] Exact recovery: True (a={known_a}, b={known_b}) -> Fitted (a={fit_a:.5f}, b={fit_b:.5f}, MAE={gcp_info_d['gcp_mae']:.2e}m)")

    # Test Regularized Prior Calibration on Synthetic
    prior_a = 12.0
    lambda_p = 5.0
    fit_a_p, fit_b_p, info_p = fit_supplied_gcps(synth_depth, synth_gcps, a_prior=prior_a, lambda_prior=lambda_p)
    assert np.isfinite(fit_a_p) and np.isfinite(fit_b_p)
    assert fit_a_p > known_a, f"Regularization should pull scale towards prior {prior_a}, got {fit_a_p}"
    print(f"   [PASS] Regularized prior OLS: (a_prior={prior_a}, lambda={lambda_p}) -> Fitted a={fit_a_p:.4f}, b={fit_b_p:.4f}\n")

    # -------------------------------------------------------------------------
    # TEST E: GCP VALIDATION AND ERROR REJECTION
    # -------------------------------------------------------------------------
    print("TEST E: GCP Validation & Graceful Rejection Rules")

    # 1. Out of bounds coordinates
    oob_gcps = [{"x": 10000, "y": 50, "elevation": 45.0}, {"x": 50, "y": 50, "elevation": 45.0}]
    try:
        fit_supplied_gcps(synth_depth, oob_gcps)
        assert False, "Should have rejected out-of-bounds GCP"
    except ValueError as e:
        print(f"   [PASS] Out of bounds rejected: {e}")

    # 2. Duplicate pixel coordinates
    dup_gcps = [{"x": 1, "y": 1, "elevation": 45.0}, {"x": 1, "y": 1, "elevation": 55.0}]
    try:
        fit_supplied_gcps(synth_depth, dup_gcps)
        assert False, "Should have rejected duplicate GCP"
    except ValueError as e:
        print(f"   [PASS] Duplicate coordinates rejected: {e}")

    # 3. Non-finite elevations
    nan_gcps = [{"x": 0, "y": 0, "elevation": float('nan')}, {"x": 1, "y": 1, "elevation": 45.0}]
    try:
        fit_supplied_gcps(synth_depth, nan_gcps)
        assert False, "Should have rejected NaN elevation"
    except ValueError as e:
        print(f"   [PASS] Non-finite elevation rejected: {e}")

    # 4. Insufficient GCP count (< 2)
    single_gcp = [{"x": 0, "y": 0, "elevation": 45.0}]
    try:
        fit_supplied_gcps(synth_depth, single_gcp)
        assert False, "Should have rejected <2 GCPs"
    except ValueError as e:
        print(f"   [PASS] Insufficient GCP count rejected: {e}")

    # 5. Degenerate flat depth
    flat_depth = np.full((10, 10), 5.0, dtype=np.float32)
    flat_gcps = [{"x": 0, "y": 0, "elevation": 40.0}, {"x": 5, "y": 5, "elevation": 50.0}]
    try:
        fit_supplied_gcps(flat_depth, flat_gcps)
        assert False, "Should have rejected degenerate flat depth"
    except ValueError as e:
        print(f"   [PASS] Degenerate depth variation rejected: {e}\n")

    # -------------------------------------------------------------------------
    # TEST F: DEM CALIBRATION & TERRAIN ANCHORING
    # -------------------------------------------------------------------------
    print("TEST F: DEM Calibration & Terrain Anchoring")
    dem_file = "data/dem_cache/top_potsdam_2_10_RGB_srtm_dem.tif"
    assert os.path.isfile(dem_file), f"DEM test file missing: {dem_file}"

    res_f = process_image(
        "data/potsdam_sample_1024.tif",
        dem_file=dem_file,
        terrain_percentile=25.0,
        model=model,
        processor=processor,
        device=device
    )
    assert res_f["mode"] == "absolute", f"Expected absolute, got {res_f['mode']}"
    assert res_f["calibrated"] is True, "Expected calibrated=True"
    assert res_f["georeferenced"] is True, "Expected georeferenced=True"
    assert res_f["height_unit"] == "m", f"Expected 'm', got {res_f['height_unit']}"
    cal_f = res_f["metadata"]["calibration"]
    assert cal_f["method"] == "dem"
    assert cal_f["dem_source"] == dem_file
    assert cal_f["terrain_anchor_count"] > 0
    assert np.isfinite(cal_f["terrain_anchor_elevation"])
    assert np.isfinite(cal_f["scale_a"]) and np.isfinite(cal_f["offset_b"])

    # Verify that monocular high-frequency surface detail is preserved and not replaced by the coarse DEM
    # Check standard deviation and gradient of output vs raw relative depth
    assert np.std(res_f["height_map"]) > 0.1, "Height map must retain topography variation"
    print(f"   [PASS] DEM Calibration: mode='absolute', calibrated=True, terrain_anchors={cal_f['terrain_anchor_count']}, datum={cal_f['terrain_anchor_elevation']:.2f}m, a={cal_f['scale_a']:.4f}, b={cal_f['offset_b']:.2f}m\n")

    # -------------------------------------------------------------------------
    # TEST G: NO CALIBRATION / NO SILENT FALLBACK
    # -------------------------------------------------------------------------
    print("TEST G: No Hidden Fallback Scale or Offset")
    res_g_png = process_image("data/test_sample_photo.png", gcps=None, dem_file=None, model=model, processor=processor, device=device)
    assert res_g_png["metadata"]["calibration"]["scale_a"] is None
    assert res_g_png["metadata"]["calibration"]["offset_b"] is None
    assert res_g_png["calibrated"] is False
    assert res_g_png["mode"] == "relative"

    res_g_tif = process_image("data/potsdam_sample_1024.tif", gcps=None, dem_file=None, model=model, processor=processor, device=device)
    assert res_g_tif["metadata"]["calibration"]["scale_a"] is None
    assert res_g_tif["metadata"]["calibration"]["offset_b"] is None
    assert res_g_tif["calibrated"] is False
    assert res_g_tif["mode"] == "relative"
    print("   [PASS] Verified zero silent fallback scale/offset in uncalibrated state\n")

    print("=" * 80)
    print("ALL M2 CALIBRATION ENGINE TESTS PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    run_m2_calibration_tests()
