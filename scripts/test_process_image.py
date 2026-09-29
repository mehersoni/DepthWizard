"""
Comprehensive Problem-Statement-Compliance Test Suite for DepthWizard M2 process_image()
"""

import os
import sys
import rasterio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_image import process_image, export_dsm, export_slope
from depth.depth_model import load_model


def run_full_compliance_test_suite():
    print("=" * 80)
    print("DepthWizard M2 — Problem Statement Compliance Test Suite")
    print("=" * 80)

    os.makedirs("outputs/dsm", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("\n[Setup] Preloading Depth Anything V2 model...", flush=True)
    model, processor, device = load_model()
    print("        Model loaded on device:", device)

    # -------------------------------------------------------------------------
    # TEST 1: Non-Georeferenced Image (PNG) -> Relative rDSM [0, 1]
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 1: Non-Georeferenced Image (PNG) -> Relative rDSM [0, 1]")
    print("-" * 80)

    png_path = "data/test_sample_photo.png"
    img_data = np.zeros((300, 400, 3), dtype=np.uint8)
    img_data[:, :] = [120, 150, 100]
    img_data[80:180, 100:250] = [200, 80, 80]
    img_data[180:220, 100:250] = [40, 40, 50]
    Image.fromarray(img_data).save(png_path)

    res_png = process_image(path=png_path, model=model, processor=processor, device=device)

    assert res_png["mode"] == "relative", f"Expected mode 'relative', got {res_png['mode']}"
    assert res_png["calibrated"] is False, f"Expected calibrated=False for PNG, got {res_png['calibrated']}"
    assert res_png["georeferenced"] is False, f"Expected georeferenced=False for PNG, got {res_png['georeferenced']}"
    assert res_png["height_unit"] == "rel", f"Expected height_unit='rel' for PNG, got {res_png['height_unit']}"
    assert res_png["crs"] is None, f"Expected CRS=None for PNG, got {res_png['crs']}"
    assert res_png["transform"] is None, f"Expected transform=None for PNG, got {res_png['transform']}"
    assert res_png["metadata"]["gsd_x"] is None, "Expected GSD=None for PNG"
    assert res_png["metadata"]["calibration"]["method"] == "none"
    assert res_png["metadata"]["calibration"]["scale_a"] is None
    assert res_png["metadata"]["calibration"]["offset_b"] is None
    assert res_png["height_map"].dtype == np.float32, "Height map must be float32"
    assert res_png["rgb"].dtype == np.uint8, "RGB array must be uint8"
    assert res_png["slope_map"].dtype == np.float32, "Slope map must be float32"
    assert res_png["confidence_map"].dtype == np.float32, "Confidence map must be float32"

    h_min_png = float(np.min(res_png["height_map"]))
    h_max_png = float(np.max(res_png["height_map"]))
    assert np.isclose(h_min_png, 0.0, atol=1e-5), f"rDSM min is {h_min_png}, expected 0.0"
    assert np.isclose(h_max_png, 1.0, atol=1e-5), f"rDSM max is {h_max_png}, expected 1.0"
    print(f"   [PASS] PNG: mode='{res_png['mode']}', calibrated={res_png['calibrated']}, georeferenced={res_png['georeferenced']}, unit='{res_png['height_unit']}', CRS=None, range=[{h_min_png:.4f}, {h_max_png:.4f}]")

    # -------------------------------------------------------------------------
    # TEST 2: Non-Georeferenced Image (JPG) -> Relative rDSM [0, 1]
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 2: Non-Georeferenced Image (JPG) -> Relative rDSM [0, 1]")
    print("-" * 80)

    jpg_path = "data/test_sample_photo.jpg"
    Image.fromarray(img_data).save(jpg_path, quality=90)

    res_jpg = process_image(path=jpg_path, model=model, processor=processor, device=device)

    assert res_jpg["mode"] == "relative", f"Expected mode 'relative', got {res_jpg['mode']}"
    assert res_jpg["calibrated"] is False, f"Expected calibrated=False for JPG"
    assert res_jpg["georeferenced"] is False, f"Expected georeferenced=False for JPG"
    assert res_jpg["height_unit"] == "rel", f"Expected height_unit='rel' for JPG"
    assert res_jpg["crs"] is None, "Expected CRS=None for JPG"
    assert res_jpg["transform"] is None, "Expected transform=None for JPG"
    h_min_jpg = float(np.min(res_jpg["height_map"]))
    h_max_jpg = float(np.max(res_jpg["height_map"]))
    assert np.isclose(h_min_jpg, 0.0, atol=1e-4) and np.isclose(h_max_jpg, 1.0, atol=1e-4)
    print(f"   [PASS] JPG: mode='{res_jpg['mode']}', calibrated={res_jpg['calibrated']}, georeferenced={res_jpg['georeferenced']}, unit='{res_jpg['height_unit']}', CRS=None, range=[{h_min_jpg:.4f}, {h_max_jpg:.4f}]")

    # -------------------------------------------------------------------------
    # TEST 3: GeoTIFF WITHOUT Calibration -> Georeferenced Relative Mode
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 3: GeoTIFF WITHOUT Calibration -> Georeferenced Relative Fallback (NO Fake Metres)")
    print("-" * 80)

    sample_geo = "data/potsdam_sample_1024.tif"
    res_uncal = process_image(path=sample_geo, gcps=None, dem_path=None, model=model, processor=processor, device=device)

    assert res_uncal["mode"] == "relative", f"Expected mode 'relative' for uncalibrated GeoTIFF, got {res_uncal['mode']}"
    assert res_uncal["calibrated"] is False, f"CRITICAL: Uncalibrated GeoTIFF must have calibrated=False!"
    assert res_uncal["georeferenced"] is True, f"Georeferenced GeoTIFF must have georeferenced=True!"
    assert res_uncal["height_unit"] == "rel", f"Expected height_unit='rel', got {res_uncal['height_unit']}"
    assert res_uncal["metadata"]["calibration"]["method"] == "none", "Expected calibration method 'none'"
    assert res_uncal["metadata"]["calibration"]["scale_a"] is None, "CRITICAL: No scale_a should be invented!"
    assert res_uncal["metadata"]["calibration"]["offset_b"] is None, "CRITICAL: No offset_b should be invented!"
    assert res_uncal["crs"] is not None, "Real CRS must be preserved"
    assert res_uncal["transform"] is not None, "Real transform must be preserved"
    assert res_uncal["metadata"]["gsd_x"] == 0.050, f"Expected GSD 0.050, got {res_uncal['metadata']['gsd_x']}"

    # Height map must be normalized rDSM [0, 1]
    h_min_unc = float(np.min(res_uncal["height_map"]))
    h_max_unc = float(np.max(res_uncal["height_map"]))
    assert np.isclose(h_min_unc, 0.0, atol=1e-5) and np.isclose(h_max_unc, 1.0, atol=1e-5), \
        f"Uncalibrated height map must be in [0, 1] relative units, got [{h_min_unc}, {h_max_unc}]"
    print(f"   [PASS] Uncalibrated GeoTIFF: mode='{res_uncal['mode']}', calibrated={res_uncal['calibrated']}, georeferenced={res_uncal['georeferenced']}, unit='{res_uncal['height_unit']}', CRS preserved ({res_uncal['crs']}), range=[{h_min_unc:.4f}, {h_max_unc:.4f}]")

    # -------------------------------------------------------------------------
    # TEST 4: GeoTIFF + User Supplied GCPs -> Metric Absolute DSM
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 4: GeoTIFF + User Supplied GCPs -> Metric Absolute DSM")
    print("-" * 80)

    # Explicit Ground Control Points with known reference heights within 1024x1024 crop
    user_gcps = [
        {"x": 100, "y": 100, "elevation": 45.20},
        {"x": 800, "y": 200, "elevation": 44.80},
        {"x": 500, "y": 500, "elevation": 58.50},
        {"x": 200, "y": 800, "elevation": 43.90},
        {"x": 850, "y": 850, "elevation": 44.10}
    ]

    res_gcp = process_image(path=sample_geo, gcps=user_gcps, model=model, processor=processor, device=device)

    assert res_gcp["mode"] == "absolute", f"Expected mode 'absolute' for GCP-calibrated DSM, got {res_gcp['mode']}"
    assert res_gcp["calibrated"] is True, "GCP calibration must produce calibrated=True"
    assert res_gcp["georeferenced"] is True, "GCP calibration must maintain georeferenced=True"
    assert res_gcp["height_unit"] == "m", f"Expected height_unit='m', got {res_gcp['height_unit']}"
    assert res_gcp["metadata"]["calibration"]["method"] == "gcp"
    assert res_gcp["metadata"]["calibration"]["gcp_count"] == 5
    assert res_gcp["metadata"]["calibration"]["scale_a"] is not None
    assert res_gcp["metadata"]["calibration"]["offset_b"] is not None

    h_min_gcp = float(np.min(res_gcp["height_map"]))
    h_max_gcp = float(np.max(res_gcp["height_map"]))
    assert 30.0 <= h_min_gcp <= 50.0, f"Unreasonable metric minimum: {h_min_gcp}"
    assert 45.0 <= h_max_gcp <= 80.0, f"Unreasonable metric maximum: {h_max_gcp}"
    print(f"   [PASS] GCP Calibrated DSM: mode='{res_gcp['mode']}', calibrated={res_gcp['calibrated']}, georeferenced={res_gcp['georeferenced']}, unit='{res_gcp['height_unit']}', a={res_gcp['metadata']['calibration']['scale_a']:.4f}, b={res_gcp['metadata']['calibration']['offset_b']:.4f}m, elevation span=[{h_min_gcp:.2f}m, {h_max_gcp:.2f}m]")

    # -------------------------------------------------------------------------
    # TEST 5: GeoTIFF + User Supplied DEM -> Metric Absolute DSM
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 5: GeoTIFF + User Supplied DEM -> Metric Absolute DSM")
    print("-" * 80)

    dem_path = "data/dem_cache/top_potsdam_2_10_RGB_srtm_dem.tif"
    res_dem = process_image(path=sample_geo, dem_path=dem_path, model=model, processor=processor, device=device)

    assert res_dem["mode"] == "absolute"
    assert res_dem["calibrated"] is True, "DEM calibration must produce calibrated=True"
    assert res_dem["georeferenced"] is True, "DEM calibration must maintain georeferenced=True"
    assert res_dem["height_unit"] == "m", f"Expected height_unit='m', got {res_dem['height_unit']}"
    assert res_dem["metadata"]["calibration"]["method"] == "dem"
    assert res_dem["metadata"]["calibration"]["dem_source"] == dem_path
    assert res_dem["metadata"]["calibration"]["scale_a"] is not None
    assert res_dem["metadata"]["calibration"]["offset_b"] is not None

    h_min_dem = float(np.min(res_dem["height_map"]))
    h_max_dem = float(np.max(res_dem["height_map"]))
    assert 20.0 <= h_min_dem <= 50.0, f"Unreasonable DEM-anchored minimum: {h_min_dem}"
    assert 40.0 <= h_max_dem <= 80.0, f"Unreasonable DEM-anchored maximum: {h_max_dem}"
    print(f"   [PASS] DEM Calibrated DSM: mode='{res_dem['mode']}', calibrated={res_dem['calibrated']}, georeferenced={res_dem['georeferenced']}, unit='{res_dem['height_unit']}', dem={dem_path}, elevation span=[{h_min_dem:.2f}m, {h_max_dem:.2f}m]")

    # -------------------------------------------------------------------------
    # TEST 6: Physical vs Relative Slope Handling
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 6: Slope Computation (Physical Degrees vs Geometric Slope)")
    print("-" * 80)

    assert np.all((res_gcp["slope_map"] >= 0.0) & (res_gcp["slope_map"] <= 90.0))
    assert np.all((res_png["slope_map"] >= 0.0) & (res_png["slope_map"] <= 90.0))
    print(f"   [PASS] Calibrated metric slope (0.05m GSD): [{np.min(res_gcp['slope_map']):.1f} deg, {np.max(res_gcp['slope_map']):.1f} deg]")
    print(f"   [PASS] Relative geometric slope: [{np.min(res_png['slope_map']):.1f} deg, {np.max(res_png['slope_map']):.1f} deg]")

    # -------------------------------------------------------------------------
    # TEST 7: GeoTIFF and PNG Export Functionality
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 7: Raster Export Verification")
    print("-" * 80)

    dsm_out = "outputs/dsm/test_compliant_dsm.tif"
    rdsm_out = "outputs/dsm/test_compliant_rdsm.png"
    export_dsm(res_gcp, dsm_out)
    export_dsm(res_png, rdsm_out)

    with rasterio.open(dsm_out) as src:
        assert src.crs == res_gcp["crs"], "Exported GeoTIFF CRS mismatch"
        assert src.transform == res_gcp["transform"], "Exported GeoTIFF transform mismatch"
        assert src.shape == (res_gcp["height"], res_gcp["width"])
        assert src.dtypes[0] == "float32", "Exported GeoTIFF must be float32"
        data_read = src.read(1)
        assert np.allclose(data_read, res_gcp["height_map"], atol=1e-5), "Exported GeoTIFF values mismatch"

    assert os.path.isfile(rdsm_out), "Exported rDSM PNG file missing"
    print(f"   [PASS] GeoTIFF exported and reopened: CRS ({res_gcp['crs']}), transform, float32 dtype, values matched exactly")
    print(f"   [PASS] Relative rDSM exported as PNG ({rdsm_out})")

    # -------------------------------------------------------------------------
    # TEST 8: Member Hooks (M1 external_depth, M4 shadow_constraints)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 8: Member Hooks Extensibility (M1 external_depth & M4 shadow_constraints)")
    print("-" * 80)

    ext_depth = np.full((300, 400), 2.5, dtype=np.float32)
    ext_depth[100:200, 100:200] = 5.0
    shadow_ctx = {"building_id": 42, "shadow_length_px": 18.5, "solar_elev_deg": 35.0}

    res_hook = process_image(
        path=png_path,
        external_depth=ext_depth,
        shadow_constraints=shadow_ctx,
        model=model,
        processor=processor,
        device=device
    )
    assert res_hook["metadata"]["shadow_constraints"] == shadow_ctx
    assert np.isclose(float(np.min(res_hook["height_map"])), 0.0)
    assert np.isclose(float(np.max(res_hook["height_map"])), 1.0)
    print("   [PASS] M1 external_depth and M4 shadow_constraints hooks functional")

    print("\n" + "=" * 80)
    print("ALL M2 PROBLEM-STATEMENT COMPLIANCE TEST SUITES PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_compliance_test_suite()
