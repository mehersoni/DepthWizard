"""
Integration test suite for FastAPI API Server (api_server.py)
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_server import app


def test_api_server_endpoints():
    print("=" * 80)
    print("DepthWizard M5 — API Server Integration Tests")
    print("=" * 80)

    client = TestClient(app)

    # 1. Test GET / and GET /health
    print("\n[1/4] Testing GET / and GET /health...")
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "DepthWizard" in resp_root.text
    print("      [PASS] GET / serves M5 3D WebGL viewer successfully")

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "online"
    assert "DepthWizard" in health_data["service"]
    print(f"      [PASS] GET /health -> {health_data}")

    # 2. Test POST /process GeoTIFF WITHOUT Calibration (Uncalibrated Mode)
    print("\n[2/4] Testing POST /process GeoTIFF WITHOUT Calibration (Uncalibrated Mode)...")
    potsdam_sample = "data/potsdam_sample_1024.tif"
    if not os.path.exists(potsdam_sample):
        potsdam_sample = "data/potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif"

    with open(potsdam_sample, "rb") as f:
        files = {"file": ("potsdam.tif", f, "image/tiff")}
        data = {"visual_size": "256"}
        resp = client.post("/process", files=files, data=data)

    assert resp.status_code == 200, f"Error: {resp.text}"
    d_unc = resp.json()
    assert d_unc["mode"] == "relative", f"Expected mode 'relative', got {d_unc['mode']}"
    assert d_unc["calibrated"] is False, f"Expected calibrated=False, got {d_unc['calibrated']}"
    assert d_unc["georeferenced"] is True, f"Expected georeferenced=True, got {d_unc['georeferenced']}"
    assert d_unc["height_unit"] == "rel", f"Expected height_unit='rel', got {d_unc['height_unit']}"
    assert len(d_unc["height_map"]) == 256 * 256
    assert d_unc["crs"] is not None, "CRS must be preserved"
    print(f"      [PASS] POST /process (Uncalibrated): mode={d_unc['mode']}, calibrated={d_unc['calibrated']}, georeferenced={d_unc['georeferenced']}, unit='{d_unc['height_unit']}', span=[{d_unc['height_min']:.2f}, {d_unc['height_max']:.2f}]")

    # 3. Test POST /process GeoTIFF WITH GCPs (Calibrated Mode)
    print("\n[3/4] Testing POST /process GeoTIFF WITH GCPs (Metric Calibrated Mode)...")
    gcps = [
        {"x": 100, "y": 100, "elevation": 45.2},
        {"x": 800, "y": 200, "elevation": 44.8},
        {"x": 500, "y": 500, "elevation": 58.5},
        {"x": 200, "y": 800, "elevation": 43.9},
        {"x": 850, "y": 850, "elevation": 44.1}
    ]

    with open(potsdam_sample, "rb") as f:
        files = {"file": ("potsdam.tif", f, "image/tiff")}
        data = {
            "visual_size": "256",
            "gcps": json.dumps(gcps)
        }
        resp = client.post("/process", files=files, data=data)

    assert resp.status_code == 200, f"Error: {resp.text}"
    d_gcp = resp.json()
    assert d_gcp["mode"] == "absolute"
    assert d_gcp["calibrated"] is True
    assert d_gcp["georeferenced"] is True
    assert d_gcp["height_unit"] == "m"
    assert d_gcp["metadata"]["calibration"]["method"] == "gcp"
    assert d_gcp["metadata"]["calibration"]["gcp_count"] == 5
    assert d_gcp["height_min"] > 25.0
    print(f"      [PASS] POST /process (GCPs): mode={d_gcp['mode']}, calibrated={d_gcp['calibrated']}, unit='{d_gcp['height_unit']}', span=[{d_gcp['height_min']:.2f}m, {d_gcp['height_max']:.2f}m]")

    # 4. Test POST /process Non-georeferenced PNG Photo
    print("\n[4/4] Testing POST /process Non-georeferenced Photo (PNG)...")
    png_path = "data/test_sample_photo.png"
    if not os.path.exists(png_path):
        img_data = np.zeros((300, 400, 3), dtype=np.uint8)
        img_data[:, :] = [120, 150, 100]
        Image.fromarray(img_data).save(png_path)

    with open(png_path, "rb") as f:
        files = {"file": ("photo.png", f, "image/png")}
        data = {"visual_size": "256"}
        resp = client.post("/process", files=files, data=data)

    assert resp.status_code == 200, f"Error: {resp.text}"
    d_png = resp.json()
    assert d_png["mode"] == "relative"
    assert d_png["calibrated"] is False
    assert d_png["georeferenced"] is False
    assert d_png["height_unit"] == "rel"
    assert d_png["crs"] is None
    assert d_png["metadata"]["calibration"]["method"] == "none"
    print(f"      [PASS] POST /process (PNG): mode={d_png['mode']}, calibrated={d_png['calibrated']}, georeferenced={d_png['georeferenced']}, unit='{d_png['height_unit']}', CRS=None, span=[{d_png['height_min']:.2f}, {d_png['height_max']:.2f}]")

    print("\n" + "=" * 80)
    print("ALL API SERVER INTEGRATION TESTS PASSED (100% SUCCESS)!")
    print("=" * 80)


if __name__ == "__main__":
    test_api_server_endpoints()
