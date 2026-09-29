"""
Unit Tests for Cartosat Satellite Data Ingestion & Evaluation Pipeline
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_io.cartosat_adapter import (
    load_cartosat_image,
    stretch_satellite_dynamic_range,
    extract_satellite_metadata,
)
from scripts.evaluate_cartosat import calculate_satellite_metrics


def test_satellite_dynamic_range_stretch():
    print("[TEST] test_satellite_dynamic_range_stretch...")
    # 12-bit DN range (0 to 2047)
    high_dyn = np.random.uniform(200.0, 1800.0, (100, 100)).astype(np.uint16)
    stretched = stretch_satellite_dynamic_range(high_dyn, lower_percentile=2.0, upper_percentile=98.0)

    assert stretched.shape == (100, 100)
    assert stretched.dtype == np.uint8
    assert stretched.min() == 0
    assert stretched.max() == 255
    print("  [PASS] test_satellite_dynamic_range_stretch passed.")


def test_metadata_extraction():
    print("[TEST] test_metadata_extraction...")
    tags = {
        "SOLAR_ELEVATION": "58.4",
        "SOLAR_AZIMUTH": "142.1",
        "SATELLITE": "CARTOSAT-2E",
        "SENSOR": "PAN",
        "INCIDENCE_ANGLE": "12.3"
    }
    meta = extract_satellite_metadata(tags)
    assert meta["solar_elevation"] == 58.4
    assert meta["solar_azimuth"] == 142.1
    assert meta["satellite_name"] == "CARTOSAT-2E"
    assert meta["sensor"] == "PAN"
    assert meta["incidence_angle"] == 12.3
    print("  [PASS] test_metadata_extraction passed.")


def test_cartosat_image_loader_pan():
    print("[TEST] test_cartosat_image_loader_pan...")
    test_path = "tmp/cartosat_test/cartosat_pan_sample.tif"
    if not os.path.exists(test_path):
        return

    carto = load_cartosat_image(test_path)
    assert carto["rgb"].ndim == 3
    assert carto["rgb"].shape[2] == 3
    assert carto["rgb"].dtype == np.uint8
    assert carto["is_panchromatic"] is True
    assert carto["solar_elevation"] == 54.2
    print("  [PASS] test_cartosat_image_loader_pan passed.")


def test_satellite_accuracy_metrics():
    print("[TEST] test_satellite_accuracy_metrics...")
    pred = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    ref = np.array([[10.5, 19.5], [31.0, 39.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    metrics = calculate_satellite_metrics(pred, ref, mask)
    assert "le90" in metrics
    assert "le68" in metrics
    assert "correlation" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] == 0.75
    print("  [PASS] test_satellite_accuracy_metrics passed.")


if __name__ == "__main__":
    print("=" * 65)
    print("Running Cartosat Pipeline Unit Tests")
    print("=" * 65)
    test_satellite_dynamic_range_stretch()
    test_metadata_extraction()
    test_cartosat_image_loader_pan()
    test_satellite_accuracy_metrics()
    print("\nALL CARTOSAT PIPELINE TESTS PASSED!")
