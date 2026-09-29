import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from evaluation.metrics import calculate_metrics, compute_error_map

def test_exact_match():
    ref = np.linspace(10.0, 50.0, 10000).reshape((100, 100)).astype(np.float32)
    pred = ref.copy()
    
    m = calculate_metrics(pred, ref)
    assert np.isclose(m['mae'], 0.0, atol=1e-6)
    assert np.isclose(m['rmse'], 0.0, atol=1e-6)
    assert np.isclose(m['valid_pixels'], 10000)
    assert np.isclose(m['variance_pred'] if 'variance_pred' in m else m['valid_pixels'], 10000)
    print(f'[PASS] test_exact_match: MAE={m["mae"]}, RMSE={m["rmse"]}, R2={m["r2"]}, r={m["correlation"]}')

def test_constant_offset():
    ref = np.linspace(10.0, 50.0, 10000).reshape((100, 100)).astype(np.float32)
    offset = 3.5
    pred = ref + offset
    
    m = calculate_metrics(pred, ref)
    assert np.isclose(m['mae'], offset, atol=1e-5)
    assert np.isclose(m['valid_pixels'], 10000)
    assert np.isclose(m['variance_pred'] if 'variance_pred' in m else m['valid_pixels'], 10000)
    assert np.isclose(m["rmse"], offset, atol=1e-5)
    assert np.isclose(m["correlation"], 1.0, atol=1e-5)
    print(f'[PASS] test_constant_offset: MAE={m["mae"]:.2f}, RMSE={m["rmse"]:.2f}, R2={m["r2"]:.2f}')

def test_nan_inf_nodata_handling():
    np.random.seed(42)
    ref = np.random.uniform(10.0, 40.0, (50, 50)).astype(np.float32)
    pred = ref + np.random.normal(0, 0.5, ref.shape).astype(np.float32)
    ref[0, 0] = -9999.0
    ref[1, 1] = np.nan
    ref[2, 2] = np.inf
    pred[3, 3] = np.nan
    pred[4, 4] = -np.inf
    m = calculate_metrics(pred, ref, nodata=-9999.0)
    assert m['valid_pixels'] == (50 * 50 - 5)
    assert np.isfinite(m['mae'])
    assert np.isfinite(m["rmse"])
    assert np.isfinite(m["correlation"])
    assert np.isfinite(m["r2"])
    print(f'[PASS] test_nan_inf_nodata_handling: {m["valid_pixels"]} pixels validated')

def test_error_map():
    ref = np.array([[10.0, 20.0], [30.0, -9999.0]], dtype=np.float32)
    pred = np.array([[12.0, 19.0], [35.0, 40.0]], dtype=np.float32)
    err = compute_error_map(pred, ref, nodata=-9999.0)
    assert np.isclose(err[0, 0], 2.0)
    assert np.isclose(err[0, 1], 1.0)
    assert np.isclose(err[1, 0], 5.0)
    assert np.isnan(err[1, 1])
    print(f'[PASS] test_error_map: absolute error mapped and masked correctly')

def test_insufficient_pixels():
    ref = np.array([[np.nan, np.nan], [np.nan, 10.0]], dtype=np.float32)
    pred = np.array([[np.nan, np.nan], [np.nan, 12.0]], dtype=np.float32)
    try:
        calculate_metrics(pred, ref)
        assert False, 'Should have failed'
    except ValueError as e:
        print(f'[PASS] test_insufficient_pixels caught: {e}')

if __name__ == '__main__':
    test_exact_match()
    test_constant_offset()
    test_nan_inf_nodata_handling()
    test_error_map()
    test_insufficient_pixels()
    print('\nALL EVALUATION METRIC TESTS PASSED!')
