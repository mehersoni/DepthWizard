import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from calibration.metric import fit_scale_offset, apply_scale_offset, calibrate_depth_to_dsm

def test_exact_affine_recovery():
    np.random.seed(42)
    depth = np.random.uniform(1.0, 50.0, (100, 100)).astype(np.float32)
    true_a = 5.0
    true_b = 10.0
    ref_dsm = true_a * depth + true_b
    
    a, b, valid_mask = fit_scale_offset(depth, ref_dsm)
    assert np.all(valid_mask)
    assert np.isclose(a, true_a, atol=1e-5), f'Expected a={true_a}, got {a}'
    assert np.isclose(b, true_b, atol=1e-5), f'Expected b={true_b}, got {b}'
    
    pred_dsm = apply_scale_offset(depth, a, b)
    assert np.allclose(pred_dsm, ref_dsm, atol=1e-4)
    print(f'[PASS] test_exact_affine_recovery: a={a:.4f}, b={b:.4f}')

def test_noisy_affine_recovery():
    np.random.seed(42)
    depth = np.linspace(5.0, 40.0, 2500).reshape((50, 50)).astype(np.float32)
    true_a = 3.5
    true_b = 25.0
    noise = np.random.normal(0, 0.05, depth.shape).astype(np.float32)
    ref_dsm = true_a * depth + true_b + noise
    
    a, b, valid_mask = fit_scale_offset(depth, ref_dsm)
    assert np.isclose(a, true_a, atol=0.05), f'Expected a~{true_a}, got {a}'
    assert np.isclose(b, true_b, atol=0.1), f'Expected b~{true_b}, got {b}'
    print(f'[PASS] test_noisy_affine_recovery: recovered a={a:.4f}, b={b:.4f}')

def test_nodata_and_nan_masking():
    np.random.seed(42)
    depth = np.random.uniform(2.0, 30.0, (60, 60)).astype(np.float32)
    true_a = 8.2
    true_b = -4.5
    ref_dsm = true_a * depth + true_b
    
    # Inject corrupted/NoData pixels
    ref_dsm[0:5, 0:5] = -9999.0
    ref_dsm[10, 10] = np.nan
    ref_dsm[20, 20] = np.inf
    depth[30, 30] = np.nan
    depth[40, 40] = np.inf
    
    cal_dsm, a, b, valid_mask = calibrate_depth_to_dsm(depth, ref_dsm, nodata=-9999.0)
    assert not valid_mask[0, 0]
    assert not valid_mask[10, 10]
    assert not valid_mask[20, 20]
    assert not valid_mask[30, 30]
    assert not valid_mask[40, 40]
    assert np.isclose(a, true_a, atol=1e-4)
    assert np.isclose(b, true_b, atol=1e-4)
    print(f'[PASS] test_nodata_and_nan_masking: masked {np.sum(~valid_mask)} invalid pixels successfully')

def test_shape_mismatch():
    d = np.ones((50, 50), dtype=np.float32)
    h = np.ones((60, 60), dtype=np.float32)
    try:
        fit_scale_offset(d, h)
        assert False, 'Should have raised ValueError on shape mismatch'
    except ValueError as e:
        print('[PASS] test_shape_mismatch caught as expected:', e)

if __name__ == '__main__':
    test_exact_affine_recovery()
    test_noisy_affine_recovery()
    test_nodata_and_nan_masking()
    test_shape_mismatch()
    print('\nALL METRIC CALIBRATION TESTS PASSED!')
