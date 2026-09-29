import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from calibration.rdsm import make_rdsm, save_rdsm

def test_normal_depth():
    np.random.seed(42)
    depth = np.linspace(10, 100, 10000).reshape((100, 100)).astype(np.float32)
    rdsm = make_rdsm(depth)
    assert rdsm.shape == (100, 100)
    assert rdsm.dtype == np.float32
    assert np.all(rdsm >= 0.0)
    assert np.all(rdsm <= 1.0)
    assert np.min(rdsm) == 0.0
    assert np.max(rdsm) == 1.0
    print('[PASS] test_normal_depth')

def test_constant_depth():
    depth = np.full((64, 64), 42.0, dtype=np.float32)
    rdsm = make_rdsm(depth)
    assert rdsm.shape == (64, 64)
    assert np.all(rdsm == 0.0)
    print('[PASS] test_constant_depth')

def test_outliers():
    np.random.seed(42)
    depth = np.random.uniform(20.0, 80.0, (100, 100)).astype(np.float32)
    # Add extreme outliers
    depth[0, 0] = -999999.0
    depth[0, 1] = 999999.0
    rdsm = make_rdsm(depth, p_low=2.0, p_high=98.0)
    assert rdsm[0, 0] == 0.0
    assert rdsm[0, 1] == 1.0
    assert np.all(rdsm >= 0.0)
    assert np.all(rdsm <= 1.0)
    print('[PASS] test_outliers')

def test_nan_and_inf_handling():
    depth = np.linspace(10, 50, 1600).reshape((40, 40)).astype(np.float32)
    depth[5, 5] = np.nan
    depth[10, 10] = np.inf
    depth[15, 15] = -np.inf
    rdsm = make_rdsm(depth)
    assert not np.isnan(rdsm).any()
    assert not np.isinf(rdsm).any()
    assert rdsm[5, 5] == 0.0
    assert rdsm[10, 10] == 0.0
    assert rdsm[15, 15] == 0.0
    assert np.all(rdsm >= 0.0)
    assert np.all(rdsm <= 1.0)
    print('[PASS] test_nan_and_inf_handling')

def test_save_rdsm():
    rdsm = np.random.uniform(0.0, 1.0, (50, 50)).astype(np.float32)
    out_path = 'outputs/rdsm/test_sample_rdsm'
    res = save_rdsm(rdsm, out_path, save_png=True)
    assert os.path.exists(res['npy'])
    assert os.path.exists(res['png'])
    loaded = np.load(res['npy'])
    assert np.allclose(rdsm, loaded)
    print('[PASS] test_save_rdsm')

if __name__ == '__main__':
    test_normal_depth()
    test_constant_depth()
    test_outliers()
    test_nan_and_inf_handling()
    test_save_rdsm()
    print('\nALL RDSM UNIT TESTS PASSED!')
