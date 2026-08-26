import os
import sys
import rasterio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calibration.rdsm import make_rdsm, save_rdsm

def run_synthetic_rdsm():
    dsm_path = os.path.join('data', 'synthetic', 'reference_dsm.tif')
    with rasterio.open(dsm_path) as src:
        gt_dsm = src.read(1)

    np.random.seed(101)
    fake_relative_depth = (gt_dsm * 2.5 + 50.0) + np.random.normal(0, 1.5, gt_dsm.shape).astype(np.float32)

    rdsm = make_rdsm(fake_relative_depth)

    out_path = os.path.join('outputs', 'rdsm', 'synthetic_rdsm')
    saved = save_rdsm(rdsm, out_path, save_png=True)
    print('Synthetic rDSM generated successfully:')
    print('  NPY:', saved['npy'], f'(shape: {rdsm.shape}, range: [{np.min(rdsm):.4f}, {np.max(rdsm):.4f}])')
    print('  PNG:', saved['png'])

if __name__ == '__main__':
    run_synthetic_rdsm()
