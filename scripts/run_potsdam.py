import os
import sys
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth.depth_model import load_model, estimate_depth
from calibration.rdsm import make_rdsm
from calibration.metric import calibrate_depth_to_dsm
from evaluation.metrics import calculate_metrics, compute_error_map

def run_potsdam_pipeline(
    rgb_path='data/potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif',
    dsm_path='data/potsdam/1_DSM/dsm_potsdam_02_10.tif',
    output_dir='outputs'
):
    os.makedirs(os.path.join(output_dir, 'dsm'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f'RGB file not found: {rgb_path}')
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f'DSM file not found: {dsm_path}')

    print('[1/5] Loading Potsdam RGB and DSM tiles...')
    with rasterio.open(rgb_path) as src_rgb:
        rgb_h, rgb_w = src_rgb.height, src_rgb.width
        rgb_arr = src_rgb.read().transpose(1, 2, 0)

    with rasterio.open(dsm_path) as src_dsm:
        dsm_h, dsm_w = src_dsm.height, src_dsm.width
        dsm_arr = src_dsm.read(1)
        dsm_meta = src_dsm.meta.copy()
        dsm_nodata = src_dsm.nodata if src_dsm.nodata is not None else -9999.0

    # Automatic dimension check
    if (rgb_h, rgb_w) != (dsm_h, dsm_w):
        raise ValueError(
            f'Spatial dimension mismatch: RGB ({rgb_h}, {rgb_w}) != DSM ({dsm_h}, {dsm_w}). '
            f'Aborting to prevent silent resizing.'
        )
    print(f'  Verified matching dimensions: {rgb_h}x{rgb_w}')

    print('[2/5] Estimating relative depth via Depth Anything V2...')
    depth_map = estimate_depth(rgb_arr)

    print('[3/5] Computing normalized rDSM...')
    rdsm = make_rdsm(depth_map)

    print('[4/5] Calibrating depth to Potsdam DSM (Least Squares)...')
    pred_dsm, est_a, est_b, valid_mask = calibrate_depth_to_dsm(
        depth_map, dsm_arr, nodata=dsm_nodata
    )

    # Save predicted DSM GeoTIFF preserving CRS and transform
    out_dsm_path = os.path.join(output_dir, 'dsm', 'potsdam_predicted.tif')
    dsm_meta.update({'dtype': 'float32', 'count': 1, 'nodata': dsm_nodata})
    with rasterio.open(out_dsm_path, 'w', **dsm_meta) as dst:
        dst.write(pred_dsm.astype(np.float32), 1)
    print(f'  Saved predicted DSM GeoTIFF to: {out_dsm_path}')

    print('[5/5] Calculating metrics and generating diagnostic plots...')
    metrics = calculate_metrics(pred_dsm, dsm_arr, valid_mask=valid_mask, nodata=dsm_nodata)
    error_map = compute_error_map(pred_dsm, dsm_arr, valid_mask=valid_mask, nodata=dsm_nodata)

    metrics_summary = {
        'tile': 'potsdam_2_10',
        'calibration_parameters': {
            'scale_a': est_a,
            'offset_b': est_b
        },
        'evaluation_metrics': metrics
    }
    metrics_save_path = os.path.join(output_dir, 'metrics', 'potsdam_baseline.json')
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f'  Saved metrics JSON to: {metrics_save_path}')

    print('='*65)
    print('DepthWizard M2 - Potsdam Evaluation Results')
    print('='*65)
    print(f'Relative Depth Range:    [{np.min(depth_map):.4f}, {np.max(depth_map):.4f}]')
    print(f'Reference DSM Range:    [{np.min(dsm_arr[valid_mask]):.3f} m, {np.max(dsm_arr[valid_mask]):.3f} m]')
    print(f'Estimated Scale a:       {est_a:.4f}')
    print(f'Estimated Offset b:      {est_b:.4f} m')
    print(f'Predicted DSM Range:    [{np.min(pred_dsm[valid_mask]):.3f} m, {np.max(pred_dsm[valid_mask]):.3f} m]')
    print(f'MAE:                       {metrics["mae"]:.4f} m')
    print(f'RMSE:                      {metrics["rmse"]:.4f} m')
    print(f'Pearson Correlation (r):   {metrics["correlation"]:.4f}')
    print(f'Coefficient of Det. (R2): {metrics["r2"]:.4f}')
    print('='*65)

    # Generate 6-Panel Diagnostic Visualization (subsample by 4x for fast high-res plotting)
    step = 4
    rgb_sub = rgb_arr[::step, ::step]
    depth_sub = depth_map[::step, ::step]
    rdsm_sub = rdsm[::step, ::step]
    ref_sub = dsm_arr[::step, ::step]
    pred_sub = pred_dsm[::step, ::step]
    err_sub = error_map[::step, ::step]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)

    axes[0, 0].imshow(rgb_sub)
    axes[0, 0].set_title('1. Potsdam Ortho RGB (2_10)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(depth_sub, cmap='inferno')
    axes[0, 1].set_title('2. Depth Anything V2 Relative Depth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Relative Depth / Disparity', rotation=270, labelpad=12)

    im2 = axes[0, 2].imshow(rdsm_sub, cmap='magma', vmin=0.0, vmax=1.0)
    axes[0, 2].set_title('3. Normalized rDSM [0, 1]', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Relative Elevation', rotation=270, labelpad=12)

    d_min = float(np.min(dsm_arr[valid_mask]))
    d_max = float(np.max(dsm_arr[valid_mask]))
    im3 = axes[1, 0].imshow(ref_sub, cmap='terrain', vmin=d_min, vmax=d_max)
    axes[1, 0].set_title('4. Reference LiDAR DSM', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Elevation (m)', rotation=270, labelpad=12)

    im4 = axes[1, 1].imshow(pred_sub, cmap='terrain', vmin=d_min, vmax=d_max)
    axes[1, 1].set_title(f'5. Calibrated DSM (H={est_a:.2f}D + {est_b:.2f})', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Elevation (m)', rotation=270, labelpad=12)

    err_p99 = float(np.nanpercentile(err_sub, 99))
    im5 = axes[1, 2].imshow(err_sub, cmap='hot', vmin=0.0, vmax=err_p99)
    axes[1, 2].set_title(f'6. Absolute Error (MAE: {metrics["mae"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    cbar5 = fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar5.set_label('Error (m)', rotation=270, labelpad=12)

    plt.tight_layout()
    fig_save_path = os.path.join(output_dir, 'figures', 'potsdam_baseline.png')
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved diagnostic figure to: {fig_save_path}')

if __name__ == '__main__':
    run_potsdam_pipeline()
