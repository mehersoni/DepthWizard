import os
import sys
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calibration.rdsm import make_rdsm, save_rdsm
from calibration.metric import calibrate_depth_to_dsm
from evaluation.metrics import calculate_metrics, compute_error_map

def run_baseline_pipeline(
    rgb_path='data/synthetic/rgb.png',
    dsm_path='data/synthetic/reference_dsm.tif',
    output_dir='outputs'
):
    os.makedirs(os.path.join(output_dir, 'rdsm'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dsm'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f'Missing RGB: {rgb_path}')
    rgb_img = Image.open(rgb_path)
    rgb_arr = np.array(rgb_img)

    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f'Missing reference DSM: {dsm_path}')
    with rasterio.open(dsm_path) as src:
        ref_dsm = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0

    np.random.seed(42)
    h_shape = ref_dsm.shape
    y_grid, x_grid = np.mgrid[0:h_shape[0], 0:h_shape[1]]

    true_sim_scale = 0.35
    true_sim_offset = 15.0
    noise = np.random.normal(0, 0.25, h_shape).astype(np.float32)
    spatial_warp = 0.4 * np.sin(2.0 * np.pi * x_grid / h_shape[1]) * np.cos(2.0 * np.pi * y_grid / h_shape[0])

    synthetic_depth = (ref_dsm * true_sim_scale + true_sim_offset + noise + spatial_warp).astype(np.float32)

    rdsm = make_rdsm(synthetic_depth)
    save_rdsm(rdsm, os.path.join(output_dir, 'rdsm', 'baseline_rdsm'), save_png=True)

    pred_dsm, est_a, est_b, valid_mask = calibrate_depth_to_dsm(synthetic_depth, ref_dsm, nodata=nodata)

    out_dsm_path = os.path.join(output_dir, 'dsm', 'baseline_predicted_dsm.tif')
    meta.update({'dtype': 'float32', 'count': 1, 'nodata': nodata})
    with rasterio.open(out_dsm_path, 'w', **meta) as dst:
        dst.write(pred_dsm.astype(np.float32), 1)

    metrics = calculate_metrics(pred_dsm, ref_dsm, valid_mask=valid_mask, nodata=nodata)
    error_map = compute_error_map(pred_dsm, ref_dsm, valid_mask=valid_mask, nodata=nodata)

    metrics_summary = {
        'calibration_parameters': {
            'estimated_scale_a': est_a,
            'estimated_offset_b': est_b,
            'theoretical_inverse_scale': 1.0 / true_sim_scale,
            'theoretical_inverse_offset': -true_sim_offset / true_sim_scale
        },
        'evaluation_metrics': metrics
    }
    with open(os.path.join(output_dir, 'baseline_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)

    print('='*60)
    print('DepthWizard M2 - Baseline Calibration & Evaluation Report')
    print('='*60)
    print(f'Depth range:              [{np.min(synthetic_depth):.4f}, {np.max(synthetic_depth):.4f}]')
    print(f'Reference DSM range:      [{np.min(ref_dsm[valid_mask]):.4f} m, {np.max(ref_dsm[valid_mask]):.4f} m]')
    print(f'Estimated scale a:        {est_a:.4f} (Ideal: ~{1.0/true_sim_scale:.4f})')
    print(f'Estimated offset b:       {est_b:.4f} (Ideal: ~{-true_sim_offset/true_sim_scale:.4f})')
    print(f'Predicted DSM range:      [{np.min(pred_dsm[valid_mask]):.4f} m, {np.max(pred_dsm[valid_mask]):.4f} m]')
    print(f'MAE:                      {metrics["mae"]:.4f} m')
    print(f'RMSE:                     {metrics["rmse"]:.4f} m')
    print(f'Correlation:              {metrics["correlation"]:.4f}')
    print(f'R2:                      {metrics["r2"]:.4f}')
    print('='*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)

    axes[0, 0].imshow(rgb_arr)
    axes[0, 0].set_title('1. Optical RGB', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(synthetic_depth, cmap='inferno')
    axes[0, 1].set_title('2. Synthetic Relative Depth D(x, y)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Relative Value', rotation=270, labelpad=12)

    im2 = axes[0, 2].imshow(rdsm, cmap='magma', vmin=0.0, vmax=1.0)
    axes[0, 2].set_title('3. Normalized rDSM [0, 1]', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Relative Elevation', rotation=270, labelpad=12)

    dsm_min = float(np.min(ref_dsm[valid_mask]))
    dsm_max = float(np.max(ref_dsm[valid_mask]))
    im3 = axes[1, 0].imshow(ref_dsm, cmap='terrain', vmin=dsm_min, vmax=dsm_max)
    axes[1, 0].set_title('4. Reference DSM H_ref(x, y)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Elevation (m)', rotation=270, labelpad=12)

    im4 = axes[1, 1].imshow(pred_dsm, cmap='terrain', vmin=dsm_min, vmax=dsm_max)
    axes[1, 1].set_title(f'5. Calibrated Metric DSM (H = {est_a:.2f}D + {est_b:.2f})', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Elevation (m)', rotation=270, labelpad=12)

    err_p99 = float(np.nanpercentile(error_map, 99))
    im5 = axes[1, 2].imshow(error_map, cmap='hot', vmin=0.0, vmax=err_p99)
    axes[1, 2].set_title(f'6. Absolute Error |H_pred - H_ref| (MAE: {metrics["mae"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    cbar5 = fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar5.set_label('Error (m)', rotation=270, labelpad=12)

    plt.tight_layout()
    fig_save_path = os.path.join(output_dir, 'figures', 'baseline_evaluation.png')
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
    print('Diagnostic figure saved to:', fig_save_path)

if __name__ == '__main__':
    run_baseline_pipeline()
