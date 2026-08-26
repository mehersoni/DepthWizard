import os
import sys
import time
import json
import torch
import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth.depth_model import estimate_depth, load_model
from depth.tiled_inference import estimate_depth_tiled
from calibration.rdsm import make_rdsm
from calibration.metric import fit_scale_offset, apply_scale_offset
from evaluation.metrics import calculate_metrics, compute_error_map

def run_tiled_experiment(
    rgb_path='data/potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif',
    dsm_path='data/potsdam/1_DSM/dsm_potsdam_02_10.tif',
    output_dir='outputs'
):
    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dsm'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)

    print('=' * 75)
    print('DepthWizard M2 - Tiled vs Full-Image Inference Experiment')
    print('=' * 75)

    print('[1/7] Loading Potsdam RGB and Reference DSM...')
    with rasterio.open(rgb_path) as src_rgb:
        rgb_arr = src_rgb.read().transpose(1, 2, 0)
        if rgb_arr.shape[2] > 3:
            rgb_arr = rgb_arr[:, :, :3]

    with rasterio.open(dsm_path) as src_dsm:
        ref_dsm = src_dsm.read(1).astype(np.float32)
        dsm_meta = src_dsm.meta.copy()
        nodata_val = src_dsm.nodata if src_dsm.nodata is not None else -9999.0

    H, W = ref_dsm.shape
    valid_mask = np.isfinite(ref_dsm) & (ref_dsm != nodata_val) & (np.abs(ref_dsm) < 1e5)
    r_valid = ref_dsm[valid_mask]
    ref_min, ref_max = float(np.min(r_valid)), float(np.max(r_valid))
    ref_span = ref_max - ref_min

    # Model loading
    model, processor, device = load_model()

    print('[2/7] Running Method A: Full-Image Inference (518x518 ViT + Upsampling)...')
    t0_full = time.time()
    depth_full = estimate_depth(rgb_arr, model=model, processor=processor, device=device)
    time_full = time.time() - t0_full

    print('[3/7] Calibrating Method A (Full-Image)...')
    scale_a_full, offset_b_full, _ = fit_scale_offset(depth_full, ref_dsm, nodata=nodata_val)
    pred_dsm_full = apply_scale_offset(depth_full, scale_a_full, offset_b_full)
    metrics_full = calculate_metrics(pred_dsm_full, ref_dsm, nodata=nodata_val)
    err_full = compute_error_map(pred_dsm_full, ref_dsm, nodata=nodata_val)

    p_valid_full = pred_dsm_full[valid_mask]
    pred_min_full, pred_max_full = float(np.min(p_valid_full)), float(np.max(p_valid_full))
    pred_span_full = pred_max_full - pred_min_full
    compression_full = pred_span_full / ref_span

    print('[4/7] Running Method B: 512x512 Overlapping Tiled Inference...')
    depth_tiled, info_tiled = estimate_depth_tiled(
        rgb_arr,
        tile_size=512,
        overlap=0.25,
        batch_size=8,
        window_type='hann',
        device=device
    )
    time_tiled = info_tiled['elapsed_seconds']

    # Save tiled depth array
    np.save(os.path.join(output_dir, 'depth', 'potsdam_tiled_depth.npy'), depth_tiled)

    print('[5/7] Calibrating Method B (Tiled)...')
    scale_a_tiled, offset_b_tiled, _ = fit_scale_offset(depth_tiled, ref_dsm, nodata=nodata_val)
    pred_dsm_tiled = apply_scale_offset(depth_tiled, scale_a_tiled, offset_b_tiled)
    metrics_tiled = calculate_metrics(pred_dsm_tiled, ref_dsm, nodata=nodata_val)
    err_tiled = compute_error_map(pred_dsm_tiled, ref_dsm, nodata=nodata_val)

    p_valid_tiled = pred_dsm_tiled[valid_mask]
    pred_min_tiled, pred_max_tiled = float(np.min(p_valid_tiled)), float(np.max(p_valid_tiled))
    pred_span_tiled = pred_max_tiled - pred_min_tiled
    compression_tiled = pred_span_tiled / ref_span

    # Save tiled DSM GeoTIFF
    tiled_dsm_path = os.path.join(output_dir, 'dsm', 'potsdam_tiled_predicted.tif')
    with rasterio.open(tiled_dsm_path, 'w', **dsm_meta) as dst:
        dst.write(pred_dsm_tiled, 1)

    print('[6/7] Calculating Comparative Metrics...')
    improvement_mae_pct = float(((metrics_full['mae'] - metrics_tiled['mae']) / metrics_full['mae']) * 100.0)
    improvement_rmse_pct = float(((metrics_full['rmse'] - metrics_tiled['rmse']) / metrics_full['rmse']) * 100.0)
    improvement_r2 = float(metrics_tiled['r2'] - metrics_full['r2'])

    comparison_data = {
        'tile_id': 'potsdam_2_10',
        'method_A_full_image': {
            'input_size': [518, 518],
            'mae_m': metrics_full['mae'],
            'rmse_m': metrics_full['rmse'],
            'pearson_correlation': metrics_full['correlation'],
            'r2': metrics_full['r2'],
            'scale_a': scale_a_full,
            'offset_b': offset_b_full,
            'predicted_min_m': pred_min_full,
            'predicted_max_m': pred_max_full,
            'predicted_span_m': pred_span_full,
            'span_compression_ratio': compression_full,
            'inference_time_seconds': time_full
        },
        'method_B_tiled_inference': {
            'tile_size': 512,
            'overlap': 0.25,
            'total_tiles': info_tiled['total_tiles'],
            'mae_m': metrics_tiled['mae'],
            'rmse_m': metrics_tiled['rmse'],
            'pearson_correlation': metrics_tiled['correlation'],
            'r2': metrics_tiled['r2'],
            'scale_a': scale_a_tiled,
            'offset_b': offset_b_tiled,
            'predicted_min_m': pred_min_tiled,
            'predicted_max_m': pred_max_tiled,
            'predicted_span_m': pred_span_tiled,
            'span_compression_ratio': compression_tiled,
            'inference_time_seconds': time_tiled,
            'peak_gpu_memory_mb': info_tiled['peak_gpu_memory_mb'],
            'device': info_tiled['device']
        },
        'comparison': {
            'improvement_MAE_percent': improvement_mae_pct,
            'improvement_RMSE_percent': improvement_rmse_pct,
            'improvement_R2': improvement_r2,
            'summary': 'Tiled inference preserves local high-frequency structural details but introduces patch-level relative scale shifts across the scene.'
        }
    }

    json_path = os.path.join(output_dir, 'metrics', 'potsdam_tiled_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    print(f'Saved comparison JSON to: {json_path}')

    print('[7/7] Generating 8-Panel Comparison Visualization...')
    step = 6
    rgb_sub = rgb_arr[::step, ::step]
    ref_sub = ref_dsm[::step, ::step]
    depth_full_sub = depth_full[::step, ::step]
    depth_tiled_sub = depth_tiled[::step, ::step]
    pred_full_sub = pred_dsm_full[::step, ::step]
    pred_tiled_sub = pred_dsm_tiled[::step, ::step]
    err_full_sub = err_full[::step, ::step]
    err_tiled_sub = err_tiled[::step, ::step]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=150)

    # Panel 1: RGB
    axes[0, 0].imshow(rgb_sub)
    axes[0, 0].set_title('1. Original RGB (6000x6000)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Full-image depth
    im2 = axes[0, 1].imshow(depth_full_sub, cmap='magma')
    axes[0, 1].set_title('2. Full-Image Relative Depth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Tiled depth
    im3 = axes[0, 2].imshow(depth_tiled_sub, cmap='magma')
    axes[0, 2].set_title('3. Tiled Relative Depth (512x512)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Panel 4: Reference DSM
    im4 = axes[0, 3].imshow(ref_sub, cmap='terrain', vmin=ref_min, vmax=ref_max)
    axes[0, 3].set_title('4. Reference LiDAR DSM', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    cbar4 = fig.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar4.set_label('Metres')

    # Panel 5: Full-image Predicted DSM
    im5 = axes[1, 0].imshow(pred_full_sub, cmap='terrain', vmin=ref_min, vmax=ref_max)
    axes[1, 0].set_title(f'5. Full-Image DSM (MAE: {metrics_full["mae"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    cbar5 = fig.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar5.set_label('Metres')

    # Panel 6: Tiled Predicted DSM
    im6 = axes[1, 1].imshow(pred_tiled_sub, cmap='terrain', vmin=ref_min, vmax=ref_max)
    axes[1, 1].set_title(f'6. Tiled DSM (MAE: {metrics_tiled["mae"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    cbar6 = fig.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar6.set_label('Metres')

    # Panel 7: Full-image Absolute Error
    vmax_err = float(np.nanpercentile(err_full_sub, 95))
    im7 = axes[1, 2].imshow(err_full_sub, cmap='magma', vmin=0, vmax=vmax_err)
    axes[1, 2].set_title(f'7. Full-Image Error (RMSE: {metrics_full["rmse"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    cbar7 = fig.colorbar(im7, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar7.set_label('Abs Error (m)')

    # Panel 8: Tiled Absolute Error
    im8 = axes[1, 3].imshow(err_tiled_sub, cmap='magma', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title(f'8. Tiled Error (RMSE: {metrics_tiled["rmse"]:.2f}m)', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    cbar8 = fig.colorbar(im8, ax=axes[1, 3], fraction=0.046, pad=0.04)
    cbar8.set_label('Abs Error (m)')

    plt.suptitle('DepthWizard M2: Full-Image vs High-Resolution Tiled Inference Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figures', 'potsdam_tiled_comparison.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison figure to: {fig_path}')

    print('\n' + '='*75)
    print('EXPERIMENT RESULTS: FULL-IMAGE VS TILED INFERENCE')
    print('='*75)
    print(f'   Metric              | Method A (Full)   | Method B (Tiled)  | Difference / Change')
    print('   ------------------------------------------------------------------------')
    print(f'   MAE (m)             | {metrics_full["mae"]:16.4f} | {metrics_tiled["mae"]:16.4f} | {improvement_mae_pct:+.2f}% (+ is better)')
    print(f'   RMSE (m)            | {metrics_full["rmse"]:16.4f} | {metrics_tiled["rmse"]:16.4f} | {improvement_rmse_pct:+.2f}% (+ is better)')
    print(f'   Pearson Correlation (r) | {metrics_full["correlation"]:16.4f} | {metrics_tiled["correlation"]:16.4f} | {metrics_tiled["correlation"]-metrics_full["correlation"]:+.4f}')
    print(f'   Coefficient of Det (R2)| {metrics_full["r2"]:16.4f} | {metrics_tiled["r2"]:16.4f} | {improvement_r2:+.4f}')
    print(f'   Predicted Height Span  | {pred_span_full:15.3f}m | {pred_span_tiled:15.3f}m | Ref Span: {ref_span:.3f}m')
    print(f'   Span Compression Ratio | {compression_full:16.4f} | {compression_tiled:16.4f} | {(compression_tiled-compression_full):+.4f}')
    print(f'   Inference Time        | {time_full:15.2f}s | {time_tiled:15.2f}s | {time_tiled/time_full:.1f}x slower')
    print('   ------------------------------------------------------------------------')

    print('\n' + '='*75)
    print('TECHNICAL ANALYSIS & CONCLUSION')
    print('='*75)
    if metrics_tiled['mae'] < metrics_full['mae']:
        print(f'[RESULT] Tiled inference IMPROVES MAE by {improvement_mae_pct:.2f}%!')
    else:
        print(f'[RESULT] Tiled inference yielded MAE = {metrics_tiled["mae"]:.4f}m (vs Full MAE = {metrics_full["mae"]:.4f}m).')
    print('='*75)

if __name__ == '__main__':
    run_tiled_experiment()
