import os
import sys
import time
import json
import torch
import rasterio
import numpy as np
from PIL import Image
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from depth.depth_model import estimate_depth, load_model
from evaluation.metrics import calculate_metrics, compute_error_map

def load_tile_data(tile_id, data_dir='data/potsdam'):
    parts = tile_id.split('_')
    dsm_str = f'{int(parts[0]):02d}_{int(parts[1]):02d}'
    rgb_path = os.path.join(data_dir, '2_Ortho_RGB', f'top_potsdam_{tile_id}_RGB.tif')
    dsm_path = os.path.join(data_dir, '1_DSM', f'dsm_potsdam_{dsm_str}.tif')

    with rasterio.open(rgb_path) as src_rgb:
        rgb = src_rgb.read().transpose(1, 2, 0)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

    with rasterio.open(dsm_path) as src_dsm:
        dsm = src_dsm.read(1).astype(np.float32)
        nodata = src_dsm.nodata if src_dsm.nodata is not None else -9999.0
        meta = src_dsm.meta.copy()

    valid_mask = np.isfinite(dsm) & (dsm != nodata) & (np.abs(dsm) < 1e5)
    return rgb, dsm, valid_mask, nodata, meta

def fit_linear_multi(depth_list, dsm_list, mask_list):
    """
    Exact OLS least squares H = a*D + b across multiple rasters using normal equations.
    """
    sum_d = 0.0
    sum_h = 0.0
    sum_dd = 0.0
    sum_dh = 0.0
    n_total = 0

    for d, h, m in zip(depth_list, dsm_list, mask_list):
        d_v = d[m].astype(np.float64)
        h_v = h[m].astype(np.float64)
        sum_d += np.sum(d_v)
        sum_h += np.sum(h_v)
        sum_dd += np.sum(d_v * d_v)
        sum_dh += np.sum(d_v * h_v)
        n_total += len(d_v)

    mean_d = sum_d / n_total
    mean_h = sum_h / n_total
    cov_dh = (sum_dh / n_total) - (mean_d * mean_h)
    var_d = (sum_dd / n_total) - (mean_d * mean_d)

    a = cov_dh / var_d if var_d > 0 else 0.0
    b = mean_h - a * mean_d
    return float(a), float(b)

def fit_isotonic_multi(depth_list, dsm_list, mask_list, n_samples=100000, seed=42):
    """
    Fit isotonic regression on a representative random subsample across calibration rasters.
    """
    np.random.seed(seed)
    samples_per_tile = n_samples // len(depth_list)
    d_sampled = []
    h_sampled = []

    for d, h, m in zip(depth_list, dsm_list, mask_list):
        d_v = d[m].astype(np.float64)
        h_v = h[m].astype(np.float64)
        k = min(samples_per_tile, len(d_v))
        idxs = np.random.choice(len(d_v), size=k, replace=False)
        d_sampled.append(d_v[idxs])
        h_sampled.append(h_v[idxs])

    d_cat = np.concatenate(d_sampled)
    h_cat = np.concatenate(h_sampled)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(d_cat, h_cat)
    return iso

def predict_isotonic_fast(iso_model, depth_array):
    """
    Ultra-fast vectorized evaluation of isotonic step-function via np.interp.
    """
    x_steps = iso_model.f_.x
    y_steps = iso_model.f_.y
    return np.interp(depth_array, x_steps, y_steps).astype(np.float32)

def evaluate_predictions(pred_dsm, ref_dsm, valid_mask, nodata=-9999.0):
    metrics = calculate_metrics(pred_dsm, ref_dsm, valid_mask=valid_mask, nodata=nodata)
    p_val = pred_dsm[valid_mask]
    r_val = ref_dsm[valid_mask]
    ref_min, ref_max = float(np.min(r_val)), float(np.max(r_val))
    pred_min, pred_max = float(np.min(p_val)), float(np.max(p_val))
    ref_span = ref_max - ref_min
    pred_span = pred_max - pred_min
    compression = pred_span / ref_span if ref_span > 0 else 0.0
    return {
        'mae_m': metrics['mae'],
        'rmse_m': metrics['rmse'],
        'pearson_correlation': metrics['correlation'],
        'r2': metrics['r2'],
        'reference_min_m': ref_min,
        'reference_max_m': ref_max,
        'reference_span_m': ref_span,
        'predicted_min_m': pred_min,
        'predicted_max_m': pred_max,
        'predicted_span_m': pred_span,
        'span_compression_ratio': compression
    }

def run_heldout_validation(output_dir='outputs'):
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print('=' * 75, flush=True)
    print('DepthWizard M2 - Held-Out Spatial Validation Experiment', flush=True)
    print('=' * 75, flush=True)

    tile_ids = ['2_10', '2_11', '2_12', '2_13']
    tile_data = {}

    model, processor, device = load_model()

    print('\n[Step 1/4] Loading Tiles and Computing Relative Depth Maps...', flush=True)
    for t_id in tile_ids:
        print(f'  Processing Potsdam Tile {t_id}...', flush=True)
        rgb, dsm, valid_mask, nodata, meta = load_tile_data(t_id)
        t0 = time.time()
        depth = estimate_depth(rgb, model=model, processor=processor, device=device)
        dt = time.time() - t0
        tile_data[t_id] = {
            'rgb': rgb,
            'dsm': dsm,
            'depth': depth,
            'valid_mask': valid_mask,
            'nodata': nodata,
            'meta': meta,
            'duration': dt
        }
        r_val = dsm[valid_mask]
        print(f'     Depth computed in {dt:.2f}s | Ref DSM Range: [{np.min(r_val):.2f}, {np.max(r_val):.2f}] m | Span: {np.max(r_val)-np.min(r_val):.2f} m', flush=True)

    # Define 4 distinct cross-validation splits
    splits = [
        {
            'split_id': 'Split_1',
            'calib_tiles': ['2_10', '2_11'],
            'heldout_tile': '2_12'
        },
        {
            'split_id': 'Split_2',
            'calib_tiles': ['2_11', '2_12'],
            'heldout_tile': '2_10'
        },
        {
            'split_id': 'Split_3',
            'calib_tiles': ['2_10', '2_12'],
            'heldout_tile': '2_11'
        },
        {
            'split_id': 'Split_4',
            'calib_tiles': ['2_10', '2_11', '2_12'],
            'heldout_tile': '2_13'
        }
    ]

    print('\n[Step 2/4] Executing Held-Out Spatial Validation Splits...', flush=True)
    results = []
    split_visual_data = {}

    for s in splits:
        sid = s['split_id']
        c_tiles = s['calib_tiles']
        h_tile = s['heldout_tile']
        print(f'\n--- Running {sid}: Calibration={c_tiles} -> Held-Out={h_tile} ---', flush=True)

        # 1. Fit Linear & Isotonic on Calibration Tiles ONLY
        c_depths = [tile_data[t]['depth'] for t in c_tiles]
        c_dsms = [tile_data[t]['dsm'] for t in c_tiles]
        c_masks = [tile_data[t]['valid_mask'] for t in c_tiles]

        a_lin, b_lin = fit_linear_multi(c_depths, c_dsms, c_masks)
        iso_model = fit_isotonic_multi(c_depths, c_dsms, c_masks)

        print(f'   Learned Calibration Parameters: a={a_lin:.4f}, b={b_lin:.4f}m', flush=True)

        # 2. In-Sample Evaluation (on Calibration Tiles)
        in_sample_lin_evals = []
        in_sample_iso_evals = []
        for t in c_tiles:
            d = tile_data[t]['depth']
            ref = tile_data[t]['dsm']
            mask = tile_data[t]['valid_mask']
            nd = tile_data[t]['nodata']

            p_lin_in = (a_lin * d + b_lin).astype(np.float32)
            p_iso_in = predict_isotonic_fast(iso_model, d)

            in_sample_lin_evals.append(evaluate_predictions(p_lin_in, ref, mask, nd))
            in_sample_iso_evals.append(evaluate_predictions(p_iso_in, ref, mask, nd))

        # 3. Held-Out Evaluation (on Held-Out Tile WITHOUT using its DSM for fitting)
        h_depth = tile_data[h_tile]['depth']
        h_ref = tile_data[h_tile]['dsm']
        h_mask = tile_data[h_tile]['valid_mask']
        h_nd = tile_data[h_tile]['nodata']

        p_lin_heldout = (a_lin * h_depth + b_lin).astype(np.float32)
        p_iso_heldout = predict_isotonic_fast(iso_model, h_depth)

        heldout_lin_eval = evaluate_predictions(p_lin_heldout, h_ref, h_mask, h_nd)
        heldout_iso_eval = evaluate_predictions(p_iso_heldout, h_ref, h_mask, h_nd)

        # In-sample averages
        avg_in_lin_mae = float(np.mean([e['mae_m'] for e in in_sample_lin_evals]))
        avg_in_lin_rmse = float(np.mean([e['rmse_m'] for e in in_sample_lin_evals]))
        avg_in_lin_r = float(np.mean([e['pearson_correlation'] for e in in_sample_lin_evals]))
        avg_in_lin_r2 = float(np.mean([e['r2'] for e in in_sample_lin_evals]))

        avg_in_iso_mae = float(np.mean([e['mae_m'] for e in in_sample_iso_evals]))
        avg_in_iso_rmse = float(np.mean([e['rmse_m'] for e in in_sample_iso_evals]))
        avg_in_iso_r = float(np.mean([e['pearson_correlation'] for e in in_sample_iso_evals]))
        avg_in_iso_r2 = float(np.mean([e['r2'] for e in in_sample_iso_evals]))

        # Calculate performance degradation relative to in-sample
        degradation_lin_mae_pct = ((heldout_lin_eval['mae_m'] - avg_in_lin_mae) / avg_in_lin_mae) * 100.0
        degradation_lin_rmse_pct = ((heldout_lin_eval['rmse_m'] - avg_in_lin_rmse) / avg_in_lin_rmse) * 100.0
        degradation_iso_mae_pct = ((heldout_iso_eval['mae_m'] - avg_in_iso_mae) / avg_in_iso_mae) * 100.0
        degradation_iso_rmse_pct = ((heldout_iso_eval['rmse_m'] - avg_in_iso_rmse) / avg_in_iso_rmse) * 100.0

        split_record = {
            'split_id': sid,
            'calibration_tiles': c_tiles,
            'heldout_tile': h_tile,
            'learned_parameters': {
                'linear_a': a_lin,
                'linear_b': b_lin
            },
            'linear_calibration': {
                'in_sample': {
                    'classification': 'IN-SAMPLE',
                    'mae_m': avg_in_lin_mae,
                    'rmse_m': avg_in_lin_rmse,
                    'pearson_correlation': avg_in_lin_r,
                    'r2': avg_in_lin_r2
                },
                'held_out': {
                    'classification': 'HELD-OUT',
                    **heldout_lin_eval,
                    'degradation_mae_percent': degradation_lin_mae_pct,
                    'degradation_rmse_percent': degradation_lin_rmse_pct
                }
            },
            'isotonic_calibration': {
                'in_sample': {
                    'classification': 'IN-SAMPLE',
                    'mae_m': avg_in_iso_mae,
                    'rmse_m': avg_in_iso_rmse,
                    'pearson_correlation': avg_in_iso_r,
                    'r2': avg_in_iso_r2
                },
                'held_out': {
                    'classification': 'HELD-OUT',
                    **heldout_iso_eval,
                    'degradation_mae_percent': degradation_iso_mae_pct,
                    'degradation_rmse_percent': degradation_iso_rmse_pct
                }
            }
        }
        results.append(split_record)

        if sid == 'Split_1':
            split_visual_data = {
                'rgb': tile_data[h_tile]['rgb'],
                'ref_dsm': h_ref,
                'pred_lin': p_lin_heldout,
                'pred_iso': p_iso_heldout,
                'err_lin': compute_error_map(p_lin_heldout, h_ref, valid_mask=h_mask),
                'err_iso': compute_error_map(p_iso_heldout, h_ref, valid_mask=h_mask),
                'heldout_tile': h_tile,
                'calib_tiles': c_tiles
            }

        print(f'   Linear In-Sample  : MAE={avg_in_lin_mae:.4f}m, RMSE={avg_in_lin_rmse:.4f}m, r={avg_in_lin_r:.4f}, R2={avg_in_lin_r2:.4f}', flush=True)
        print(f'   Linear HELD-OUT   : MAE={heldout_lin_eval["mae_m"]:.4f}m, RMSE={heldout_lin_eval["rmse_m"]:.4f}m, r={heldout_lin_eval["pearson_correlation"]:.4f}, R2={heldout_lin_eval["r2"]:.4f} (Delta MAE: {degradation_lin_mae_pct:+.2f}%)', flush=True)
        print(f'   Isotonic In-Sample: MAE={avg_in_iso_mae:.4f}m, RMSE={avg_in_iso_rmse:.4f}m, r={avg_in_iso_r:.4f}, R2={avg_in_iso_r2:.4f}', flush=True)
        print(f'   Isotonic HELD-OUT : MAE={heldout_iso_eval["mae_m"]:.4f}m, RMSE={heldout_iso_eval["rmse_m"]:.4f}m, r={heldout_iso_eval["pearson_correlation"]:.4f}, R2={heldout_iso_eval["r2"]:.4f} (Delta MAE: {degradation_iso_mae_pct:+.2f}%)', flush=True)

    mean_heldout_lin_mae = float(np.mean([r['linear_calibration']['held_out']['mae_m'] for r in results]))
    mean_heldout_lin_rmse = float(np.mean([r['linear_calibration']['held_out']['rmse_m'] for r in results]))
    mean_heldout_lin_r = float(np.mean([r['linear_calibration']['held_out']['pearson_correlation'] for r in results]))
    mean_heldout_lin_r2 = float(np.mean([r['linear_calibration']['held_out']['r2'] for r in results]))

    mean_heldout_iso_mae = float(np.mean([r['isotonic_calibration']['held_out']['mae_m'] for r in results]))
    mean_heldout_iso_rmse = float(np.mean([r['isotonic_calibration']['held_out']['rmse_m'] for r in results]))
    mean_heldout_iso_r = float(np.mean([r['isotonic_calibration']['held_out']['pearson_correlation'] for r in results]))
    mean_heldout_iso_r2 = float(np.mean([r['isotonic_calibration']['held_out']['r2'] for r in results]))

    output_json_data = {
        'experiment': 'Held-Out Spatial Validation on ISPRS Potsdam',
        'model': 'Depth Anything V2 (ViT-Small)',
        'splits': results,
        'cross_validation_summary': {
            'linear_calibration': {
                'mean_heldout_mae_m': mean_heldout_lin_mae,
                'mean_heldout_rmse_m': mean_heldout_lin_rmse,
                'mean_heldout_pearson_r': mean_heldout_lin_r,
                'mean_heldout_r2': mean_heldout_lin_r2
            },
            'isotonic_calibration': {
                'mean_heldout_mae_m': mean_heldout_iso_mae,
                'mean_heldout_rmse_m': mean_heldout_iso_rmse,
                'mean_heldout_pearson_r': mean_heldout_iso_r,
                'mean_heldout_r2': mean_heldout_iso_r2
            }
        }
    }

    json_path = os.path.join(output_dir, 'metrics', 'heldout_validation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, indent=2)
    print(f'\n[Step 3/4] Saved held-out validation JSON to: {json_path}', flush=True)

    print('\n[Step 4/4] Generating Held-Out Spatial Validation Figure...', flush=True)
    fig = plt.figure(figsize=(22, 14), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    sub = 6
    rgb_s = split_visual_data['rgb'][::sub, ::sub]
    ref_s = split_visual_data['ref_dsm'][::sub, ::sub]
    pred_lin_s = split_visual_data['pred_lin'][::sub, ::sub]
    pred_iso_s = split_visual_data['pred_iso'][::sub, ::sub]
    err_lin_s = split_visual_data['err_lin'][::sub, ::sub]
    err_iso_s = split_visual_data['err_iso'][::sub, ::sub]

    vmin_h = float(np.min(ref_s))
    vmax_h = float(np.max(ref_s))
    vmax_err = float(np.nanpercentile(err_lin_s, 95))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_s)
    ax1.set_title(f'1. Held-Out RGB (Tile {split_visual_data["heldout_tile"]})', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ref_s, cmap='terrain', vmin=vmin_h, vmax=vmax_h)
    ax2.set_title('2. Ground Truth LiDAR Reference DSM', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('Metres')

    s1_lin = results[0]['linear_calibration']['held_out']
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(pred_lin_s, cmap='terrain', vmin=vmin_h, vmax=vmax_h)
    ax3.set_title(f'3. Held-Out Linear DSM (MAE: {s1_lin["mae_m"]:.2f}m, R2: {s1_lin["r2"]:.2f})', fontsize=12, fontweight='bold')
    ax3.axis('off')
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label('Metres')

    s1_iso = results[0]['isotonic_calibration']['held_out']
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(pred_iso_s, cmap='terrain', vmin=vmin_h, vmax=vmax_h)
    ax4.set_title(f'4. Held-Out Isotonic DSM (MAE: {s1_iso["mae_m"]:.2f}m, R2: {s1_iso["r2"]:.2f})', fontsize=12, fontweight='bold')
    ax4.axis('off')
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label('Metres')

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(err_lin_s, cmap='magma', vmin=0, vmax=vmax_err)
    ax5.set_title(f'5. Linear Absolute Error Map (RMSE: {s1_lin["rmse_m"]:.2f}m)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    cb5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.set_label('Error (m)')

    ax6 = fig.add_subplot(gs[1, 2])
    split_labels = [r['split_id'] for r in results]
    x = np.arange(len(split_labels))
    width = 0.35

    mae_in_lin = [r['linear_calibration']['in_sample']['mae_m'] for r in results]
    mae_out_lin = [r['linear_calibration']['held_out']['mae_m'] for r in results]

    ax6.bar(x - width/2, mae_in_lin, width, label='In-Sample (Train)', color='#4C72B0', alpha=0.85)
    ax6.bar(x + width/2, mae_out_lin, width, label='Held-Out (Test)', color='#DD8452', alpha=0.85)
    ax6.set_ylabel('MAE (metres)', fontsize=11, fontweight='bold')
    ax6.set_title('6. Generalization Gap: In-Sample vs Held-Out MAE', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(split_labels, fontsize=10, fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.suptitle('DepthWizard M2: Spatial Held-Out Cross-Validation on ISPRS Potsdam', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figures', 'heldout_validation.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved held-out validation figure to: {fig_path}', flush=True)

    print('\n' + '=' * 105, flush=True)
    print('HELD-OUT SPATIAL VALIDATION RESULTS TABLE', flush=True)
    print('=' * 105, flush=True)
    print(f"{'Method':<12} | {'Type':<10} | {'Calibration Scenes':<22} | {'Held-Out Scene':<14} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'Pearson r':<10} | {'R2':<8}", flush=True)
    print('-' * 105, flush=True)

    for r in results:
        calib_str = ', '.join(r['calibration_tiles'])
        held_str = r['heldout_tile']

        lin_in = r['linear_calibration']['in_sample']
        print(f"{'Linear':<12} | {'IN-SAMPLE':<10} | {calib_str:<22} | {'[Self/Train]':<14} | {lin_in['mae_m']:<9.4f} | {lin_in['rmse_m']:<9.4f} | {lin_in['pearson_correlation']:<10.4f} | {lin_in['r2']:<8.4f}", flush=True)

        lin_out = r['linear_calibration']['held_out']
        print(f"{'Linear':<12} | {'HELD-OUT':<10} | {calib_str:<22} | {held_str:<14} | {lin_out['mae_m']:<9.4f} | {lin_out['rmse_m']:<9.4f} | {lin_out['pearson_correlation']:<10.4f} | {lin_out['r2']:<8.4f}", flush=True)

        iso_in = r['isotonic_calibration']['in_sample']
        print(f"{'Isotonic':<12} | {'IN-SAMPLE':<10} | {calib_str:<22} | {'[Self/Train]':<14} | {iso_in['mae_m']:<9.4f} | {iso_in['rmse_m']:<9.4f} | {iso_in['pearson_correlation']:<10.4f} | {iso_in['r2']:<8.4f}", flush=True)

        iso_out = r['isotonic_calibration']['held_out']
        print(f"{'Isotonic':<12} | {'HELD-OUT':<10} | {calib_str:<22} | {held_str:<14} | {iso_out['mae_m']:<9.4f} | {iso_out['rmse_m']:<9.4f} | {iso_out['pearson_correlation']:<10.4f} | {iso_out['r2']:<8.4f}", flush=True)
        print('-' * 105, flush=True)

    print('=' * 105, flush=True)

if __name__ == '__main__':
    run_heldout_validation()
