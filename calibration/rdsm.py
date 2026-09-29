#!/usr/bin/env python3
import os, numpy as np, matplotlib.pyplot as plt


def make_rdsm(depth, p_low=2.0, p_high=98.0, eps=1e-7):
    if depth is None or not isinstance(depth, np.ndarray):
        raise ValueError('Input depth must be a NumPy array.')
    if depth.size == 0:
        raise ValueError('Input depth array cannot be empty.')
    arr = np.squeeze(depth).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D depth array, got shape {arr.shape}')
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[valid_mask]
    d_min = float(np.percentile(values, p_low))
    d_max = float(np.percentile(values, p_high))
    if abs(d_max - d_min) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    rdsm = (arr - d_min) / (d_max - d_min + eps)
    rdsm = np.clip(rdsm, 0.0, 1.0)
    rdsm[~valid_mask] = 0.0
    return rdsm.astype(np.float32)


def save_rdsm(rdsm, output_path, save_png=True, colormap='magma', dpi=150):
    if rdsm is None or not isinstance(rdsm, np.ndarray):
        raise ValueError('rdsm must be a NumPy array.')
    base_path, _ = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    npy_path = f'{base_path}.npy'
    np.save(npy_path, rdsm.astype(np.float32))
    results = {'npy': npy_path}
    if save_png:
        png_path = f'{base_path}.png'
        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
        im = ax.imshow(rdsm, cmap=colormap, vmin=0.0, vmax=1.0)
        ax.set_title('Relative Digital Surface Model [rDSM = 0..1]', fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Elevation (Unitless [0, 1])', rotation=270, labelpad=15, fontsize=10)
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches='tight')
        plt.close()
        results['png'] = png_path
    return results
