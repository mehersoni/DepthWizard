"""
Module: calibration.gcp_calibration
Description: Sparse Ground Control Point (GCP) Elevation Calibration (Ultra-Fast).
Supports both linear OLS/Ridge and non-linear (polynomial, isotonic) calibration.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.linear_model import Ridge


def sample_random_gcps(
    valid_mask: np.ndarray,
    k: int,
    seed: int = 42,
    valid_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample K points randomly across valid pixels."""
    rng = np.random.RandomState(seed)
    if valid_indices is None:
        valid_indices = np.argwhere(valid_mask)
    if len(valid_indices) < k:
        raise ValueError(f"Not enough valid pixels ({len(valid_indices)}) for k={k} GCPs.")
    chosen = rng.choice(len(valid_indices), size=k, replace=False)
    coords = valid_indices[chosen]
    return coords[:, 0], coords[:, 1]


def sample_grid_stratified_gcps(
    valid_mask: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample K points by partitioning the raster into grid cells and sampling
    within distinct cells.
    """
    rng = np.random.RandomState(seed)
    h, w = valid_mask.shape

    grid_rows = int(np.ceil(np.sqrt(k)))
    grid_cols = int(np.ceil(k / grid_rows))

    cell_h = h // grid_rows
    cell_w = w // grid_cols

    cell_indices = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
    rng.shuffle(cell_indices)
    selected_cells = cell_indices[:k]

    rows = []
    cols = []

    for r_idx, c_idx in selected_cells:
        r_start = r_idx * cell_h
        r_end = h if r_idx == grid_rows - 1 else (r_idx + 1) * cell_h
        c_start = c_idx * cell_w
        c_end = w if c_idx == grid_cols - 1 else (c_idx + 1) * cell_w

        # Pick random coordinates in the cell
        rand_r = rng.randint(r_start, r_end)
        rand_c = rng.randint(c_start, c_end)
        rows.append(rand_r)
        cols.append(rand_c)

    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def sample_terrain_structure_gcps(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    seed: int = 42,
    terrain_indices: Optional[np.ndarray] = None,
    structure_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample K points stratified by elevation relief:
    - ~50% from low-depth regions (ground / terrain)
    - ~50% from high-depth regions (elevated buildings / structures)
    """
    rng = np.random.RandomState(seed)
    if terrain_indices is None or structure_indices is None:
        d_valid = depth[valid_mask]
        p30 = np.percentile(d_valid, 30.0)
        p70 = np.percentile(d_valid, 70.0)
        terrain_indices = np.argwhere(valid_mask & (depth <= p30))
        structure_indices = np.argwhere(valid_mask & (depth >= p70))

    k_terrain = max(1, int(round(k * 0.5))) if k > 1 else 1
    k_struct = k - k_terrain

    rows = []
    cols = []

    chosen_t = rng.choice(len(terrain_indices), size=k_terrain, replace=False)
    for idx in chosen_t:
        rows.append(terrain_indices[idx][0])
        cols.append(terrain_indices[idx][1])

    if k_struct > 0:
        chosen_s = rng.choice(len(structure_indices), size=k_struct, replace=False)
        for idx in chosen_s:
            rows.append(structure_indices[idx][0])
            cols.append(structure_indices[idx][1])

    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def fit_gcp_calibration(
    gcp_depth: np.ndarray,
    gcp_ref: np.ndarray,
    scale_prior: float = 6.9373
) -> Tuple[float, float]:
    """
    Fit metric elevation parameters H = a * D + b from K Ground Control Points.
    - If K == 1: Uses scale_prior and solves offset b = H_1 - a * D_1.
    - If K >= 2: Solves least squares H = a * D + b with small ridge regularization.
    """
    k = len(gcp_depth)
    if k == 0:
        raise ValueError("Cannot fit calibration with 0 GCPs.")

    if k == 1:
        a = float(scale_prior)
        b = float(gcp_ref[0] - a * gcp_depth[0])
        return a, b

    var_d = float(np.var(gcp_depth))
    if var_d < 1e-6:
        a = float(scale_prior)
        b = float(np.mean(gcp_ref) - a * np.mean(gcp_depth))
        return a, b

    # OLS fit
    cov_dh = np.cov(gcp_depth, gcp_ref)[0, 1]
    a = float(cov_dh / var_d)
    b = float(np.mean(gcp_ref) - a * np.mean(gcp_depth))

    if a <= 0.0 or a > 50.0:
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(gcp_depth.reshape(-1, 1), gcp_ref)
        a = float(ridge.coef_[0])
        b = float(ridge.intercept_)
        if a <= 0.0:
            a = float(scale_prior)
            b = float(np.mean(gcp_ref) - a * np.mean(gcp_depth))

    return a, b


def apply_gcp_calibration(
    depth: np.ndarray,
    a: float,
    b: float
) -> np.ndarray:
    """Apply metric scaling H = a * D + b."""
    return (a * depth + b).astype(np.float32)


def fit_gcp_calibration_nonlinear(
    gcp_depth: np.ndarray,
    gcp_ref: np.ndarray,
    method: str = 'polynomial_deg3'
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit non-linear calibration from sparse GCPs.

    Methods:
        - 'polynomial_deg2': 2nd-degree polynomial (min 3 GCPs)
        - 'polynomial_deg3': 3rd-degree polynomial (min 4 GCPs)
        - 'polynomial_deg4': 4th-degree polynomial (min 5 GCPs)
        - 'isotonic': monotonic regression (min 5 GCPs)

    Returns:
        model: fitted model (coefficients array for poly, IsotonicRegression for isotonic)
        info: calibration metadata dict
    """
    gcp_depth = np.asarray(gcp_depth, dtype=np.float64).ravel()
    gcp_ref = np.asarray(gcp_ref, dtype=np.float64).ravel()
    k = len(gcp_depth)

    if method == 'isotonic':
        if k < 5:
            raise ValueError(f'Isotonic GCP calibration requires at least 5 GCPs, got {k}.')
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(gcp_depth, gcp_ref)
        info = {
            'method': 'gcp_isotonic',
            'gcp_count': k,
            'x_steps': iso.f_.x.tolist(),
            'y_steps': iso.f_.y.tolist()
        }
        return iso, info

    if method.startswith('polynomial_deg'):
        degree = int(method.replace('polynomial_deg', ''))
        min_gcps = degree + 1
        if k < min_gcps:
            raise ValueError(f'Polynomial deg-{degree} GCP calibration requires at least {min_gcps} GCPs, got {k}.')
        coeffs = np.polyfit(gcp_depth, gcp_ref, deg=degree)
        h_pred = np.polyval(coeffs, gcp_depth)
        residuals = h_pred - gcp_ref
        info = {
            'method': f'gcp_polynomial_deg{degree}',
            'gcp_count': k,
            'degree': degree,
            'coefficients': [float(c) for c in coeffs],
            'gcp_mae': float(np.mean(np.abs(residuals))),
            'gcp_rmse': float(np.sqrt(np.mean(residuals ** 2)))
        }
        return coeffs, info

    raise ValueError(f'Unknown non-linear GCP method: {method}')


def apply_gcp_calibration_nonlinear(
    depth: np.ndarray,
    model: Any,
    method: str = 'polynomial_deg3'
) -> np.ndarray:
    """Apply non-linear GCP calibration to a full depth array."""
    orig_shape = depth.shape
    d_arr = np.squeeze(depth).astype(np.float64).ravel()
    if method == 'isotonic':
        if hasattr(model, 'predict'):
            result = model.predict(d_arr.reshape(-1, 1))
        else:
            result = np.interp(d_arr, model['x_steps'], model['y_steps'])
    elif method.startswith('polynomial_deg'):
        result = np.polyval(model, d_arr)
    else:
        raise ValueError(f'Unknown non-linear GCP method: {method}')
    return result.reshape(orig_shape).astype(np.float32)
