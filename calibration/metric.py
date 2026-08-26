import numpy as np
from typing import Optional, Dict, Any, Tuple


def fit_scale_offset(predicted_depth, reference_dsm, nodata=-9999.0):
    if predicted_depth is None or reference_dsm is None:
        raise ValueError('Inputs cannot be None')
    d_arr = np.squeeze(predicted_depth).astype(np.float64)
    h_arr = np.squeeze(reference_dsm).astype(np.float64)
    if d_arr.ndim != 2 or h_arr.ndim != 2:
        raise ValueError('Expected 2D arrays')
    if d_arr.shape != h_arr.shape:
        raise ValueError('Shape mismatch: ' + str(d_arr.shape) + ' vs ' + str(h_arr.shape))
    valid_mask = np.isfinite(d_arr) & np.isfinite(h_arr) & (h_arr != nodata) & (np.abs(h_arr) < 1e5)
    num_valid = int(np.sum(valid_mask))
    if num_valid < 2:
        raise ValueError('Insufficient valid pixels: ' + str(num_valid))
    d_valid = d_arr[valid_mask]
    h_valid = h_arr[valid_mask]
    A = np.column_stack([d_valid, np.ones_like(d_valid)])
    params, residuals, rank, s = np.linalg.lstsq(A, h_valid, rcond=None)
    a = float(params[0])
    b = float(params[1])
    return a, b, valid_mask


def apply_scale_offset(depth, a, b):
    if depth is None:
        raise ValueError('Input depth cannot be None')
    d_arr = np.squeeze(depth).astype(np.float32)
    metric_dsm = a * d_arr + b
    return metric_dsm.astype(np.float32)


def calibrate_depth_to_dsm(depth, reference_dsm, nodata=-9999.0):
    a, b, valid_mask = fit_scale_offset(depth, reference_dsm, nodata=nodata)
    calibrated_dsm = apply_scale_offset(depth, a, b)
    return calibrated_dsm, a, b, valid_mask


def _get_valid_pixels(
    predicted_depth: np.ndarray,
    reference_dsm: np.ndarray,
    nodata: float = -9999.0,
    max_samples: int = 200000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract valid pixel pairs, optionally subsampling for large arrays."""
    d_arr = np.squeeze(predicted_depth).astype(np.float64)
    h_arr = np.squeeze(reference_dsm).astype(np.float64)
    if d_arr.shape != h_arr.shape:
        raise ValueError(f'Shape mismatch: {d_arr.shape} vs {h_arr.shape}')
    valid_mask = np.isfinite(d_arr) & np.isfinite(h_arr) & (h_arr != nodata) & (np.abs(h_arr) < 1e5)
    if np.sum(valid_mask) < 2:
        raise ValueError('Insufficient valid pixels for calibration.')
    d_valid = d_arr[valid_mask]
    h_valid = h_arr[valid_mask]
    if len(d_valid) > max_samples:
        rng = np.random.RandomState(seed)
        idxs = rng.choice(len(d_valid), size=max_samples, replace=False)
        d_valid = d_valid[idxs]
        h_valid = h_valid[idxs]
    return d_valid, h_valid, valid_mask


def fit_poly_calibration(
    predicted_depth: np.ndarray,
    reference_dsm: np.ndarray,
    degree: int = 3,
    nodata: float = -9999.0,
    max_samples: int = 200000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit polynomial calibration H = c0 + c1*D + c2*D^2 + ... + c_deg*D^deg.

    Returns:
        coefficients: numpy array of polynomial coefficients [c_deg, ..., c1, c0] (np.polyfit order)
        info: dict with method, degree, sample count
    """
    if degree < 1 or degree > 5:
        raise ValueError(f'Polynomial degree must be in [1, 5], got {degree}')
    d_valid, h_valid, valid_mask = _get_valid_pixels(predicted_depth, reference_dsm, nodata, max_samples)
    coeffs = np.polyfit(d_valid, h_valid, deg=degree)
    info = {
        'method': f'polynomial_deg{degree}',
        'degree': degree,
        'sample_count': len(d_valid),
        'coefficients': [float(c) for c in coeffs]
    }
    return coeffs, info


def apply_poly_calibration(depth: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply polynomial calibration using coefficients from np.polyfit."""
    d_arr = np.squeeze(depth).astype(np.float64)
    return np.polyval(coeffs, d_arr).astype(np.float32)


def fit_isotonic_calibration(
    predicted_depth: np.ndarray,
    reference_dsm: np.ndarray,
    nodata: float = -9999.0,
    max_samples: int = 200000,
    seed: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit isotonic regression calibration (non-parametric monotonic mapping).

    Returns:
        model: fitted IsotonicRegression instance
        info: dict with method, sample count, x_steps, y_steps for fast inference
    """
    from sklearn.isotonic import IsotonicRegression
    d_valid, h_valid, valid_mask = _get_valid_pixels(predicted_depth, reference_dsm, nodata, max_samples)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(d_valid, h_valid)
    info = {
        'method': 'isotonic',
        'sample_count': len(d_valid),
        'x_steps': iso.f_.x.tolist(),
        'y_steps': iso.f_.y.tolist()
    }
    return iso, info


def apply_isotonic_calibration(depth: np.ndarray, iso_model: Any) -> np.ndarray:
    """Apply isotonic calibration using fitted IsotonicRegression or step arrays."""
    orig_shape = depth.shape
    d_arr = np.squeeze(depth).astype(np.float64).ravel()
    if hasattr(iso_model, 'predict'):
        result = iso_model.predict(d_arr.reshape(-1, 1))
    else:
        x_steps, y_steps = iso_model['x_steps'], iso_model['y_steps']
        result = np.interp(d_arr, x_steps, y_steps)
    return result.reshape(orig_shape).astype(np.float32)


def calibrate_depth_to_dsm_nonlinear(
    depth: np.ndarray,
    reference_dsm: np.ndarray,
    method: str = 'polynomial_deg3',
    nodata: float = -9999.0,
    max_samples: int = 200000
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Calibrate depth to metric DSM using non-linear methods.

    Methods:
        - 'polynomial_deg2': 2nd-degree polynomial
        - 'polynomial_deg3': 3rd-degree polynomial (default)
        - 'polynomial_deg4': 4th-degree polynomial
        - 'isotonic': non-parametric monotonic regression

    Returns:
        calibrated_dsm: float32 metric height map
        method_name: string identifying the method used
        info: calibration metadata dict
    """
    if method == 'isotonic':
        model, info = fit_isotonic_calibration(depth, reference_dsm, nodata=nodata, max_samples=max_samples)
        calibrated = apply_isotonic_calibration(depth, model)
    elif method.startswith('polynomial_deg'):
        degree = int(method.replace('polynomial_deg', ''))
        coeffs, info = fit_poly_calibration(depth, reference_dsm, degree=degree, nodata=nodata, max_samples=max_samples)
        calibrated = apply_poly_calibration(depth, coeffs)
    else:
        raise ValueError(f'Unknown non-linear calibration method: {method}. '
                         f'Use polynomial_deg2, polynomial_deg3, polynomial_deg4, or isotonic.')
    return calibrated, method, info
