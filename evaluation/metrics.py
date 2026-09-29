import numpy as np

def calculate_metrics(predicted_dsm, reference_dsm, valid_mask=None, nodata=-9999.0):
    if predicted_dsm is None or reference_dsm is None:
        raise ValueError('Inputs cannot be None')
    p_arr = np.squeeze(predicted_dsm).astype(np.float64)
    r_arr = np.squeeze(reference_dsm).astype(np.float64)
    if p_arr.ndim != 2 or r_arr.ndim != 2:
        raise ValueError('Expected 2D arrays')
    if p_arr.shape != r_arr.shape:
        raise ValueError('Shape mismatch: ' + str(p_arr.shape) + ' vs ' + str(r_arr.shape))
    mask = np.isfinite(p_arr) & np.isfinite(r_arr) & (r_arr != nodata) & (np.abs(r_arr) < 1e5)
    if valid_mask is not None:
        mask = mask & np.squeeze(valid_mask).astype(bool)
    num_valid = int(np.sum(mask))
    if num_valid < 2:
        raise ValueError('Insufficient valid pixels for metrics calculation: ' + str(num_valid))
    p = p_arr[mask]
    r = r_arr[mask]
    diff = p - r
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    var_p = np.var(p)
    var_r = np.var(r)
    if var_p > 1e-12 and var_r > 1e-12:
        corr = float(np.corrcoef(p, r)[0, 1])
    else:
        corr = 1.0 if np.allclose(p, r) else 0.0
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    ss_res = np.sum(diff ** 2)
    if ss_tot > 1e-12:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 1.0 if np.allclose(p, r) else 0.0
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'r2': r2,
        'valid_pixels': num_valid
    }

def compute_error_map(predicted_dsm, reference_dsm, valid_mask=None, nodata=-9999.0):
    if predicted_dsm is None or reference_dsm is None:
        raise ValueError('Inputs cannot be None')
    p_arr = np.squeeze(predicted_dsm).astype(np.float32)
    r_arr = np.squeeze(reference_dsm).astype(np.float32)
    if p_arr.shape != r_arr.shape:
        raise ValueError('Shape mismatch: ' + str(p_arr.shape) + ' vs ' + str(r_arr.shape))
    mask = np.isfinite(p_arr) & np.isfinite(r_arr) & (r_arr != nodata) & (np.abs(r_arr) < 1e5)
    if valid_mask is not None:
        mask = mask & np.squeeze(valid_mask).astype(bool)
    abs_err = np.abs(p_arr - r_arr)
    abs_err[~mask] = np.nan
    return abs_err
