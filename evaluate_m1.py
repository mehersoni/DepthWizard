"""
M1 Perception Module - Scientific Benchmark Harness
Evaluates model accuracy against a Ground Truth DSM using Held-Out Validation.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import rasterio
import cv2

from depth.preprocess import load_and_validate_image
from depth.models import DepthEstimator

def evaluate_model(image_path: str, gt_dsm_path: str, model_size: str, num_calib_points: int = 100):
    # 1. Load Data
    result = load_and_validate_image(image_path)
    if result is None:
        return None
    image, metadata = result
    
    # Load actual Ground Truth DSM using Rasterio
    try:
        with rasterio.open(gt_dsm_path) as src:
            gt_dsm = src.read(1).astype(np.float32)
    except Exception as e:
        print(f"Error loading GT DSM: {e}")
        return None
        
    # Ensure dimensions match (resize GT if necessary)
    w, h = image.size
    if gt_dsm.shape != (h, w):
         gt_dsm = cv2.resize(gt_dsm, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. Run Inference
    estimator = DepthEstimator()
    estimator.initialize(model_size=model_size)
    
    start_time = time.time()
    # Assuming your predict_with_confidence returns (depth_array, confidence_array)
    inference_result = estimator.predict_with_confidence(image) 
    latency = time.time() - start_time
    
    if inference_result is None:
        return None
    
    pred_depth, _ = inference_result
    
    # 3. Flatten arrays and remove invalid GT pixels (e.g., NoData values)
    valid_mask = np.isfinite(gt_dsm) & (gt_dsm > -1000)
    
    if np.sum(valid_mask) < num_calib_points * 2:
        print("Not enough valid pixels in GT DSM for evaluation.")
        return None
        
    pred_flat = pred_depth[valid_mask].reshape(-1, 1)
    gt_flat = gt_dsm[valid_mask].reshape(-1, 1)
    
    # 4. Held-Out Calibration (Train/Test Split)
    indices = np.arange(len(pred_flat))
    np.random.shuffle(indices)
    
    calib_idx = indices[:num_calib_points]
    eval_idx = indices[num_calib_points:]
    
    # 5. Fit RANSAC Calibration ONLY on the calibration points
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(pred_flat[calib_idx], gt_flat[calib_idx])
    
    # 6. Apply Calibration to the held-out evaluation points
    pred_calibrated_eval = ransac.predict(pred_flat[eval_idx])
    gt_eval = gt_flat[eval_idx]
    
    # 7. Calculate Metrics
    pearson_corr, _ = pearsonr(pred_flat.flatten(), gt_flat.flatten())
    spearman_corr, _ = spearmanr(pred_flat.flatten(), gt_flat.flatten())
    
    mae = mean_absolute_error(gt_eval, pred_calibrated_eval)
    rmse = np.sqrt(mean_squared_error(gt_eval, pred_calibrated_eval))
    
    return {
        "Model Tier": model_size,
        "Latency (s)": round(latency, 3),
        "Pearson R": round(pearson_corr, 4),
        "Spearman \u03c1": round(spearman_corr, 4),
        "Held-Out MAE (m)": round(mae, 3),
        "Held-Out RMSE (m)": round(rmse, 3)
    }

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Update these paths to match your new LiDAR GeoTIFFs
    test_image = os.path.join(root_dir, "demoImages", "rgb.tif")
    gt_dsm = os.path.join(root_dir, "demoImages", "dsm.tif") 
    
    print("=" * 100)
    print(" M1 PERCEPTION - STRUCTURAL & METRIC BENCHMARK ")
    print("=" * 100)
    
    results = []
    for size in ["Small", "Base", "Large"]:
        res = evaluate_model(test_image, gt_dsm, model_size=size)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        print("=" * 100)