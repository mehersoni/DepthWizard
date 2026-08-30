import csv
import os
import sys
import glob
import cv2 as cv
import numpy as np

sys.path.insert(0, 'c:/DeepthWizard')
from tmp.final_m4_inference import predict_building_height_m4, parse_tfw

out_lines = []

def print_log(msg=""):
    print(msg)
    out_lines.append(msg)

root_dir = 'c:/DeepthWizard'
csv_path = os.path.join(root_dir, 'output', 'potsdam_full_results.csv')

with open(csv_path, 'r') as f:
    rows = list(csv.DictReader(f))

print(f"Total building records in CSV: {len(rows)}")

# Query representative examples
cases_def = [
    ("Small Successful Prediction", "2_10", 4),
    ("Medium Successful Prediction", "2_10", 2),
    ("Large Successful Prediction", "2_10", 1),
    ("FALSE_SHORT_SHADOW Failure", "2_10", 3),
    ("FALSE_LONG_SHADOW Failure", "2_10", 5),
    ("LOW CONFIDENCE Prediction", "3_10", 3),
    ("REJECTED Prediction", "2_10", 68)
]

categories = []
for title, t_id, b_id in cases_def:
    match = [r for r in rows if r['tile_id'] == t_id and int(r['building_id']) == b_id][0]
    categories.append((title, match))

dataset_dir = os.path.join(root_dir, 'Dataset', 'Potsdam')

print("\n" + "="*100)
print("              FINAL INFERENCE DEMONSTRATION ACROSS 7 REPRESENTATIVE CASES")
print("="*100)

for cat_name, row in categories:
    tile_id = row['tile_id']
    bldg_id = int(row['building_id'])
    gt_h = float(row['gt_height_m'])
    
    # Locate Tile Files
    rgb_matches = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_{tile_id}_RGB.tif"), recursive=True)
    if not rgb_matches:
        rgb_matches = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_0{tile_id}_RGB.tif"), recursive=True)
        
    label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_{tile_id}_label_noBoundary.tif"), recursive=True)
    if not label_matches:
        label_matches = glob.glob(os.path.join(dataset_dir, "**", f"*_0{tile_id}_label_noBoundary.tif"), recursive=True)

    if not rgb_matches or not label_matches:
        print(f"Error: Tile files for {tile_id} not found.")
        continue

    rgb_img = cv.imread(rgb_matches[0])
    label_img = cv.imread(label_matches[0])
    
    tfw_path = rgb_matches[0].replace('.tif', '.tfw')
    gsd = parse_tfw(tfw_path) if os.path.exists(tfw_path) else 0.05

    # Extract building contour from label mask
    if len(label_img.shape) == 3:
        building_mask = ((label_img[:, :, 2] > 200) & (label_img[:, :, 1] > 200) & (label_img[:, :, 0] > 200)).astype(np.uint8) * 255
    else:
        building_mask = (label_img == 255).astype(np.uint8) * 255

    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(building_mask, connectivity=8)
    
    if bldg_id >= num_labels:
        print(f"Building #{bldg_id} out of bounds for tile {tile_id}.")
        continue

    single_mask = (labels_im == bldg_id).astype(np.uint8) * 255
    cnts, _ = cv.findContours(single_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        print(f"No contour found for building #{bldg_id}.")
        continue
    cnt = max(cnts, key=cv.contourArea)

    # RUN INFERENCE (STRICTLY NO GT INPUT)
    pred_res = predict_building_height_m4(
        rgb_img=rgb_img,
        building_contour=cnt,
        meters_per_pixel=gsd,
        sun_elevation_deg=41.8
    )

    pred_h = pred_res['predicted_height_m']
    shadow_m = pred_res['shadow_length_m']
    status = pred_res['status']
    conf = pred_res['confidence']
    term = pred_res['termination_reason']
    abs_err = abs(pred_h - gt_h) if pred_h > 0 else 0.0

    print_log(f"\nCategory: {cat_name}")
    print_log(f"  Potsdam Tile: {tile_id} | Building ID: #{bldg_id}")
    print_log(f"  Inputs Supplied: RGB Image ({rgb_img.shape}), Contour ({len(cnt)} pts), GSD ({gsd} m/px), Solar Elev (41.8 deg)")
    print_log(f"  Inference Outputs:")
    print_log(f"    - Base Point: ({pred_res['base_point'][0]:.1f}, {pred_res['base_point'][1]:.1f})")
    print_log(f"    - Tip Point: ({pred_res['tip_point'][0]:.1f}, {pred_res['tip_point'][1]:.1f})")
    print_log(f"    - Predicted Shadow Length: {shadow_m:.2f} m ({pred_res['shadow_length_px']:.1f} px)")
    print_log(f"    - PREDICTED HEIGHT: {pred_h:.2f} m")
    print_log(f"    - Status: {status} | Confidence: {conf:.2f} | Termination: {term}")
    print_log(f"  Post-Hoc Evaluation ONLY: GT Height = {gt_h:.2f} m | Absolute Error = {abs_err:.2f} m")

print_log("="*100)

with open('c:/DeepthWizard/tmp/multicase_demo_output.txt', 'w') as f:
    f.write('\n'.join(out_lines))

