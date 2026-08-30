import os
import sys
import glob
import cv2 as cv
import numpy as np

sys.path.insert(0, 'c:/DeepthWizard')
from tmp.final_m4_inference import predict_building_height_m4, parse_tfw

root_dir = 'c:/DeepthWizard'
dataset_dir = os.path.join(root_dir, 'Dataset', 'Potsdam')

tile_id = '2_10'
rgb_path = glob.glob(os.path.join(dataset_dir, "**", f"top_potsdam_{tile_id}_RGB.tif"), recursive=True)[0]
label_path = glob.glob(os.path.join(dataset_dir, "**", f"*_{tile_id}_label_noBoundary.tif"), recursive=True)[0]

rgb_img = cv.imread(rgb_path)
label_img = cv.imread(label_path)

tfw_path = rgb_path.replace('.tif', '.tfw')
gsd = parse_tfw(tfw_path) if os.path.exists(tfw_path) else 0.05

if len(label_img.shape) == 3:
    building_mask = ((label_img[:, :, 2] > 200) & (label_img[:, :, 1] > 200) & (label_img[:, :, 0] > 200)).astype(np.uint8) * 255
else:
    building_mask = (label_img == 255).astype(np.uint8) * 255

num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(building_mask, connectivity=8)

test_bldgs = [1, 2, 3, 4, 5]
out_lines = []
out_lines.append("M4 REPRODUCIBILITY & DETERMINISM AUDIT REPORT")
out_lines.append("="*70)

all_deterministic = True

for b_id in test_bldgs:
    single_mask = (labels_im == b_id).astype(np.uint8) * 255
    cnts, _ = cv.findContours(single_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv.contourArea)

    runs = []
    for run_idx in range(3):
        res = predict_building_height_m4(
            rgb_img=rgb_img,
            building_contour=cnt,
            meters_per_pixel=gsd,
            sun_elevation_deg=41.8
        )
        runs.append(res)

    h0, h1, h2 = runs[0]['predicted_height_m'], runs[1]['predicted_height_m'], runs[2]['predicted_height_m']
    l0, l1, l2 = runs[0]['shadow_length_m'], runs[1]['shadow_length_m'], runs[2]['shadow_length_m']
    s0, s1, s2 = runs[0]['status'], runs[1]['status'], runs[2]['status']

    is_equal = (h0 == h1 == h2) and (l0 == l1 == l2) and (s0 == s1 == s2)
    if not is_equal:
        all_deterministic = False

    out_lines.append(f"Building #{b_id}:")
    out_lines.append(f"  Run 1 -> Height: {h0:.4f}m, Shadow: {l0:.4f}m, Status: {s0}")
    out_lines.append(f"  Run 2 -> Height: {h1:.4f}m, Shadow: {l1:.4f}m, Status: {s1}")
    out_lines.append(f"  Run 3 -> Height: {h2:.4f}m, Shadow: {l2:.4f}m, Status: {s2}")
    out_lines.append(f"  DETERMINISTIC REPEATABILITY: {'PASS' if is_equal else 'FAIL'}")
    out_lines.append("-" * 70)

out_lines.append(f"OVERALL DETERMINISM AUDIT: {'PASS' if all_deterministic else 'FAIL'}")
out_lines.append("="*70)

out_path = os.path.join(root_dir, 'tmp', 'reproducibility_report.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))

print(f"Report saved to {out_path}")
