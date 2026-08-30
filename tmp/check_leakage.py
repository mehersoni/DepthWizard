import os
import sys

target_files = [
    'shadow/m4_physical_raycast_experiment.py',
    'shadow/geometry.py',
    'shadow/confidence.py',
    'shadow/height.py',
    'shadow/detector.py',
    'shadow/cleaner.py',
    'shadow/scale.py',
    'tmp/final_m4_inference.py'
]

keywords = ['gt_height', 'ground_truth', 'gt_', 'height_labels']

root_dir = 'c:/DeepthWizard'
lines_out = []
lines_out.append("DATA-LEAKAGE AUDIT REPORT")
lines_out.append("="*60)

leakage_found = False

for rel_p in target_files:
    full_p = os.path.join(root_dir, rel_p)
    if not os.path.exists(full_p):
        continue
    with open(full_p, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if s.startswith('#') or s.startswith('"""') or s.startswith('*'):
                continue
            for kw in keywords:
                if kw in s.lower():
                    lines_out.append(f"MATCH: {rel_p}:{idx} -> {s}")
                    leakage_found = True

lines_out.append("="*60)
lines_out.append(f"LEAKAGE FOUND: {leakage_found}")

out_path = os.path.join(root_dir, 'tmp', 'leakage_report.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out))

print(f"Report saved to {out_path}")
