import os

artifacts = [
    ('output/potsdam_full_results.csv', 'Full dataset evaluation CSV (1,760 buildings)'),
    ('output/potsdam_full_progress.json', 'Incremental tile evaluation progress log'),
    ('output/potsdam_full_validation_report.md', 'Full Potsdam dataset validation report'),
    ('output/m5_transition_analysis.md', 'M5 solar azimuth & multi-transition diagnostic report'),
    ('output/m6_multiscale_analysis.md', 'M6 multi-scale ray profile smoothing diagnostic report'),
    ('output/FINAL_M4_VALIDATION_REPORT.md', 'Final M4 algorithm validation report'),
    ('output/FINAL_PROJECT_DEMO_REPORT.md', 'Final demonstration & visual diagnostic report'),
    ('output/FINAL_PROJECT_SUMMARY.md', 'Executive summary of project findings'),
    ('README.md', 'Main project documentation and usage guide'),
    ('output/final_demo_1_small_success.png', 'Visual demonstration: Small building success overlay'),
    ('output/final_demo_2_medium_success.png', 'Visual demonstration: Medium building success overlay'),
    ('output/final_demo_3_large_success.png', 'Visual demonstration: Large building success overlay'),
    ('output/final_demo_4_false_short.png', 'Visual demonstration: FALSE_SHORT failure overlay'),
    ('output/final_demo_5_false_long.png', 'Visual demonstration: FALSE_LONG failure overlay'),
    ('output/final_demo_6_low_confidence.png', 'Visual demonstration: LOW CONFIDENCE prediction overlay')
]

root_dir = 'c:/DeepthWizard'
lines_out = []
lines_out.append("| Artifact | Exists | Readable | Size (KB) | Purpose | Status |")
lines_out.append("| :--- | :---: | :---: | :---: | :--- | :---: |")

all_ok = True
for rel_path, purpose in artifacts:
    full_path = os.path.join(root_dir, rel_path)
    exists = os.path.exists(full_path)
    readable = False
    size_kb = 0.0
    if exists:
        size_kb = os.path.getsize(full_path) / 1024.0
        try:
            with open(full_path, 'rb') as f:
                f.read(100)
            readable = True
        except Exception:
            readable = False
    
    status_str = "PASS" if (exists and readable) else "FAIL"
    if status_str == "FAIL":
        all_ok = False
        
    lines_out.append(f"| `{rel_path}` | {exists} | {readable} | `{size_kb:.1f}` | {purpose} | **{status_str}** |")

out_path = os.path.join(root_dir, 'tmp', 'artifact_inventory.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out))

print(f"Artifact inventory saved to {out_path}")
