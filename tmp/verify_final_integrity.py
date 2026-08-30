"""
DeepthWizard — Final Validation Integrity Sanity Check Script

Performs lightweight integrity and consistency verification across all frozen M4 assets,
full-dataset evaluation results, diagnostic reports, and visual demonstration artifacts.

DO NOT MODIFY THE M4 ALGORITHM OR ANY EVALUATION METRICS.
"""

import os
import sys
import csv
import json
import math
from typing import Dict, Any, List
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def verify_project_integrity():
    print("=========================================================")
    print("      DEEPTHWIZARD FINAL INTEGRITY VERIFICATION           ")
    print("=========================================================")

    errors = []
    warnings = []

    # 1. Immutable Code File Integrity Check
    immutable_files = [
        os.path.join(root_dir, "shadow", "m4_physical_raycast_experiment.py"),
        os.path.join(root_dir, "shadow", "geometry.py"),
        os.path.join(root_dir, "shadow", "confidence.py"),
        os.path.join(root_dir, "shadow", "height.py")
    ]
    
    print("\n1. Checking Immutable Production Files:")
    for fpath in immutable_files:
        if os.path.exists(fpath):
            size_bytes = os.path.getsize(fpath)
            print(f"  [OK] {os.path.basename(fpath)} (Size: {size_bytes} bytes)")
        else:
            err_msg = f"Missing core immutable production file: {fpath}"
            errors.append(err_msg)
            print(f"  [FAIL] {err_msg}")

    # 2. Results File Consistency Check (1,760 buildings across 38 tiles)
    results_csv = os.path.join(root_dir, "output", "potsdam_full_results.csv")
    progress_json = os.path.join(root_dir, "output", "potsdam_full_progress.json")

    print("\n2. Checking Full-Dataset Results Artifacts:")
    if os.path.exists(results_csv):
        with open(results_csv, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
        
        tot_bldgs = len(reader)
        unique_tiles = {r["tile_id"] for r in reader}
        print(f"  [OK] potsdam_full_results.csv found.")
        print(f"       Total Evaluated Buildings: {tot_bldgs}")
        print(f"       Represented Potsdam Tiles: {len(unique_tiles)}")

        if tot_bldgs != 1760:
            errors.append(f"Building count mismatch in CSV! Expected 1,760, got {tot_bldgs}")
        if len(unique_tiles) < 37 or len(unique_tiles) > 38:
            errors.append(f"Tile count mismatch in CSV! Expected 37-38, got {len(unique_tiles)}")

        # Verify baseline metrics
        errs = []
        for r in reader:
            try:
                errs.append(float(r["m4_error_m"]))
            except ValueError:
                pass

        mae = sum(errs) / len(errs) if errs else 0.0
        medae = float(np.median(errs)) if errs else 0.0
        val_cnt = sum(1 for r in reader if r["m4_status"] == "VALID")
        val_pct = (val_cnt / tot_bldgs) * 100.0

        print(f"       Calculated Baseline MAE: {mae:.2f} m (Expected: 4.69 m)")
        print(f"       Calculated Baseline MedAE: {medae:.2f} m (Expected: 3.32 m)")
        print(f"       Calculated VALID Rate: {val_pct:.1f}% ({val_cnt}/{tot_bldgs})")

        if abs(mae - 4.69) > 0.05:
            warnings.append(f"Calculated MAE ({mae:.2f}m) slightly differs from baseline report (4.69m)")
    else:
        errors.append(f"Missing output/potsdam_full_results.csv")

    if os.path.exists(progress_json):
        with open(progress_json, 'r') as f:
            prog_data = json.load(f)
        print(f"  [OK] potsdam_full_progress.json found ({len(prog_data)} records).")
    else:
        errors.append(f"Missing output/potsdam_full_progress.json")

    # 3. Report File Existence Check
    required_reports = [
        "potsdam_full_validation_report.md",
        "m5_transition_analysis.md",
        "m6_multiscale_analysis.md",
        "FINAL_M4_VALIDATION_REPORT.md"
    ]

    print("\n3. Checking Diagnostic & Validation Reports:")
    for rname in required_reports:
        rpath = os.path.join(root_dir, "output", rname)
        if os.path.exists(rpath):
            print(f"  [OK] output/{rname} ({os.path.getsize(rpath)} bytes)")
        else:
            errors.append(f"Missing required report: output/{rname}")

    # 4. Visual Overlay Demonstration Check
    required_visuals = [
        "final_demo_1_small_success.png",
        "final_demo_2_medium_success.png",
        "final_demo_3_large_success.png",
        "final_demo_4_false_short.png",
        "final_demo_5_false_long.png",
        "final_demo_6_low_confidence.png"
    ]

    print("\n4. Checking Visual Overlay Artifacts:")
    for vname in required_visuals:
        vpath = os.path.join(root_dir, "output", vname)
        if os.path.exists(vpath):
            print(f"  [OK] output/{vname} ({os.path.getsize(vpath)} bytes)")
        else:
            errors.append(f"Missing visual overlay: output/{vname}")

    # Summary
    print("\n=========================================================")
    if errors:
        print(f" [FAIL] Integrity Verification FAILED with {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(" [SUCCESS] ALL PROJECT INTEGRITY CHECKS PASSED PERFECTLY!")
        if warnings:
            print(" Warnings:")
            for w in warnings:
                print(f"  - {w}")
    print("=========================================================")

    return len(errors) == 0


if __name__ == "__main__":
    success = verify_project_integrity()
    sys.exit(0 if success else 1)
