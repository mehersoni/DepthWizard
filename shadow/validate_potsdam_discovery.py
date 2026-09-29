"""
M4 Shadow Cue Module - Potsdam Dataset Discovery & Reference Pairing Script (STEP 1)

Scans demoImages/ dynamically to discover ISPRS Potsdam RGB tiles, extracts tile IDs,
pairs matching DSM, nDSM, TFW georeferencing world files, and ground-truth building label rasters.
Verifies raster dimensions and reports reference data availability using clean relative paths.

Usage:
    python shadow/validate_potsdam_discovery.py
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional
from PIL import Image

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_potsdam_tile_id(filename: str) -> Optional[str]:
    """
    Extracts Potsdam tile ID (e.g. '2_10') from filename patterns.
    Handles 'top_potsdam_2_10_RGB.tif', 'dsm_potsdam_02_10.tif', etc.
    """
    m = re.search(r'potsdam_0?(\d+)_0?(\d+)', filename, re.IGNORECASE)
    if m:
        row, col = int(m.group(1)), int(m.group(2))
        return f"{row}_{col}"
    return None


def discover_potsdam_dataset(
    root_dir: Optional[str] = None,
    demo_dir_name: str = "demoImages",
    dataset_dir_name: str = os.path.join("Dataset", "Potsdam")
) -> List[Dict[str, Any]]:
    """
    Discovers all Potsdam RGB tiles in demoImages/, pairs matching DSM, nDSM, TFW, and building label files,
    checks image dimensions, and verifies relative paths.
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    demo_dir = os.path.join(root_dir, demo_dir_name)
    dataset_dir = os.path.join(root_dir, dataset_dir_name)

    if not os.path.exists(demo_dir):
        raise FileNotFoundError(f"demoImages directory not found at relative path: {os.path.relpath(demo_dir, root_dir)}")

    demo_files = os.listdir(demo_dir)
    
    # Filter for Potsdam RGB TIFF images
    rgb_files = [
        f for f in sorted(demo_files)
        if "potsdam" in f.lower() and "rgb" in f.lower() and f.lower().endswith(".tif")
    ]

    discovered_records = []

    for rgb_file in rgb_files:
        tile_id = parse_potsdam_tile_id(rgb_file)
        if not tile_id:
            continue

        r_str, c_str = tile_id.split("_")
        r_num, c_num = int(r_str), int(c_str)
        row_z, col_z = f"{r_num:02d}", f"{c_num:02d}"

        # RGB Path
        rgb_full = os.path.join(demo_dir, rgb_file)
        rgb_rel = os.path.relpath(rgb_full, root_dir)

        # 1. DSM Path Search (demoImages first, fallback to Dataset/Potsdam/)
        dsm_candidates = [
            os.path.join(demo_dir, f"dsm_potsdam_{row_z}_{col_z}.tif"),
            os.path.join(dataset_dir, "1_DSM", "1_DSM", f"dsm_potsdam_{row_z}_{col_z}.tif")
        ]
        dsm_path = next((p for p in dsm_candidates if os.path.exists(p)), None)
        dsm_rel = os.path.relpath(dsm_path, root_dir) if dsm_path else None

        # 2. nDSM Path Search (lastools & ownapproach)
        ndsm_las_candidates = [
            os.path.join(demo_dir, f"dsm_potsdam_{row_z}_{col_z}_normalized_lastools.jpg"),
            os.path.join(dataset_dir, "1_DSM_normalisation", "1_DSM_normalisation", f"dsm_potsdam_{row_z}_{col_z}_normalized_lastools.jpg")
        ]
        ndsm_las_path = next((p for p in ndsm_las_candidates if os.path.exists(p)), None)
        ndsm_las_rel = os.path.relpath(ndsm_las_path, root_dir) if ndsm_las_path else None

        ndsm_own_candidates = [
            os.path.join(demo_dir, f"dsm_potsdam_{row_z}_{col_z}_normalized_ownapproach.jpg"),
            os.path.join(dataset_dir, "1_DSM_normalisation", "1_DSM_normalisation", f"dsm_potsdam_{row_z}_{col_z}_normalized_ownapproach.jpg")
        ]
        ndsm_own_path = next((p for p in ndsm_own_candidates if os.path.exists(p)), None)
        ndsm_own_rel = os.path.relpath(ndsm_own_path, root_dir) if ndsm_own_path else None

        # 3. TFW Georeferencing World File Search
        tfw_candidates = [
            os.path.join(demo_dir, f"top_potsdam_{r_num}_{c_num}_RGB.tfw"),
            os.path.join(dataset_dir, "2_Ortho_RGB", "2_Ortho_RGB", f"top_potsdam_{r_num}_{c_num}_RGB.tfw")
        ]
        tfw_path = next((p for p in tfw_candidates if os.path.exists(p)), None)
        tfw_rel = os.path.relpath(tfw_path, root_dir) if tfw_path else None

        # 4. Building Ground-Truth Label Search
        label_candidates = [
            os.path.join(demo_dir, f"top_potsdam_{r_num}_{c_num}_label.tif"),
            os.path.join(demo_dir, f"top_potsdam_{r_num}_{c_num}_label_noBoundary.tif"),
            os.path.join(dataset_dir, "5_Labels", "5_Labels", f"top_potsdam_{r_num}_{c_num}_label.tif"),
            os.path.join(dataset_dir, "5_Labels_noBoundary", "5_Labels_noBoundary", f"top_potsdam_{r_num}_{c_num}_label_noBoundary.tif")
        ]
        label_path = next((p for p in label_candidates if os.path.exists(p)), None)
        label_rel = os.path.relpath(label_path, root_dir) if label_path else None

        # Image Dimensions Check
        try:
            with Image.open(rgb_full) as img:
                rgb_dim = img.size  # (W, H)
        except Exception:
            rgb_dim = None

        dsm_dim = None
        if dsm_path:
            try:
                with Image.open(dsm_path) as img:
                    dsm_dim = img.size
            except Exception:
                dsm_dim = None

        label_dim = None
        if label_path:
            try:
                with Image.open(label_path) as img:
                    label_dim = img.size
            except Exception:
                label_dim = None

        is_6000x6000 = bool(rgb_dim == (6000, 6000))

        record = {
            "tile_id": tile_id,
            "rgb_path": rgb_rel,
            "rgb_dim": rgb_dim,
            "is_6000x6000": is_6000x6000,
            "dsm_path": dsm_rel,
            "dsm_dim": dsm_dim,
            "ndsm_lastools_path": ndsm_las_rel,
            "ndsm_ownapproach_path": ndsm_own_rel,
            "tfw_path": tfw_rel,
            "label_path": label_rel,
            "label_dim": label_dim,
            "missing_files": []
        }

        if not dsm_rel:
            record["missing_files"].append("DSM")
        if not ndsm_las_rel and not ndsm_own_rel:
            record["missing_files"].append("nDSM")
        if not tfw_rel:
            record["missing_files"].append("TFW")
        if not label_rel:
            record["missing_files"].append("Label")

        discovered_records.append(record)

    return discovered_records


def run_discovery_report():
    print("=" * 115)
    print(" STEP 1 — ISPRS POTSDAM DATASET DISCOVERY MATRIX ")
    print("=" * 115)

    records = discover_potsdam_dataset()

    print(f"\nDiscovered {len(records)} Potsdam RGB tile(s) in demoImages/:\n")

    # Table Header
    print("-" * 115)
    print(f"{'Tile ID':<8} | {'RGB Path':<32} | {'Dimensions':<12} | {'DSM':<8} | {'nDSM':<8} | {'TFW':<8} | {'Label':<8}")
    print("-" * 115)

    all_6000 = True
    all_relative = True

    for r in records:
        tid = r["tile_id"]
        rgb_p = r["rgb_path"]
        dim_str = f"{r['rgb_dim'][0]}x{r['rgb_dim'][1]}" if r['rgb_dim'] else "Unknown"
        dsm_st = "FOUND" if r["dsm_path"] else "MISSING"
        ndsm_st = "FOUND" if (r["ndsm_lastools_path"] or r["ndsm_ownapproach_path"]) else "MISSING"
        tfw_st = "FOUND" if r["tfw_path"] else "MISSING"
        lbl_st = "FOUND" if r["label_path"] else "MISSING"

        if not r["is_6000x6000"]:
            all_6000 = False

        for p_val in [r["rgb_path"], r["dsm_path"], r["ndsm_lastools_path"], r["tfw_path"], r["label_path"]]:
            if p_val and (os.path.isabs(p_val) or ":" in p_val):
                all_relative = False

        print(f"{tid:<8} | {rgb_p:<32} | {dim_str:<12} | {dsm_st:<8} | {ndsm_st:<8} | {tfw_st:<8} | {lbl_st:<8}")

    print("-" * 115)

    print("\n[PAIRED FILE PATHS DETAILS]")
    for r in records:
        print(f"\nTile ID: {r['tile_id']}")
        print(f"  * RGB Image Path   : {r['rgb_path']} (Dimensions: {r['rgb_dim']})")
        print(f"  * DSM Raster Path  : {r['dsm_path']} (Dimensions: {r['dsm_dim']})")
        print(f"  * nDSM (lastools)  : {r['ndsm_lastools_path']}")
        print(f"  * nDSM (ownappr)   : {r['ndsm_ownapproach_path']}")
        print(f"  * TFW World File   : {r['tfw_path']}")
        print(f"  * Ground-Truth Lbl : {r['label_path']} (Dimensions: {r['label_dim']})")
        if r["missing_files"]:
            print(f"  * Missing Items    : {', '.join(r['missing_files'])}")
        else:
            print("  * Missing Items    : NONE (All reference files present)")

    print("\n" + "=" * 115)
    print(" STEP 1 DISCOVERY VERIFICATION SUMMARY ")
    print("=" * 115)
    print(f"1. Potsdam Tiles Discovered     : {len(records)} ({', '.join(r['tile_id'] for r in records)})")
    print(f"2. Image Resolution Verification: {'PASS (6000x6000)' if all_6000 else 'FAIL'}")
    print(f"3. Path Safety Verification    : {'PASS (Relative paths only)' if all_relative else 'FAIL'}")
    print(f"4. Missing File Safety Handling: PASS (Gracefully reported without crashing)")
    print(f"5. Pipeline Height Estimation  : NOT EXECUTED (Step 1 requirement)")
    print("=" * 115)

    return records


if __name__ == "__main__":
    run_discovery_report()
