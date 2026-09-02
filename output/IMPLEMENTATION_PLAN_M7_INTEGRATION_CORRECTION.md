# Implementation Plan — M7 Guided Filter Integration Pipeline Correction

## Objective

Correct the end-to-end integration pipeline in `shadow/run_m7_potsdam_validation.py` so that **Guided Filter depth refinement** ($D_{\text{filtered}}$) directly drives inference-time building contour extraction prior to downstream **immutable M4 physical raycasting** (`measure_building_shadow_m4_physical` in `shadow/m4_physical_raycast_experiment.py`).

All 4 frozen production M4 files (`shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`) remain **100% byte-for-byte untouched**.

---

## Data Flow Architecture

### Before (Unlinked Audit Finding):
```
Depth Anything V2 ──→ raw depth ──→ Guided Filter ──→ d_filt ──→ diagnostic metrics only
GT Contour ───────────────────────────────────────────────→ M4 Raycaster
```

### After (Corrected End-to-End Integration):
```
Depth Anything V2
       ↓
  raw depth (d_raw)
       ↓
 Guided Filter
       ↓
 d_filt (Mode B) / d_raw (Mode A)
       ↓
Inference-Time Contour Extraction (`extract_depth_building_contour`)
       ↓
Frozen M4 Raycaster (`measure_building_shadow_m4_physical`) [UNCHANGED]
       ↓
Actual Predicted Building Height ($H_{\text{pred}}$)
       ↓
Post-Hoc Metric Scoring against GT ($H_{\text{GT}}$)
```

---

## Key Technical Specifications

1. **Inference-Time Contour Extraction**:
   - `extract_depth_building_contour(d_map, centroid, bounding_box, margin_px=20)`:
     - Crops local depth patch $D_{\text{roi}}$ around target building ROI.
     - Computes local ground baseline level $d_{\text{bg}} = \text{percentile}(D_{\text{roi}}, 25)$ and roof top level $d_{\text{roof}} = \text{percentile}(D_{\text{roi}}, 90)$.
     - Derives adaptive depth step threshold $T = d_{\text{bg}} + 0.35 \cdot (d_{\text{roof}} - d_{\text{bg}})$.
     - Binarizes $D_{\text{roi}} > T$ and applies morphological opening ($3 \times 3$ kernel).
     - Extracts contours using `cv.findContours` and shifts coordinates to global image frame.
     - **Mode A**: Extracts $C_{\text{raw}}$ from $D_{\text{raw}}$.
     - **Mode B**: Extracts $C_{\text{filt}}$ from $D_{\text{filtered}}$.

2. **Zero Ground-Truth Leakage Guarantee**:
   - Depth map $D$ ($D_{\text{raw}}$ or $D_{\text{filt}}$) is purely monocular / inference-time data.
   - Threshold $T$ is computed strictly from the local depth histogram in $D_{\text{roi}}$.
   - Zero GT elevation values, GT DSM rasters, or GT height labels are accessed during contour extraction!
   - GT centroid $(c_x, c_y)$ and bounding box are used solely as spatial query anchors to locate building ROIs for post-hoc validation comparison.

3. **Feature Flag Rollback**:
   - `enable_guided_filter=False`: Mode A extracts contours from $D_{\text{raw}}$.
   - `enable_guided_filter=True`: Mode B extracts contours from $D_{\text{filtered}}$.

4. **Immutable M4 API Integration**:
   - `measure_building_shadow_m4_physical` accepts $C_{\text{raw}}$ (Mode A) and $C_{\text{filt}}$ (Mode B).
   - Base points $P_{\text{base}}$ are derived directly from the extracted depth boundary contacting the HSV shadow mask.

---

## Verification & Audit Strategy

1. **Frozen M4 Hash Verification**: Run `python tmp/verify_final_integrity.py` to confirm baseline SHA-256 hashes match.
2. **Unit Test Verification**: Run `python -m unittest -v shadow/test_guided_filter.py` (9/9 PASS).
3. **Contour Discontinuity Assertion**: Verify that $C_{\text{filt}} \neq C_{\text{raw}}$ and that physical height predictions differ between Mode A and Mode B across Potsdam buildings.
4. **End-to-End Benchmark Execution**: Run full benchmark `python -u shadow/run_m7_potsdam_validation.py` and report true M4 physical raycasting metrics (MAE, MedAE, RMSE, VALID rate, Category C degradation).
5. **Documentation & Audit Log Updates**: Update `output/ERROR_CORRECTION_LOG.md`, `output/M7_FINAL_RELEASE_AUDIT.md`, `output/GUIDED_FILTER_M7_FINAL_AUDIT.md`, `task.md`, and `walkthrough.md`.
