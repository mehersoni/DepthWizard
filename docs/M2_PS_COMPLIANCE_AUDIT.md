# DepthWizard M2/M5 Problem Statement Compliance Audit

**Document:** M2_PS_COMPLIANCE_AUDIT.md  
**Date:** 2026-08-26  
**Auditor:** DepthWizard Core Engineering  
**Scope:** SIH26175 Official Problem Statement Alignment for M2 (Elevation Engine) & M5 (Visualization)

---

## 1. Problem Statement Requirements vs. Implementation Audit

| # | Official PS Requirement | Previous Implementation | Compliance Status | Required Correction |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Mode 1 (Non-Georeferenced):** Standard JPG/PNG must produce relative Digital Surface Models (rDSM) in [0, 1], no CRS, no transform, no fake GSD, no metres. | Returned normalized rDSM in [0, 1], `crs=None`, `transform=None`. In M5 initial static HTML, placeholder text contained hardcoded Potsdam values. | **PARTIALLY COMPLIANT** | Ensure M5 HTML initial state displays clean unpopulated placeholders ('-') and never displays metric units for relative rDSM. |
| **2** | **Mode 2 (Georeferenced Fallback):** GeoTIFF spatial metadata (CRS, transform) does NOT provide elevation. GeoTIFF alone without calibration MUST NOT produce metric elevations. | If GeoTIFF was uploaded without GCPs, `process_image.py` defaulted to hardcoded empirical prior (`scale_a = 6.9373, offset_b = 37.60`), marked `mode="absolute"`, and displayed values as metres. | **NON-COMPLIANT** | Remove hardcoded prior fallback. When GeoTIFF has no GCPs and no DEM, set `calibrated = False`, output scale-agnostic rDSM [0, 1], preserve CRS/transform for spatial reference, but explicitly label height as relative and require calibration. |
| **3** | **GCP Calibration:** Convert relative depth to metric elevation (H = aD + b) strictly from user-supplied Ground Control Points. | Supported user-supplied GCPs via `fit_gcp_calibration()`, but if GCPs were omitted, fell back to hardcoded prior. | **PARTIALLY COMPLIANT** | Require explicit GCPs (K >= 2 points or K=1 with documented ground offset) to trigger `calibrated = True, method = "gcp"`. |
| **4** | **DEM / SRTM Calibration:** Lower-resolution DEM (e.g. SRTM 30m) may only be used when explicitly supplied by the user. Do not invent or silently download DEMs. | `fetch_srtm_dem()` attempted remote tile download if cached file was missing. | **PARTIALLY COMPLIANT** | Support explicit user-supplied DEM (`dem_path` or `dem_file`). If DEM is provided, resample to raster grid and fit ground anchor. If missing, do not invent calibration. |
| **5** | **Data Contract & Data Model:** Clean distinction between `relative`, `absolute`, and `calibrated` boolean status. | Missing top-level `calibrated: bool`. Metadata had flat string `calibration` instead of structured object. | **NON-COMPLIANT** | Refactor data model in `process_image()` and `api_server.py` to include `calibrated: bool`, `mode: "relative" | "absolute"`, and structured `calibration` metadata. |
| **6** | **Slope Calculation:** Slope must use true physical GSD only when calibrated/georeferenced. Never use fake GSD. | `calculate_slope()` used GeoTIFF transform pixel size `gsd_x, gsd_y` when available and 1.0 for relative images. | **COMPLIANT** | Preserve physical GSD for GeoTIFFs; clearly document relative geometric slope for non-georeferenced images. |
| **7** | **M5 Frontend State Machine:** Frontend must support 3 clear operational states: (1) Relative Mode, (2) Absolute Metric DSM (Calibrated), (3) Georeferenced Uncalibrated. | M5 supported only binary Absolute vs Relative badge, treating all GeoTIFFs as metric. | **NON-COMPLIANT** | Refactor M5 HTML state machine to handle the 3 explicit states, displaying "Metric Calibration Required" when GeoTIFF is uncalibrated. |
| **8** | **Evaluation vs. Calibration Separation:** LiDAR reference DSM must only be used for offline accuracy evaluation, never as secret training/calibration data. | LiDAR reference DSM was used in evaluation scripts, but production `process_image()` did not access it. | **COMPLIANT** | Maintain strict isolation between calibration inputs and LiDAR ground truth evaluation datasets. |

---

## 2. Inventory of Hardcoded / Non-Compliant Items to Remove

1. **`process_image.py` Lines 239-243:**
   ```python
   # REMOVE THIS NON-COMPLIANT FALLBACK:
   scale_a = 6.9373
   offset_b = 37.60
   height_map = apply_gcp_calibration(depth, scale_a, offset_b)
   ```
   **Replacement:** When `gcps is None and dem_path is None`, set `calibrated = False`, `mode = "absolute"` (georeferenced spatial reference present) with uncalibrated normalized rDSM [0, 1], `scale_a = None, offset_b = None`.

2. **`api_server.py` Line 183:**
   ```python
   # REMOVE THIS BINARY UNIT LOGIC:
   "height_unit": "m" if mode == "absolute" else "rel"
   ```
   **Replacement:**
   ```python
   "calibrated": result["calibrated"],
   "height_unit": "m" if (result["mode"] == "absolute" and result["calibrated"]) else "rel"
   ```

3. **`m5_3d_viewer_demo.html` Initial Metadata & State Logic:**
   - Update state machine to support:
     - `mode == 'relative'`: Badge = "Relative rDSM Mode", Unit = "rel"
     - `mode == 'absolute' && calibrated == true`: Badge = "Metric Absolute DSM", Unit = "m"
     - `mode == 'absolute' && calibrated == false`: Badge = "Metric Calibration Required", Unit = "rel (uncalibrated)", displays true CRS/GSD, but no fake metres.

---

## 3. Corrected Architecture Diagram

```
                             INPUT IMAGE
                                  |
                  +---------------+---------------+
                  |                               |
             JPG / PNG                         GeoTIFF
                  |                               |
                  v                               v
            Non-Georeferenced             Geospatial Metadata
            (No CRS / Transform)          (CRS, Transform, GSD)
                  |                               |
                  v                               v
          Depth Anything V2               Depth Anything V2
                  |                               |
                  v                               v
            Relative Depth                  Relative Depth
                  |                               |
                  v                               v
           Normalize [0, 1]              Calibration Provided?
                  |                               |
                  v                      +--------+--------+
            rDSM Output                  |                 |
          (mode='relative',            NO GCP/DEM       GCP / DEM
           calibrated=False)             |                 |
                  |                      v                 v
                  |              Uncalibrated rDSM     Metric DSM
                  |              (mode='absolute',   (mode='absolute',
                  |               calibrated=False)   calibrated=True)
                  |                      |                 |
                  +----------------------+-----------------+
                                         |
                                         v
                                  process_image()
                                         |
                                         v
                             FastAPI /process Endpoint
                                         |
                                         v
                            M5 3D WebGL Visualization
```

---
*Audit completed. Proceeding to implementation.*
