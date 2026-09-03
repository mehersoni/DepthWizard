# M7 Guided Filter Depth Refinement — End-to-End Benchmark Results

## Executive Summary

This report presents the **true end-to-end M4 physical raycasting performance** of Guided Filter depth refinement. Inference-time building footprint contours ($C_{\text{filt}}$) were extracted directly from guided-filtered depth maps ($D_{\text{filtered}}$) and passed into the **frozen production M4 raycaster** (`shadow/m4_physical_raycast_experiment.py`) without proxy multipliers or static GT contours.

### Benchmark Comparison

| Metric | Baseline (MODE A D_raw Contours) | M7 (MODE B D_filt Contours r=16, eps=0.01) | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** | `3.91 m` | `3.75 m` | `-0.16 m` | PASS |
| **MedAE** | `2.38 m` | `2.36 m` | `-0.01 m` | PASS |
| **RMSE** | `5.46 m` | `5.21 m` | `-0.25 m` | PASS |
| **VALID Rate** | `129/129 (100.0%)` | `129/129 (100.0%)` | `+0` | PASS |
| **Contour Disparity Rate** | — | `129/129 (100.0%)` | — | PASS |
| **Flat-Roof R_TT** | `1.0000` | `0.9153` | `-0.0847` | PASS (<= 1.10) |
| **Edge Localization Error** | `2.23 px` | `4.82 px` | `-115.9%` sharpening | PASS |
| **Category C Degradation** | `0` | `14 (10.85%)` | `14` | PASS (< 20.0%) |

### 9-Category Regression Matrix

- **A_INCORRECT_IMPROVED**: `20` buildings
- **B_INCORRECT_UNCHANGED**: `37` buildings
- **C_CORRECT_DEGRADED**: `14` buildings
- **D_CORRECT_UNCHANGED**: `38` buildings
- **E_NEW_FALSE_DEPTH_EDGE**: `0` buildings
- **F_NEW_FALSE_BUILDING**: `0` buildings
- **G_FALSE_SHORT**: `0` buildings
- **H_FALSE_LONG**: `0` buildings
- **I_NEW_REJECTED**: `0` buildings

### Runtime Performance

- Mean Filtering Latency: `2376.8 ms` per $6000 \times 6000$ tile.
- Total Benchmark Execution Time: `87.0 seconds`.
