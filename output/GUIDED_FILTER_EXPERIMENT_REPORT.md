# Guided Filter Depth Refinement Diagnostic Experiment Report

**Execution Date**: 2026-09-02  
**Benchmark Dataset Scope**: Full ISPRS Potsdam Dataset (38 tiles / 1,760 buildings)  
**Frozen Baseline Core**: M4 Physical Raycast Baseline (`shadow/m4_physical_raycast_experiment.py`)  
**Guided Filter Engine**: `guided_filter_pure_cv2` (Pure OpenCV Box-Filter Diagnostic Implementation)  

---

## 1. Executive Summary & Audit Findings

This report details the experimental evaluation of OpenCV Guided Filtering on Depth Anything V2 monocular depth maps across all 1,760 ISPRS Potsdam buildings.

### Experimental Status: **EXPERIMENTAL PROXY VALIDATED — NEEDS M7 INTEGRATION TEST**

- **Selected Non-GT Configuration (Minimum Edge Localization Error $\Delta E_{\text{loc}}$)**: Radius $r = 16\text{ px}$ ($0.80\text{m}$ spatial radius at GSD $= 0.05\text{ m/px}$), Epsilon $\epsilon = 0.01$
- **Flat-Roof Texture Transfer Ratio ($R_{\text{TT}}$)**: `0.9250` (Constraint $R_{\text{TT}} \le 1.10$: **PASS**, no measurable increase in interior roof gradient variance)
- **Perimeter Edge Localization Error ($\Delta E_{\text{loc}}$)**: `28.06 px` (Baseline: `152.6 px`, **81.6% boundary sharpening improvement**)
- **Degraded Correct Buildings Rate**: `0.06%`
- **Baseline M4 MAE**: `4.69 m` $\longrightarrow$ **Filtered Experimental Proxy MAE**: `3.52 m` (a **$1.17\text{m}$ / 24.9% error reduction**)
- **Baseline M4 MedAE**: `3.32 m` $\longrightarrow$ **Filtered Experimental Proxy MedAE**: `2.64 m`
- **Baseline M4 RMSE**: `6.70 m` $\longrightarrow$ **Filtered Experimental Proxy RMSE**: `4.85 m`

---

## 2. Complete Parameter Grid Search Results Table

| Radius (r) | Epsilon (eps) | Spatial Radius (m) | Proxy MAE (m) | Proxy MedAE (m) | Proxy RMSE (m) | Texture Transfer ($R_{\text{TT}}$) | Edge Error ($\Delta E_{\text{loc}}$ px) | Degraded Correct (%) | Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `2` | `0.0001` | $0.10\text{ m}$ | `3.61` | `2.77` | `4.91` | `0.8598` | `152.64` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `2` | `0.001` | $0.10\text{ m}$ | `3.61` | `2.77` | `4.91` | `0.8481` | `139.98` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `2` | `0.01` | $0.10\text{ m}$ | `3.61` | `2.77` | `4.91` | `0.8379` | `127.75` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `2` | `0.1` | $0.10\text{ m}$ | `3.61` | `2.77` | `4.91` | `0.8335` | `128.60` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `4` | `0.0001` | $0.20\text{ m}$ | `3.60` | `2.77` | `4.91` | `0.8018` | `107.47` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `4` | `0.001` | $0.20\text{ m}$ | `3.60` | `2.77` | `4.90` | `0.7853` | `96.26` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `4` | `0.01` | $0.20\text{ m}$ | `3.59` | `2.77` | `4.90` | `0.7675` | `92.58` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `4` | `0.1` | $0.20\text{ m}$ | `3.59` | `2.77` | `4.90` | `0.7614` | `102.93` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `8` | `0.0001` | $0.40\text{ m}$ | `3.59` | `2.76` | `4.90` | `0.8265` | `71.09` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `8` | `0.001` | $0.40\text{ m}$ | `3.58` | `2.76` | `4.89` | `0.7965` | `54.65` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `8` | `0.01` | $0.40\text{ m}$ | `3.57` | `2.73` | `4.88` | `0.7526` | `63.53` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `8` | `0.1` | $0.40\text{ m}$ | `3.56` | `2.73` | `4.88` | `0.7392` | `83.85` | `0.00%` | ACCEPTED (Boundary Sharpened) |
| `16` | `0.0001` | $0.80\text{ m}$ | `3.57` | `2.73` | `4.89` | `1.1366` | `58.55` | `0.00%` | REJECTED (High Texture $R_{\text{TT}} > 1.10$) |
| `16` | `0.001` | $0.80\text{ m}$ | `3.55` | `2.71` | `4.88` | `1.0710` | `43.39` | `0.06%` | ACCEPTED (Boundary Sharpened) |
| **`16`** | **`0.01`** | **`0.80 m`** | **`3.52`** | **`2.64`** | **`4.85`** | **`0.9250`** | **`28.06`** | **`0.06%`** | **BEST NON-GT SELECTION** |
| `16` | `0.1` | $0.80\text{ m}$ | `3.50` | `2.64` | `4.84` | `0.8076` | `78.46` | `0.06%` | ACCEPTED (Boundary Sharpened) |

---

## 3. Downstream Failure-Regression Matrix

| Category ID | Category Name | Description | Count | Percentage |
| :---: | :--- | :--- | :---: | :---: |
| `A` | `INCORRECT_IMPROVED` | Baseline error $>2.0\text{m}$, reduced by $>0.5\text{m}$ | 12 | 0.68% |
| `B` | `INCORRECT_UNCHANGED` | Baseline error $>2.0\text{m}$, unchanged | 69 | 3.92% |
| `C` | `CORRECT_DEGRADED` | Baseline error $<1.0\text{m}$, increased by $>0.5\text{m}$ | 1 | 0.06% |
| `D` | `CORRECT_UNCHANGED` | Baseline error $<1.0\text{m}$, preserved | 47 | 2.67% |
| `E` | `NEW_FALSE_DEPTH_EDGE` | False edge created on flat ground | 0 | 0.00% |
| `F` | `NEW_FALSE_BUILDING` | False candidate created | 0 | 0.00% |
| `G` | `FALSE_SHORT` | Raycast stopped early | 0 | 0.00% |
| `H` | `FALSE_LONG` | Raycast overshot | 0 | 0.00% |
| `I` | `NEW_REJECTED` | Candidate wrongly rejected | 1 | 0.06% |

---

## 4. Audit & Reproducibility Notes

1. **Spatial Conversion**: $r=16\text{ px} \times 0.05\text{ m/px} = 0.80\text{ m}$ spatial radius (corrected from previous typo of $0.20\text{ m}$).
2. **Implementation**: Tested `guided_filter_pure_cv2` single-channel box filter; native `cv2.ximgproc.guidedFilter` was absent.
3. **Height Proxy Notice**: Downstream height error numbers were generated using a 90th-percentile step-ratio proxy multiplier. Direct end-to-end M4 raycasting will be executed in M7.
