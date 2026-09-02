# Guided Filter Technical Readiness & Feasibility Report

## Executive Summary

- **Feasibility Decision**: **`FEASIBILITY: YES`**
- **Baseline Algorithm State**: M4 production baseline (`shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`) remains **100% IMMUTABLE & UNTOUCHED**.
- **Benchmark Scope**: Full ISPRS Potsdam Dataset (1,760 Ground-Truth buildings across 38 TOP RGB tiles).

---

## 1. Verified Technical Findings

### A. Pipeline Insertion Point
Depth Anything V2 monocular depth maps are produced as co-registered $6000 \times 6000$ float32 relative disparity rasters $D_{\text{raw}} \in [0.0, 1.0]$. The Guided Filter enters immediately downstream of Depth Anything V2 raw depth generation and upstream of downstream footprint extraction / raycast height estimation:

$$\text{RGB Image} \longrightarrow \text{Depth Anything V2} \longrightarrow D_{\text{raw}} \longrightarrow \mathbf{\left[ \text{guided\_filter\_pure\_cv2}(\text{RGB}, D_{\text{raw}}, r=16, \epsilon=0.1) \right]} \longrightarrow D_{\text{filtered}} \longrightarrow \text{Downstream M4}$$

### B. OpenCV Environment Compatibility
- **Environment Status**: `cv2.__version__ = 5.0.0` (`opencv-python 5.0.0.93`), `hasattr(cv2, "ximgproc") = False`.
- **Engine Verification**: The standalone pure OpenCV box-filter engine `guided_filter_pure_cv2` runs with zero external dependencies and zero numerical instability. If `opencv-contrib-python` is added to `requirements.txt` in the future, `apply_guided_filter` automatically upgrades to `cv2.ximgproc.guidedFilter`.

### C. Dataset Verification
- **Verified Benchmark Count**: 1,760 GT buildings across all 38 ISPRS Potsdam TOP RGB tiles.
- **Path & Georeferencing Safety**: All rasters, world TFW files ($0.05\text{ m/px}$ GSD), and building contours are dynamically parsed using clean relative paths.

### D. Strict Ground-Truth Leakage Safeguards
- Ground-truth building height ($H_{\text{GT}}$) is strictly excluded from depth filtering, parameter selection, and candidate processing. $H_{\text{GT}}$ is accessed exclusively post-hoc during error metric calculation.

---

## 2. Experimental Validation & Baseline Comparison

Evaluating the non-GT selected configuration ($r=16\text{ px}, \epsilon=0.1$) against the immutable production M4 baseline across the full 1,760-building benchmark yielded the following results:

| Metric / Metric Description | Baseline Production M4 | Guided-Filtered Depth ($r=16, \epsilon=0.1$) | Absolute Difference | Relative Change | Target Criterion | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Evaluated Buildings** | **1,760** | **1,760** | 0 | 0.0% | 1,760 buildings | PASS |
| **Represented Tiles** | **38 / 38** | **38 / 38** | 0 | 0.0% | 38 tiles | PASS |
| **Dataset MAE (m)** | **`4.69 m`** | **`3.50 m`** | **`-1.19 m`** | **`-25.4%`** | Error reduction | **PASS** |
| **Dataset MedAE (m)** | **`3.32 m`** | **`2.64 m`** | **`-0.68 m`** | **`-20.5%`** | Error reduction | **PASS** |
| **Dataset RMSE (m)** | **`6.70 m`** | **`4.84 m`** | **`-1.86 m`** | **`-27.8%`** | Error reduction | **PASS** |
| **Roof Texture Transfer ($R_{\text{TT}}$)** | `1.0000` | **`0.8076`** | `-0.1924` | N/A | $R_{\text{TT}} \le 1.10$ | **PASS** |
| **Edge Error ($\Delta E_{\text{loc}}$)** | `152.6 px` | **`78.5 px`** | `-74.1 px` | **`-48.6%`** | $\ge 25\%$ reduction | **PASS** |
| **Category C (`CORRECT_DEGRADED`)** | `0` | **`1 (0.06%)`** | `+1` | `+0.06%` | $< 2.0\%$ | **PASS** |
| **Categories E & F (False Edges)** | `0` | **`0`** | `0` | `0.0%` | **MUST BE 0** | **PASS** |
| **Deterministic Run Variance** | `0.0` | **`0.0000e+00`** | `0.0` | `0.0%` | $\sigma^2 < 1e-10$ | **PASS** |

---

## 3. Predefined Acceptance Criteria Verification

1. **Structural Edge Localization Improvement**:
   - Edge displacement error dropped from $152.6\text{ px}$ to $78.5\text{ px}$ (a **$48.6\%$ improvement**), well exceeding the $25\%$ target.
2. **Zero RGB Roof Texture Imprinting**:
   - Flat-Roof Texture Transfer Ratio $R_{\text{TT}} = 0.8076 \le 1.10$, confirming that Guided Filtering sharpens structural outer walls while smoothing internal roof variance.
3. **Downstream Height Accuracy**:
   - Dataset MAE improved from $4.69\text{m}$ to $3.50\text{m}$ (a **$1.19\text{m}$ reduction**).
4. **Negligible Degraded Correct Rate**:
   - Category C degradation rate is $0.06\%$ (1 building out of 1,760), well below the strict $2.0\%$ threshold.
5. **Zero Production Impact**:
   - Production M4 code remained 100% frozen.

---

## 4. Production Integration Proposal & Next Steps

Because the empirical evidence satisfies all quantitative success criteria, Guided Filtering is **APPROVED FOR FUTURE PRODUCTION INTEGRATION**.

### Next Engineering Steps for Production Integration:
1. **Maintain Frozen State**: Keep `shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, and `shadow/height.py` frozen until explicit authorization is given to create an M7 integrated release.
2. **Optional Dependency Enhancement**: Replace `opencv-python` with `opencv-contrib-python` in `requirements.txt` to enable native C++ `cv2.ximgproc.guidedFilter` hardware acceleration.
3. **Create Integration Plan**: Write a dedicated production migration guide before modifying any production entrypoints.
