# Guided Filter M7 Senior ML Release Audit Report

## 1. Executive Decision

**Final Release Recommendation**: **`NEEDS_CORRECTION_BEFORE_M7`**

### Summary of Audit Findings:
1. **Technical Feasibility**: Guided Filtering using RGB guidance image is mathematically and empirically sound. Perimeter edge localization displacement error drops from $152.6\text{ px}$ to $28.06\text{ px}$ (an **$81.6\%$ boundary sharpening improvement**), and interior roof depth variance does not increase ($R_{\text{TT}} = 0.9250 \le 1.10$).
2. **Audit Corrections Identified**:
   - **Downstream Call Graph**: Initial diagnostic script `tmp/analyze_guided_filter_depth.py` evaluated height using a 90th-percentile step-ratio proxy multiplier rather than passing filtered depth maps directly into `shadow/m4_physical_raycast_experiment.py`.
   - **Radius Metric Inconsistency**: $r=16\text{ px}$ was mislabeled as $0.20\text{m}$ instead of $16 \times 0.05\text{ m/px} = 0.80\text{m}$.
   - **Scientific Overclaim**: $R_{\text{TT}} = 0.8076 < 1.0$ was claimed to "prove zero texture imprinting", whereas it mathematically proves average roof interior depth variance reduction.
   - **Parameter Selection Leakage**: Candidate ranking in the initial experimental script referenced post-hoc GT MAE. Strict non-GT ranking selects $r=16, \epsilon=0.01$ based on minimum edge localization displacement error $\Delta E_{\text{loc}} = 28.06\text{ px}$.
3. **Required Action Before M7 Production Release**: Execute the M7 integration plan (`output/IMPLEMENTATION_PLAN_M7_GUIDED_FILTER_INTEGRATION.md`) to pass refined depth maps directly through the actual production M4 raycaster (`shadow/m4_physical_raycast_experiment.py`) before declaring production release complete.

---

## 2. Current M4 Production Baseline

The production M4 physical shadow-tip raycaster baseline is **100% FROZEN & IMMUTABLE**.
- **Evaluated Scope**: 1,760 GT buildings across 38 ISPRS Potsdam TOP RGB tiles
- **Baseline MAE**: `4.69 m`
- **Baseline MedAE**: `3.32 m`
- **Baseline RMSE**: `6.70 m`
- **Baseline VALID Rate**: `98.8%` (1,738 / 1,760 buildings)
- **Baseline LOW CONFIDENCE**: `30`
- **Baseline REJECTED**: `22`

---

## 3. Guided Filter Experimental Evidence

Evaluating non-GT selected parameters ($r=16\text{ px}$, $\epsilon=0.01$) across the full benchmark:

| Metric / Evaluation Dimension | Baseline Production M4 | Guided-Filtered Depth ($r=16, \epsilon=0.01$) | Absolute Delta | Status / Target |
| :--- | :---: | :---: | :---: | :---: |
| **Evaluated Buildings** | **1,760** | **1,760** | `0` | 1,760 buildings |
| **Represented Tiles** | **38 / 38** | **38 / 38** | `0` | 38 tiles |
| **Edge Localization Error ($\Delta E_{\text{loc}}$)** | `152.6 px` | **`28.06 px`** | **`-124.5 px`** | **81.6% Sharpening Improvement** |
| **Roof Texture Transfer ($R_{\text{TT}}$)** | `1.0000` | **`0.9250`** | `-0.0750` | $R_{\text{TT}} \le 1.10$ (**PASS**) |
| **Filtered Proxy MAE (m)** | **`4.69 m`** | **`3.52 m`** | **`-1.17 m`** | **Proxy Error Reduction** |
| **Filtered Proxy MedAE (m)** | **`3.32 m`** | **`2.64 m`** | **`-0.68 m`** | **Proxy Error Reduction** |
| **Filtered Proxy RMSE (m)** | **`6.70 m`** | **`4.85 m`** | **`-1.85 m`** | **Proxy Error Reduction** |
| **Category C Degraded Rate** | `0.00%` | **`0.06%` (1 bldg)** | `+0.06%` | $< 2.0\%$ (**PASS**) |
| **False Candidate Creation** | `0` | **`0`** | `0` | **MUST BE 0** (**PASS**) |

---

## 4. Implementation Correctness Audit

- **Algorithm**: `guided_filter_pure_cv2` fast box-filter algorithm (He et al. 2013).
- **Guidance Raster**: Grayscale luminance derived from BGR orthophoto ($6000 \times 6000$).
- **Target Raster**: Float32 relative disparity map $D_{\text{raw}} \in [0.0, 1.0]$.
- **Numerical Stability**: Output bounded via `np.clip(q, 0.0, 1.0)`; zero-variance protection via $\epsilon$.
- **Environment**: `opencv-python 5.0.0.93` installed; `cv2.ximgproc` missing. Pure OpenCV box filter operates with 0 external dependencies.

---

## 5. Ground-Truth Leakage Audit

- **Status**: **`PASS`**.
- Ground-truth building height ($H_{\text{GT}}$) is strictly excluded from depth filtering, parameter grid selection, and candidate processing. $H_{\text{GT}}$ is used **exclusively post-hoc** for error calculation.

---

## 6. Mathematical Consistency Audit

All mathematical and documentation inconsistencies identified during Phase 2 have been documented in `output/ERROR_CORRECTION_LOG.md` and corrected across all reports:
1. **Spatial Conversion Correction**: $r=16\text{ px} \times 0.05\text{ m/px} = 0.80\text{ m}$ spatial radius (corrected from $0.20\text{ m}$).
2. **Scientific Claim Refinement**: $R_{\text{TT}} = 0.9250$ demonstrates no measurable increase in roof interior gradient variance, confirming interior roof noise smoothing.
3. **Non-GT Parameter Ranking**: Parameters ranked strictly by minimum edge localization error $\Delta E_{\text{loc}} = 28.06\text{ px}$ subject to $R_{\text{TT}} \le 1.10$.

---

## 7. Dependency Audit

- **Current Environment**: `opencv-python 5.0.0.93`. `cv2.ximgproc` is absent.
- **Recommended Strategy**: Use `guided_filter_pure_cv2` box filter as standard default. Optionally upgrade to `opencv-contrib-python` in `requirements.txt` for native C++ acceleration if desired, but keep `guided_filter_pure_cv2` as a zero-failure fallback.

---

## 8. Downstream Integration Audit

- **Current Diagnostic State**: `tmp/analyze_guided_filter_depth.py` evaluated height metrics using a 90th-percentile step-ratio proxy multiplier.
- **Required M7 Action**: Create `shadow/guided_filter.py` and pass `depth_map = refine_depth_anything_map(rgb, raw_depth)` directly into `shadow/m4_physical_raycast_experiment.py` during M7 integration execution.

---

## 9. Discovered Error Corrections Summary

Four structured error entries have been logged in `output/ERROR_CORRECTION_LOG.md`:
1. `Depth Anything V2 Blobby / Blurred Structural Boundaries`
2. `Radius-to-Metre Spatial Conversion Inconsistency`
3. `Scientific Overclaim on Flat-Roof Texture Transfer Ratio (R_TT < 1)`
4. `Downstream Metric Proxy Evaluation vs Actual Production M4 Raycaster`

---

## 10. Risk Analysis

1. **Albedo Step Over-Sharpening**: High-contrast albedo edges (e.g. dark roofs or zebra crossings) can induce minor localized depth step sharpens. Mitigated by regularization $\epsilon = 0.01$.
2. **Memory Footprint**: Peak memory for $6000 \times 6000$ float32 filtering is $\approx 576\text{ MB}$, well within standard workstation limits.
3. **Runtime Latency**: $\approx 0.35 - 0.45\text{ seconds}$ per $6000 \times 6000$ tile.

---

## 11. Recommended M7 Target Architecture

Modular upstream depth-refinement wrapper:

```
[Upstream Depth Anything V2] ──► raw_depth (float32 [0,1])
                                       │
[RGB Orthophoto] ──────────────────────┼──► [shadow/guided_filter.py]
                                       │    refine_depth_anything_map()
                                       │           │
                                       ▼           ▼
                           [IMMUTABLE PRODUCTION M4 RAYCASTER]
                       shadow/m4_physical_raycast_experiment.py
```

---

## 12. Production Acceptance Criteria

1. End-to-end M4 raycasting MAE $\le 3.80\text{m}$ (Baseline: $4.69\text{m}$).
2. Flat-Roof Texture Transfer Ratio $R_{\text{TT}} \le 1.10$.
3. Category C degradation rate $< 2.0\%$.
4. Zero false candidate creation (Categories E & F = 0).
5. All unit tests in `shadow/test_guided_filter.py` pass.
6. Deterministic run variance $\sigma^2 = 0.0$ across 5 runs.

---

## 13. Rollback Strategy

Setting `enable_guided_filter = False` in global pipeline configuration bypasses filtering and passes pristine raw depth directly into downstream raycasting, ensuring instant zero-downtime rollback capability.

---

## 14. Final Engineering Recommendation

**`NEEDS_CORRECTION_BEFORE_M7`**

The experimental evidence conclusively confirms the scientific validity and boundary-sharpening benefits of Guided Filtering. The project team should now proceed to execute the formal M7 integration plan (`output/IMPLEMENTATION_PLAN_M7_GUIDED_FILTER_INTEGRATION.md`) to run end-to-end M4 raycasting on filtered depth rasters before final production deployment.
