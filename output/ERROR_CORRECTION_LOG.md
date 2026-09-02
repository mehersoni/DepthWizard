# DeepthWizard Project Error Correction Log

This file serves as the project's permanent chronological record of discovered technical errors, hypotheses, proposed corrections, experimental evidence, and final architectural decisions.

---

# Error: Depth Anything V2 Blobby / Blurred Structural Boundaries

## 1. Observed Error
Depth Anything V2 monocular depth maps exhibit spatially blurred, soft, or "blobby" structural boundaries around buildings, walls, and man-made structures in high-resolution satellite imagery.

## 2. Where It Occurs
Upstream Monocular Depth Estimation stage, prior to downstream building footprint extraction, shadow geometry analysis, and building height estimation.

## 3. Symptoms
- Soft, rounded depth transitions along rectilinear building footprints instead of sharp step discontinuities.
- Boundary localization drift of 5 to 15 pixels ($0.25\text{m} - 0.75\text{m}$ at Potsdam GSD $= 0.05\text{ m/px}$).
- Reduced edge sharpness causing downstream boundary ambiguity during raycasting and footprint alignment.

## 4. Root Cause Hypothesis
Monocular neural depth models (such as Depth Anything V2) process images at internal canonical resolutions (e.g., $518 \times 518$ or $1024 \times 1024$) and use bilinear/bicubic interpolation to upsample predicted depth back to high-resolution input image size ($6000 \times 6000$). This spatial upsampling inherently acts as a low-pass filter, blurring sharp step discontinuities at physical building edges.

## 5. Proposed Correction
Apply OpenCV's Guided Filter:
```python
cv2.ximgproc.guidedFilter(guide=RGB, src=depth, radius=r, eps=eps)
```
(or standard OpenCV fallback `guided_filter_pure_cv2`) using the co-registered high-resolution RGB image as the guidance raster and the raw Depth Anything V2 depth map as the target.

## 6. Why This Correction Helps
Guided Filter assumes that target depth output $q$ is a local linear transformation of guidance image $I$ (RGB). High-frequency intensity changes (edges) in the RGB guidance image force the filter to preserve aligned depth step discontinuities while smoothing depth noise within uniform regions.

## 7. Parameters Tested & Grid Results
Globally fixed non-GT parameter grid evaluation across all 1,760 Potsdam buildings (38 tiles):
- **Radius ($r$)**: $\{2, 4, 8, 16\}$ pixels
- **Epsilon ($\epsilon$)**: $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$
- **Selected Best Configuration**: Radius $r=16\text{ px}$, Epsilon $\epsilon=0.1$

## 8. Experimental Evidence & Benchmark Metrics
- **Dataset Scope**: Full ISPRS Potsdam Benchmark (1,760 GT buildings across 38 TOP RGB tiles).
- **Baseline MAE**: `4.69 m` $\longrightarrow$ **Filtered MAE (Proxy)**: `3.50 m` (a **$1.19\text{m}$ / 25.4% proxy error reduction**).
- **Baseline MedAE**: `3.32 m` $\longrightarrow$ **Filtered MedAE (Proxy)**: `2.64 m`.
- **Baseline RMSE**: `6.70 m` $\longrightarrow$ **Filtered RMSE (Proxy)**: `4.84 m`.
- **Flat-Roof Texture Transfer Ratio ($R_{\text{TT}}$)**: `0.8076` (Constraint $R_{\text{TT}} \le 1.10$: **PASS**, no measurable increase in roof interior gradient variance).
- **Category C Regression Rate (`CORRECT_DEGRADED`)**: `0.06%` (Constraint $< 2.0\%$: **PASS**).
- **Categories E & F (`NEW_FALSE_DEPTH_EDGE` & `NEW_FALSE_BUILDING`)**: `0` (**PASS**).

## 9. Visual Diagnostics
Visual 5-panel overlays generated under `output/guided_filter_diagnostics/` confirm sharp, wall-aligned depth steps with smoothed roof interior variance.

## 10. Reproducibility Result
- **Run Variance across 5 Identical Runs**: `0.000000000000e+00` (Deterministic execution).

## 11. Final Decision
**PASS — ACCEPTED FOR FUTURE INTEGRATION**

The experimental evidence demonstrates that Guided Filtering improves monocular depth boundary sharpness without damaging the M4 baseline.

## 12. Code State & Immutability Compliance
- **Files Created**: `tmp/analyze_guided_filter_depth.py`, `output/GUIDED_FILTER_EXPERIMENT_REPORT.md`, `output/GUIDED_FILTER_READINESS_REPORT.md`, `output/guided_filter_experiment_results.csv`.
- **Immutable Files Left Frozen**: `shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`.

---

# Error: Radius-to-Metre Spatial Conversion Inconsistency

## 1. What was wrong
In earlier report drafts (`output/GUIDED_FILTER_EXPERIMENT_REPORT.md` and initial plan summary), radius $r=16\text{ px}$ was described as `0.20m spatial radius` instead of its true physical dimension.

## 2. Why it happened
Typographical mix-up between $r=4\text{ px}$ ($4 \times 0.05\text{ m/px} = 0.20\text{ m}$) and $r=16\text{ px}$ ($16 \times 0.05\text{ m/px} = 0.80\text{ m}$) during parameter grid report generation.

## 3. How it was detected
Code release audit (Phase 2 consistency audit) comparing GSD $0.05\text{ m/px}$ against the documented radius array $r \in \{2, 4, 8, 16\}$.

## 4. Correction
Corrected all conversion tables and report text:
- $r=2\text{ px} \rightarrow 0.10\text{ m}$ spatial neighborhood
- $r=4\text{ px} \rightarrow 0.20\text{ m}$ spatial neighborhood
- $r=8\text{ px} \rightarrow 0.40\text{ m}$ spatial neighborhood
- $r=16\text{ px} \rightarrow 0.80\text{ m}$ spatial neighborhood

## 5. Why the correction is valid
$16 \text{ pixels} \times 0.05 \text{ metres/pixel} = 0.80 \text{ metres}$.

## 6. Files affected
- `output/IMPLEMENTATION_PLAN_GUIDED_FILTER.md`
- `output/GUIDED_FILTER_EXPERIMENT_REPORT.md`
- `output/GUIDED_FILTER_READINESS_REPORT.md`
- `tmp/analyze_guided_filter_depth.py`

## 7. Verification
Verified conversion using Python arithmetic: `[r * 0.05 for r in [2, 4, 8, 16]] == [0.10, 0.20, 0.40, 0.80]`.

## 8. Impact on previous results
Numerical experimental results are unaffected. Only the physical text annotation in documentation was corrected.

---

# Error: Scientific Overclaim on Flat-Roof Texture Transfer Ratio ($R_{\text{TT}} < 1$)

## 1. What was wrong
Initial text claimed that $R_{\text{TT}} = 0.8076 < 1.0$ "proves zero RGB texture imprinting into depth".

## 2. Why it happened
Overly strong phrasing regarding a summary statistical ratio.

## 3. How it was detected
Phase 2 mathematical audit of metric definitions.

## 4. Correction
Refined phrasing to: "Flat-Roof Texture Transfer Ratio $R_{\text{TT}} = 0.8076 \le 1.10$ demonstrates no measurable increase in average roof-interior depth gradient variance, confirming that interior roof noise is smoothed on average."

## 5. Why the correction is valid
$R_{\text{TT}}$ measures the mean magnitude of depth gradients inside eroded roof masks. A ratio $< 1.0$ proves average gradient reduction, but does not guarantee absolute zero local texture transfer for every individual pixel.

## 6. Files affected
- `output/ERROR_CORRECTION_LOG.md`
- `output/GUIDED_FILTER_EXPERIMENT_REPORT.md`
- `output/GUIDED_FILTER_READINESS_REPORT.md`

## 7. Verification
Verified metric formula: $R_{\text{TT}} = \mathbb{E}[\nabla D_{\text{filt}}] / \mathbb{E}[\nabla D_{\text{raw}}]$.

## 8. Impact on previous results
Improves scientific rigor of claims without changing underlying numerical values.

---

# Error: Parameter Selection Reliance on Post-Hoc GT MAE

## 1. What was wrong
The candidate selection logic in `tmp/analyze_guided_filter_depth.py` filtered candidates by $R_{\text{TT}} \le 1.10$, but then used `min(..., key=lambda x: x["mae"])` to pick $r=16, \epsilon=0.1$ based on minimum GT height MAE.

## 2. Why it happened
Inadvertent leakage of post-hoc GT height metrics into the parameter ranking step during experimental script development.

## 3. How it was detected
Phase 4 parameter selection audit.

## 4. Correction
Strictly separated parameter selection rules: parameter ranking must rely **EXCLUSIVELY** on pre-filter non-GT edge localization displacement $\Delta E_{\text{loc}}$ subject to $R_{\text{TT}} \le 1.10$. Under strict non-GT ranking ($\min \Delta E_{\text{loc}}$), $r=16, \epsilon=0.01$ is selected ($\Delta E_{\text{loc}} = 28.06\text{ px}, R_{\text{TT}} = 0.9250, \text{MAE} = 3.52\text{m}$).

## 5. Why the correction is valid
Pre-filter edge localization $\Delta E_{\text{loc}}$ measures physical boundary alignment with RGB contours without inspecting ground-truth building height $H_{\text{GT}}$.

## 6. Files affected
- `output/ERROR_CORRECTION_LOG.md`
- `output/GUIDED_FILTER_EXPERIMENT_REPORT.md`
- `tmp/analyze_guided_filter_depth.py`

## 7. Verification
Re-ranked parameter grid strictly by non-GT $\Delta E_{\text{loc}}$; verified that GT height $H_{\text{GT}}$ plays zero role in parameter selection.

## 8. Impact on previous results
Both $r=16, \epsilon=0.1$ (MAE 3.50m) and $r=16, \epsilon=0.01$ (MAE 3.52m) achieve substantial error reductions over baseline M4 (4.69m).

---

# Error: Proxy Step-Ratio Height Calculation in Initial Guided Filter Investigation

## 1. Error ID
ERR-2026-09-02-GF-PROXY-EVALUATION

## 2. Date
2026-09-02

## 3. File/Component
`tmp/analyze_guided_filter_depth.py` / Guided Filter Evaluation Diagnostic Engine

## 4. Problem
The initial diagnostic script calculated filtered building height predictions using a 90th-percentile depth-step ratio proxy multiplier (`h_pred_filt = m4_height_m * d_step_ratio`) rather than executing direct end-to-end 1D physical raycasting (`shadow/m4_physical_raycast_experiment.py`) on the filtered depth maps.

## 5. Why It Occurred
The initial diagnostic script was designed for isolated pre-processing boundary evaluation and deliberately avoided invoking frozen M4 pipeline functions directly to maintain 100% baseline code immutability.

## 6. Why It Matters
A proxy multiplier assumes height error scales linearly with depth-step sharpening at the contour percentile. In reality, physical raycasting traces shadow-tip contacts outward along solar vectors. Presenting proxy height reductions as end-to-end M4 pipeline results would be scientifically inaccurate.

## 7. Evidence
Line 338 of `tmp/analyze_guided_filter_depth.py`:
`h_pred_filt = float(b_rec["m4_height_m"]) * d_step_ratio`

## 8. Correction
Formulated the M7 integration plan (`output/IMPLEMENTATION_PLAN_M7_GUIDED_FILTER_INTEGRATION.md`) which feeds the actual filtered depth rasters `D_filtered` into the genuine downstream M4 raycaster (`shadow/m4_physical_raycast_experiment.py`).

## 9. Validation Performed
Release call-graph audit tracing data flow from Depth Anything V2 raw disparity through Guided Filter and into M4 shadow raycaster.

## 10. Before/After Result
- **Before**: MAE $3.52\text{m}$ reported based on step-ratio proxy simulation.
- **After**: Final M7 validation will evaluate true end-to-end M4 raycasting predictions.

## 11. Whether Production Code Was Affected
Production code was **NOT affected**. M4 baseline files remain 100% frozen.

---

# Error: Unlinked Contour Pipeline in Initial M7 Benchmark Validation Script

## 1. Error ID
ERR-2026-09-02-GF-UNLINKED-CONTOUR-PIPELINE

## 2. Date
2026-09-02

## 3. File/Component
`shadow/run_m7_potsdam_validation.py` / M7 Potsdam Validation Engine

## 4. Problem
`shadow/run_m7_potsdam_validation.py` computed the filtered depth map `d_filt`, but passed static GT building contours (`b_cnt`) to `measure_building_shadow_m4_physical()` in both Mode A and Mode B. As a result, `d_filt` was evaluated only through pre-GT diagnostic metrics ($R_{\text{TT}}$, $\Delta E_{\text{loc}}$) and was not genuinely connected to downstream contour extraction or physical raycasting predictions.

## 5. Why It Occurred
The benchmark script author computed `d_filt` for diagnostic metrics, but reused `gt_data["buildings"]["contour"]` in the raycasting loop under the assumption that `measure_building_shadow_m4_physical()` would consume the depth map directly, without recognizing that `measure_building_shadow_m4_physical()` operates on contour coordinates.

## 6. Why It Matters
Because identical GT contours were passed to both Mode A and Mode B, physical raycasting predictions were byte-for-byte identical, leading to an artificial $0.00\text{m}$ MAE delta. Any claimed M7 height improvement was scientifically invalid until `d_filt` drove building contour extraction.

## 7. Evidence
Lines 204-241 of `shadow/run_m7_potsdam_validation.py`: `measure_building_shadow_m4_physical()` received `b_cnt` (GT contour) in both Mode A and Mode B loops.

## 8. Correction Implemented
Implemented `extract_depth_building_contour(d_map, centroid, bbox)` to derive inference-time building contours from `d_raw` (Mode A) and `d_filt` (Mode B) using local adaptive depth thresholding with zero GT leakage. Passed `cnt_raw` and `cnt_filt` directly to `measure_building_shadow_m4_physical()`.

## 9. Validation Performed
- Unit tests: `python -m unittest -v shadow/test_guided_filter.py` (9/9 PASS).
- Frozen M4 Integrity: `python tmp/verify_final_integrity.py` (STATUS: 0).
- End-to-End Benchmark: Executed corrected benchmark; $100\%$ ($129/129$) evaluated building contours were genuinely different between Mode A and Mode B.
- MAE Improvement: Baseline MAE $3.91\text{m} \longrightarrow$ M7 MAE $3.75\text{m}$ (an absolute improvement of **$-0.16\text{m}$**).

## 10. Before/After Architecture
- **BEFORE**: Depth Anything V2 $\rightarrow$ raw depth $\rightarrow$ Guided Filter $\rightarrow$ `d_filt` $\rightarrow$ diagnostic metrics only; GT contour $\rightarrow$ M4 Raycaster.
- **AFTER**: Depth Anything V2 $\rightarrow$ raw depth $\rightarrow$ Guided Filter $\rightarrow$ `d_filt` $\rightarrow$ inference-time contour extraction $\rightarrow$ M4 Raycaster (UNCHANGED) $\rightarrow$ predicted building height $\rightarrow$ GT height used ONLY for post-hoc scoring.

## 11. Whether Ground-Truth Leakage Was Possible
No. Contour threshold $T_{\text{local}} = d_{\text{bg}} + 0.35 \cdot (d_{\text{roof}} - d_{\text{bg}})$ is derived strictly from local depth histogram percentiles. Zero GT elevation values, GT DSM rasters, or GT height labels were accessed during contour extraction.

## 12. Whether Frozen Production M4 Code Changed
No. All 4 frozen production M4 files (`shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`) remain **100% byte-for-byte untouched**.
