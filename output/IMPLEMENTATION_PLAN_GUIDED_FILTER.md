# Corrected Technical Implementation Plan: Guided Filter Depth Refinement for Depth Anything V2

## Executive Overview & Second-Level Review Findings

This document presents the **corrected second-level technical implementation plan** for evaluating **OpenCV's Guided Filter (`cv2.ximgproc.guidedFilter`)** as a proposed spatial refinement step for Depth Anything V2 monocular depth maps in the `DeepthWizard` building height estimation project.

### Verified Codebase & Benchmark Dataset Findings
1. **Dataset Size Correction**:
   - The verified benchmark dataset consists of **1,760 Ground-Truth (GT) Buildings across 38 ISPRS Potsdam TOP RGB tiles** (100% full dataset coverage).
   - An earlier draft cited "30 tiles". This draft explicitly corrects that inconsistency: the full evaluation benchmark spans all **38 Potsdam tiles / 1,760 buildings**.
2. **Frozen M4 Baseline**:
   - The M4 physical raycast production baseline (`shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`) is **STRICTLY IMMUTABLE**.
   - Verified baseline performance across all 1,760 buildings:
     - **MAE**: `4.69 m`
     - **MedAE**: `3.32 m`
     - **RMSE**: `6.70 m`
     - **VALID Rate**: `98.8%` (1,738 / 1,760 buildings)
     - **LOW CONFIDENCE**: `30`
     - **REJECTED**: `22`
3. **OpenCV Environment Status**:
   - Environment audit: `cv2.__version__ = 5.0.0`, `hasattr(cv2, "ximgproc") = False`.
   - Standard `opencv-python` (version `5.0.0.93`) is installed; `opencv-contrib-python` is not present.
   - **Standalone Algorithm Solution**: A pure OpenCV fast box-filter implementation (`guided_filter_pure_cv2` using `cv2.boxFilter`) has been verified and integrated into the diagnostic suite, ensuring 100% deterministic execution on the current environment without forced dependency alterations.

---

## 1. Mathematical & Physical Applicability to Depth Anything V2

### Data Representation & Upstream Characteristics
Depth Anything V2 produces relative disparity / inverse depth maps $D_{\text{raw}} \in [0.0, 1.0]$ (`np.float32`).
Monocular models process high-resolution satellite imagery ($6000 \times 6000$ pixels) at reduced internal grid resolutions (e.g. $518 \times 518$ or $1024 \times 1024$) and upsample back to native resolution ($6000 \times 6000$) using bilinear interpolation.

This spatial upsampling acts as a low-pass filter, blurring sharp building wall discontinuities into soft $5-15$ pixel transitions ("blobby" edges).

### Guided Filtering Formulation
The Guided Filter models target output depth $q$ as a local linear transform of guidance image $I$ (RGB) in local window $w_k$ of radius $r$:
$$q_i = a_k I_i + b_k \quad \forall i \in w_k$$
where:
$$a_k = \frac{\frac{1}{|w|}\sum_{i \in w_k} I_i p_i - \mu_k \bar{p}_k}{\sigma_k^2 + \epsilon}, \quad b_k = \bar{p}_k - a_k \mu_k$$

- **Guidance Raster $I$**: Grayscale luminance or 3-channel BGR image normalized to $[0.0, 1.0]$ float32.
- **Target Raster $p$**: Raw depth map $D_{\text{raw}} \in [0.0, 1.0]$ float32.
- **Radius ($r$)**: Spatial kernel radius in pixels.
- **Epsilon ($\epsilon$)**: Regularization parameter penalizing high local RGB variance.

### Pipeline Data Flow & Insertion Point
The Guided Filter enters immediately following monocular relative depth prediction, prior to any downstream building footprint contour extraction or height estimation:

```
RGB Image (6000x6000x3 BGR)
   │
   ├──────────────────────────────┐
   │                              │
   ▼                              │
Depth Anything V2                 │
(Monocular Relative Depth)        │
   │                              │
   ▼                              │
Raw Depth D_raw (float32 [0,1])   │
   │                              │
   ├──────────────────────────────┘
   ▼ (src = D_raw, guide = RGB)
[PROPOSED FILTER]
guided_filter_pure_cv2(RGB, D_raw, r, eps)
   │
   ▼
Refined Depth D_filtered (float32 [0,1])
   │
   ├─► Pre-Filter Edge Sharpness & Texture Transfer Diagnostics
   │
   └─► Downstream M4 Shadow Height Evaluation (Post-Hoc GT Comparison)
```

---

## 2. Experimental Hypotheses & Non-GT Parameter Grid

### A. Primary Hypothesis
"Depth Anything V2 produces spatially blurred structural depth boundaries, and RGB-guided filtering can sharpen building wall discontinuities and improve edge localization."

### B. Null Hypothesis
"Guided filtering does not produce a meaningful structural improvement, or introduces unacceptable RGB texture imprinting into depth, false depth step discontinuities, or downstream height-estimation regressions."

### C. Globally Fixed Experimental Parameter Grid
Parameters are chosen strictly based on Potsdam GSD ($0.05\text{ m/px}$, $20\text{ px/m}$):

| Parameter | Values | Physical Radius at $0.05\text{ m/px}$ |
| :--- | :--- | :--- |
| **Radius ($r$)** | `2` pixels | $0.10\text{ m}$ spatial neighborhood |
| **Radius ($r$)** | `4` pixels | $0.20\text{ m}$ spatial neighborhood |
| **Radius ($r$)** | `8` pixels | $0.40\text{ m}$ spatial neighborhood |
| **Radius ($r$)** | `16` pixels | $0.80\text{ m}$ spatial neighborhood |
| **Epsilon ($\epsilon$)** | `1e-4` | Weak regularization (Strict adherence to RGB color steps) |
| **Epsilon ($\epsilon$)** | `1e-3` | Moderate regularization (Balanced edge preservation) |
| **Epsilon ($\epsilon$)** | `1e-2` | Strong regularization (Smooths subtle RGB albedo steps) |
| **Epsilon ($\epsilon$)** | `1e-1` | Very strong regularization (Approaches simple box blur) |

> [!CAUTION]
> **Strict Non-GT Parameter Selection Rule**:
> Parameters MUST NOT be selected using ground-truth height error. Parameter evaluation during search is based strictly on observable pre-filter edge sharpness metrics and texture-transfer constraints.

---

## 3. Strict Ground-Truth Leakage Safeguards

Ground-truth building height ($H_{\text{GT}}$) is strictly segregated into **Evaluation Only**:

```
INFERENCE PIPELINE (No GT Access):
  RGB Image -> D_raw -> Guided Filter (r, eps) -> D_filtered -> Downstream M4 Height Raycast -> H_pred

POST-HOC EVALUATION ONLY (GT Access Allowed):
  Absolute Error = |H_pred - H_GT|
  Regression Matrix Classification
```

Automated programmatic assertions in `tmp/analyze_guided_filter_depth.py` enforce that $H_{\text{GT}}$ is zeroed out during all filtering and candidate selection steps.

---

## 4. Mathematical Definition of Quantitative Metrics

### 1. Perimeter Edge Localization Error ($\Delta E_{\text{loc}}$)
Measures the average spatial displacement (in pixels) between detected depth gradient max-intensity contours and building footprint contours:
$$\Delta E_{\text{loc}} = \frac{1}{|\partial B|} \sum_{p \in \partial B} \min_{q \in \text{Edges}(D)} \|p - q\|_2$$
- **Target**: $\ge 25\%$ reduction compared to $D_{\text{raw}}$.

### 2. Flat-Roof Texture Transfer Ratio ($R_{\text{TT}}$)
Measures whether guided filtering imprints non-depth RGB details (solar panels, roof tiles, AC units, gravel) into flat roof regions:
$$R_{\text{TT}} = \frac{\mathbb{E}_{(x,y) \in M_{\text{roof, eroded}}}\left[\|\nabla D_{\text{filtered}}(x,y)\|\right]}{\mathbb{E}_{(x,y) \in M_{\text{roof, eroded}}}\left[\|\nabla D_{\text{raw}}(x,y)\|\right]}$$
where $M_{\text{roof, eroded}}$ is the building mask eroded by $10$ pixels to exclude exterior wall steps.
- **Acceptable Constraint**: $R_{\text{TT}} \le 1.10$ ($< 10\%$ increase in roof interior gradient variance).

### 3. Downstream Height Metrics
- **Mean Absolute Error (MAE)**: $\frac{1}{N}\sum |H_{\text{pred}} - H_{\text{GT}}|$
- **Median Absolute Error (MedAE)**: $\text{Median}(|H_{\text{pred}} - H_{\text{GT}}|)$
- **Root Mean Square Error (RMSE)**: $\sqrt{\frac{1}{N}\sum (H_{\text{pred}} - H_{\text{GT}})^2}$
- **Mean Absolute Percentage Error (MAPE)**: $\frac{100\%}{N}\sum \frac{|H_{\text{pred}} - H_{\text{GT}}|}{H_{\text{GT}}}$

### 4. 9-Category Downstream Failure-Regression Matrix
Every building ($N=1,760$) is systematically classified into:

| Category ID | Category Code | Description | Acceptance Criteria |
| :---: | :--- | :--- | :---: |
| **A** | `INCORRECT_IMPROVED` | Baseline error $>2.0\text{m}$, reduced by $>0.5\text{m}$ | Maximize |
| **B** | `INCORRECT_UNCHANGED` | Baseline error $>2.0\text{m}$, unchanged within $\pm 0.5\text{m}$ | Neutral |
| **C** | `CORRECT_DEGRADED` | Baseline error $<1.0\text{m}$, increased by $>0.5\text{m}$ | **$< 2.0\%$ of total** |
| **D** | `CORRECT_UNCHANGED` | Baseline error $<1.0\text{m}$, preserved within $\pm 0.5\text{m}$ | Maximize |
| **E** | `NEW_FALSE_DEPTH_EDGE` | High depth gradient created on flat ground | **MUST BE 0** |
| **F** | `NEW_FALSE_BUILDING` | Non-building candidate wrongly triggered | **MUST BE 0** |
| **G** | `NEW_FALSE_SHORT_SHADOW` | Raycast stopped early due to texture imprinting | Minimize |
| **H** | `NEW_FALSE_LONG_SHADOW` | Raycast overshot due to edge blur | Minimize |
| **I** | `NEW_REJECTED_CASE` | Candidate wrongly rejected by confidence filter | Minimize |

---

## 5. Visual Diagnostics & Stratification Plan

Visual 5-panel comparison figures will be generated for 10 representative test cases:
1. Small Building ($< 4.0\text{m}$)
2. Medium Building ($4.0 - 12.0\text{m}$)
3. Large Building ($\ge 12.0\text{m}$)
4. Blurry Boundary Baseline Case
5. Complex Roof Geometry (Gable/Multi-tier)
6. High Roof Texture (Solar Panels / Skylights)
7. Adjacent Buildings ($< 2.0\text{m}$ Gap)
8. Vegetation Canopy Adjacency
9. Road Paint / Shadow Transition
10. Known Baseline Failure Case (`FALSE_SHORT` / `FALSE_LONG`)

---

## 6. Immutable Files & Rollback Protocol

The following M4 core files remain **100% IMMUTABLE**:
- `shadow/m4_physical_raycast_experiment.py`
- `shadow/geometry.py`
- `shadow/confidence.py`
- `shadow/height.py`

All diagnostic work is isolated to `tmp/analyze_guided_filter_depth.py` and output reports under `output/`.
