# Technical Implementation Plan: M7 Guided Filter Depth Refinement Integration

## Executive Summary

- **Plan Status**: **`M7 PLAN STATUS: READY FOR IMPLEMENTATION`**
- **Core Goal**: Safely integrate OpenCV Guided Filtering (`shadow/guided_filter.py`) upstream of the frozen M4 physical raycasting pipeline to refine Depth Anything V2 monocular depth boundaries and measure true end-to-end building height estimation improvements across all 1,760 ISPRS Potsdam GT buildings.
- **CRITICAL IMMUTABILITY GUARANTEE**: The baseline M4 production raycasting pipeline (`shadow/m4_physical_raycast_experiment.py`, `shadow/geometry.py`, `shadow/confidence.py`, `shadow/height.py`) remains **100% FROZEN & UNTOUCHED**.

---

## 1. Repository Audit & Current Architecture Findings

### A. Data Representation & Flow
- **Input Rasters**: Co-registered ISPRS Potsdam orthophoto ($6000 \times 6000 \times 3$, BGR `uint8`) and Depth Anything V2 relative disparity map $D_{\text{raw}}$ ($6000 \times 6000$, `float32` in range $[0.0, 1.0]$).
- **Physical Resolution**: Ground Sample Distance GSD $= 0.05\text{ m/px}$.
- **Downstream M4 Integration Point**:
  - Raw monocular depth $D_{\text{raw}}$ exhibits spatial upsampling blur along building footprints.
  - Guided Filter refines $D_{\text{raw}} \rightarrow D_{\text{filtered}}$ using BGR orthophoto guidance $I_{\text{RGB}}$.
  - Contour extraction on $D_{\text{filtered}}$ yields boundary points $P_0(x_0, y_0)$ aligned with physical walls.
  - Base points $P_0(x_0, y_0)$ enter `measure_building_shadow_m4_physical` in `shadow/m4_physical_raycast_experiment.py` for outward 1D physical shadow-tip raycasting.

### B. Correction of Proxy Height Calculation
- **Previous Audit Discovery**: Initial diagnostic script `tmp/analyze_guided_filter_depth.py` evaluated height error using a 90th-percentile step-ratio proxy (`h_pred_filt = m4_height_m * d_step_ratio`).
- **M7 End-to-End Requirement**: M7 passes actual filtered depth maps $D_{\text{filtered}}$ through the genuine downstream M4 physical raycaster (`shadow/m4_physical_raycast_experiment.py`) to measure true end-to-end height predictions without proxy multipliers.

---

## 2. Proposed Architecture & Insertion Point

```
ISPRS Potsdam Orthophoto (RGB 6000x6000x3)
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         │
Depth Anything V2                                  │
(Monocular Neural Disparity)                       │
         │                                         │
         ▼                                         │
Raw Depth D_raw (float32 [0,1])                    │
         │                                         │
         ├─────────────────────────────────────────┘
         ▼ (src = D_raw, guide = RGB)
[NEW M7 MODULE: shadow/guided_filter.py]
refine_depth_anything_map(RGB, D_raw, config)
         │
         ▼
Refined Depth D_filtered (float32 [0,1])
         │
         ▼
Building Contour Extraction (Sharp Footprints)
         │
         ▼
[IMMUTABLE PRODUCTION M4 RAYCASTER]
shadow/m4_physical_raycast_experiment.py
measure_building_shadow_m4_physical()
         │
         ▼
Actual Predicted Height (h_pred_m)
```

---

## 3. Comprehensive Technical Specification

### 1. New Module Architecture
- **New Code Module**: `shadow/guided_filter.py`
- **New Unit Test Module**: `shadow/test_guided_filter.py`
- **New Integration Validation Suite**: `shadow/run_m7_potsdam_validation.py`

### 2. Exact Function Interface
```python
def refine_depth_anything_map(
    guide_image: np.ndarray,
    raw_depth: np.ndarray,
    radius: int = 16,
    eps: float = 0.01,
    use_contrib_if_available: bool = True
) -> np.ndarray:
    """
    Refines Depth Anything V2 monocular disparity map using co-registered RGB guidance.
    
    Parameters:
        guide_image: (H, W, 3) BGR/RGB orthophoto (uint8 or float32 [0, 1]).
        raw_depth: (H, W) float32 relative disparity map in range [0.0, 1.0].
        radius: Spatial neighborhood kernel radius (r=16 px = 0.80m at 0.05m/px GSD).
        eps: Regularization parameter penalizing high local RGB variance.
        use_contrib_if_available: If True, uses cv2.ximgproc.guidedFilter when present.
        
    Returns:
        Refined disparity map q (H, W) float32 clipped to [0.0, 1.0].
    """
```

### 3. Input / Output Contracts
- **Input `guide_image`**: Must be 2D grayscale or 3D 3-channel image matching spatial dimensions $(H, W)$ of `raw_depth`.
- **Input `raw_depth`**: 2D `np.float32` matrix with non-negative relative disparity values.
- **Output Contract**: Returns 2D `np.float32` matrix with identical shape $(H, W)$, guaranteed finite (`np.isfinite`), and clipped to $[0.0, 1.0]$.

### 4. RGB / Depth Alignment Validation
Assert spatial shape equality:
```python
if guide_image.shape[:2] != raw_depth.shape[:2]:
    raise ValueError(f"Spatial dimension mismatch: guide {guide_image.shape[:2]} vs depth {raw_depth.shape[:2]}")
```

### 5. Data Type & Bounding Validation
- Cast `raw_depth` to `np.float32`.
- Clip filtered output $q$ to $[0.0, 1.0]$ via `np.clip(q, 0.0, 1.0)`.

### 6. NaN / Inf Safeguards
Sanitize inputs before filtering:
```python
raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=1.0, neginf=0.0)
```

### 7. OpenCV Contrib Implementation
If `hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")`, invoke C++ native implementation `cv2.ximgproc.guidedFilter`.

### 8. Pure OpenCV Fallback Engine
If `cv2.ximgproc` is missing, execute standalone `guided_filter_pure_cv2` using standard `cv2.boxFilter`.

### 9. Dependency Strategy
- Keep `opencv-python` as default.
- If `opencv-contrib-python` is added to `requirements.txt` in the future, `refine_depth_anything_map` automatically upgrades to `cv2.ximgproc.guidedFilter` with zero code changes.

### 10. Configuration Management
```python
@dataclass(frozen=True)
class GuidedFilterConfig:
    radius: int = 16          # 0.80m physical radius at 0.05m/px GSD
    eps: float = 0.01         # Regularization parameter
    enable_guided_filter: bool = True
```

### 11. Feature Flag & Rollback Strategy
- Setting `enable_guided_filter = False` bypasses `refine_depth_anything_map` and passes pristine raw depth directly into downstream contour extraction, providing instant zero-downtime rollback.

### 12. Ground-Truth Leakage Audit & Scanning
- Static grep checks verify that `shadow/guided_filter.py` contains zero references to `gt_height`, `dsm`, `ndsm`, or ground-truth labels.
- Ground-truth height is accessed **exclusively post-hoc** inside `shadow/run_m7_potsdam_validation.py` during validation score calculation.

### 13. Frozen-File Integrity Checks
SHA-256 hash & file size verification prior to and after test execution:
- `shadow/m4_physical_raycast_experiment.py`
- `shadow/geometry.py`
- `shadow/confidence.py`
- `shadow/height.py`

### 14. Unit Tests (`shadow/test_guided_filter.py`)
- Test 1: Shape and datatype preservation.
- Test 2: Bounded output $[0.0, 1.0]$ verification.
- Test 3: Synthetic step-edge sharpening verification.
- Test 4: Uniform intensity stability verification.
- Test 5: Fallback engine numerical equivalence check.

### 15. End-to-End Benchmark Execution (`shadow/run_m7_potsdam_validation.py`)
Run actual M4 physical raycasting on all 1,760 GT Potsdam buildings across 38 TOP RGB tiles, comparing baseline raw depth vs M7 guided-filtered depth.

### 16. Reproducibility Test
Verify 5 identical repeated runs yield deterministic variance $\sigma^2 = 0.0$.

### 17. Visual Diagnostic Generation
Generate 5-panel cropped diagnostic overlays under `output/m7_guided_filter_visuals/`.

### 18. Runtime & Memory Evaluation
- Peak Memory Footprint per Tile ($6000 \times 6000$ float32): $\approx 576\text{ MB}$.
- Execution Latency: $< 0.5\text{ seconds}$ per tile.

### 19. Files Modified vs Frozen
- **Immutable Frozen Files**:
  - `shadow/m4_physical_raycast_experiment.py`
  - `shadow/geometry.py`
  - `shadow/confidence.py`
  - `shadow/height.py`
- **New Modules To Be Created**:
  - `shadow/guided_filter.py`
  - `shadow/test_guided_filter.py`
  - `shadow/run_m7_potsdam_validation.py`

### 20. Explicit GO / NO-GO Acceptance Criteria
1. Full end-to-end Potsdam dataset MAE $\le 3.80\text{m}$ (Baseline: $4.69\text{m}$) OR statistically meaningful improvement without regression.
2. Flat-Roof Texture Transfer Ratio $R_{\text{TT}} \le 1.10$.
3. Category C degradation rate $< 2.0\%$.
4. Zero newly created false candidate categories (Categories E & F = 0).
5. All unit tests in `shadow/test_guided_filter.py` pass.
6. Frozen M4 files 100% untouched.
7. Zero ground-truth leakage.
8. Deterministic repeated execution ($\sigma^2 = 0.0$).
