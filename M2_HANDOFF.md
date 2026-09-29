# DepthWizard M2 — Elevation Estimation & Calibration Engine Handoff

## 1. Executive Summary & Module Scope

Module **M2** is the core elevation estimation algorithm of the **DepthWizard (SIH26175)** system. Its primary responsibility is transforming single-view optical RGB remote sensing imagery into surface elevation representations adhering strictly to official problem statement semantics:

1. **Non-Georeferenced Mode (PNG / JPG)**:
   - Output: Relative Digital Surface Model ($\text{rDSM} \in [0, 1]$).
   - $\text{mode} = \text{"relative"}$, $\text{calibrated} = \text{False}$, $\text{height\_unit} = \text{"rel"}$.
   - $\text{CRS} = \text{None}$, $\text{transform} = \text{None}$, $\text{GSD} = \text{None}$.
   - **Zero fabricated metric elevations**.

2. **Georeferenced Mode (GeoTIFF)**:
   - Preserves spatial reference, CRS, and Affine transform.
   - **Uncalibrated**: Returns scale-agnostic $\text{rDSM} \in [0, 1]$ with $\text{height\_unit} = \text{"rel"}$ and $\text{calibrated} = \text{False}$.
   - **Calibrated (User GCPs or DEM/SRTM)**: Solves the linear transformation $H(x,y) = a \cdot D(x,y) + b$ to produce true metric elevation in metres ($\text{height\_unit} = \text{"m"}$, $\text{mode} = \text{"absolute"}$, $\text{calibrated} = \text{True}$).

---

## 2. Depth Sign Convention & Mathematical Orientation

### Depth Anything V2 Output Convention
Monocular depth estimation backbones (specifically *Depth Anything V2*) predict affine-invariant relative disparity/depth representations $D(x, y)$. 

- In satellite and aerial nadir imagery, elevated structures (e.g. building roofs, tree canopies) are physically closer to the sensor and exhibit distinct disparity signatures.
- When fitting against ground truth elevation anchors where $H_{\text{roof}} > H_{\text{ground}}$, the least-squares scale parameter $a$ adjusts naturally according to the relative disparity gradient.
- **Scientific Rule**: DepthWizard **never** artificially forces $a > 0$ or $a < 0$. The sign and magnitude of $a$ and $b$ are determined strictly by linear least-squares regression against verified metric ground elevation points:

$$H(x, y) = a \cdot D(x, y) + b$$

---

## 3. Calibration Mathematical Formulations

### A. Standard Ordinary Least Squares (OLS) GCP Calibration
Given $K \ge 2$ user-supplied ground control points $(D_i, H_i)$ where $D_i = D(y_i, x_i)$ is sampled from the monocular raster and $H_i$ is the known physical elevation in metres:

$$\begin{bmatrix} a \\ b \end{bmatrix} = \left( \mathbf{A}^T \mathbf{A} \right)^{-1} \mathbf{A}^T \mathbf{H}$$

where:

$$\mathbf{A} = \begin{bmatrix} D_1 & 1 \\ D_2 & 1 \\ \vdots & \vdots \\ D_K & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} H_1 \\ H_2 \\ \vdots \\ H_K \end{bmatrix}$$

### B. Regularized / Prior-Guided Calibration
When an explicit scale prior $a_{\text{prior}}$ is provided with regularization strength $\lambda > 0$, the objective function is:

$$\min_{a, b} \sum_{i=1}^K \left( a D_i + b - H_i \right)^2 + \lambda \left( a - a_{\text{prior}} \right)^2$$

Setting partial derivatives $\frac{\partial E}{\partial a} = 0$ and $\frac{\partial E}{\partial b} = 0$ yields the $2 \times 2$ normal system:

$$\begin{bmatrix} \sum_{i=1}^K D_i^2 + \lambda & \sum_{i=1}^K D_i \\ \sum_{i=1}^K D_i & K \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} \sum_{i=1}^K D_i H_i + \lambda a_{\text{prior}} \\ \sum_{i=1}^K H_i \end{bmatrix}$$

> **Default in Production**: $a_{\text{prior}} = \text{None}$, $\lambda_{\text{prior}} = 0.0$ (standard OLS, no hardcoded dataset priors).

### C. Residual Metrics Calculation
For each GCP $i \in \{1, \dots, K\}$:

$$\text{residual}_i = (a D_i + b) - H_i$$

$$\text{GCP\_MAE} = \frac{1}{K} \sum_{i=1}^K \left| \text{residual}_i \right|, \quad \text{GCP\_RMSE} = \sqrt{\frac{1}{K} \sum_{i=1}^K \text{residual}_i^2}$$

---

## 4. GCP Validation & Rejection Pipeline

Before solving the calibration system, `fit_supplied_gcps()` enforces strict validation checks:
1. **Raster Bounds**: Coordinates $(x, y)$ must fall within $[0, W) \times [0, H)$. Geographic/projected coordinates are converted via inverse affine transform $\sim\text{transform} \cdot (X, Y)$. Out-of-bounds coordinates are rejected.
2. **Collocation / Duplicates**: Duplicate coordinates mapping to the exact same pixel $(r, c)$ are rejected.
3. **Finite Elevations**: Non-finite (`NaN`, `Inf`) elevation inputs are rejected.
4. **Minimum Control Count**: Requires at least $K \ge 2$ valid GCPs.
5. **Depth Variation**: If $\max(D_{\text{gcp}}) - \min(D_{\text{gcp}}) < 10^{-7}$, the configuration is rejected as degenerate to prevent rank-deficient inversion.

---

## 5. DEM Calibration & Terrain Anchoring

Coarse DEMs (e.g. SRTM GeoTIFFs, $30\,\text{m}$ GSD) represent broad regional topography but lack building-level resolution.

### Terrain Datum Decoupling:
1. **Bilinear Reprojection**: Coarse DEM is warped to the target image grid using `rasterio.warp.reproject(Resampling.bilinear)`.
2. **Terrain Candidate Selection**: Pixels in the bottom percentile (`terrain_percentile`, default $25\%$) of the relative depth distribution are identified as open ground / terrain candidates:
   $$\text{TerrainMask} = \left\{ (x, y) \mid D(x, y) \le \text{percentile}\left( D, 25.0 \right) \right\}$$
3. **Datum Offset Estimation**: The median DEM elevation over the terrain mask defines the terrain datum:
   $$H_{\text{terrain\_anchor}} = \text{median}\left( \text{DEM}_{\text{reprojected}}[\text{TerrainMask}] \right)$$
4. **Detail Preservation**: Monocular high-frequency surface detail (rooftops, trees, parapets) is preserved and mapped onto the metric datum without being smoothed or overwritten by the coarse DEM.

---

## 6. Scientific Integrity Boundaries

- **ISPRS Potsdam / Benchmark LiDAR DSMs**: Reference LiDAR DSMs are strictly reserved for post-inference benchmarking, error analysis, and validation experiments (`scripts/error_analysis.py`). They are **never** imported into the production calibration path.
- **Zero Hallucination**: If no GCPs or DEM are provided, the system refuses to output metric elevation, preserving true relative semantics ($\text{rDSM} \in [0, 1]$).

---

## 7. Python API Contract

### Function: `process_image`
```python
from process_image import process_image

result = process_image(
    path="data/potsdam_sample_1024.tif",
    gcps=[
        {"x": 100, "y": 100, "elevation": 44.52},
        {"x": 800, "y": 150, "elevation": 45.10},
        {"x": 512, "y": 512, "elevation": 58.30}
    ],
    dem_path=None,               # Optional path to SRTM / DEM GeoTIFF
    use_shadows=True,           # Optional M4 shadow integration
    a_prior=None,               # Optional scale prior
    lambda_prior=0.0,           # Optional prior regularization weight
    terrain_percentile=25.0     # Optional terrain anchor percentile
)
```

### Result Schema:
```python
{
    "height_map": np.ndarray,      # float32 (H, W) in metres (or [0, 1] if uncalibrated)
    "width": int,
    "height": int,
    "mode": "absolute" | "relative",
    "calibrated": True | False,
    "georeferenced": True | False,
    "height_unit": "m" | "rel",
    "rgb": np.ndarray,             # uint8 (H, W, 3)
    "slope_map": np.ndarray,       # float32 (H, W) in degrees [0, 90]
    "confidence_map": np.ndarray,  # float32 (H, W) in [0, 1]
    "crs": CRS | None,
    "transform": Affine | None,
    "metadata": {
        "calibration": {
            "method": "gcp" | "dem" | "none",
            "gcp_count": int,
            "dem_source": str | None,
            "scale_a": float | None,
            "offset_b": float | None,
            "gcp_mae": float | None,
            "gcp_rmse": float | None,
            "gcp_residuals": list[float] | None,
            "terrain_anchor_count": int | None,
            "terrain_anchor_elevation": float | None
        },
        "shadow_mode": "metric" | "structural" | "disabled",
        "shadow_constraints": dict | None,
        "m4_active": bool,
        ...
    }
}
```

---

## 8. Verified Test Matrix

All test suites pass with 100% compliance:
- **`scripts/test_m2_calibration.py`**: Tests A through G (PNG uncalibrated, GeoTIFF uncalibrated, 5 GCPs, Exact Synthetic Least-Squares, GCP Rejection Rules, DEM Terrain Anchoring, Zero Silent Fallback).
- **`scripts/test_process_image.py`**: Full 8-stage official problem statement compliance test suite.
- **`scripts/test_m6_full_features.py`**: Full end-to-end browser and API acceptance suite.
