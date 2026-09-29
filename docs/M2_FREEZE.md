# DepthWizard M2 — Official Freeze Specification

**Status:** FROZEN  
**Module:** M2 (Elevation Estimation & Metric Calibration Engine)  
**Project:** SIH26175 — DepthWizard  
**Freeze Date:** August 26, 2026  

---

## 1. Frozen API Contract

Module M2 provides the canonical elevation inference entry point:

```python
from process_image import process_image

result = process_image(
    path: str,
    gcps: Optional[List[Dict[str, float]]] = None,
    dem_path: Optional[str] = None,
    dem_file: Optional[str] = None,
    use_shadows: bool = True,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    device: Optional[Any] = None,
    refinement: Optional[Any] = None,
    external_depth: Optional[np.ndarray] = None,
    shadow_constraints: Optional[Dict[str, Any]] = None,
    a_prior: Optional[float] = None,
    lambda_prior: float = 0.0,
    terrain_percentile: float = 25.0
) -> Dict[str, Any]
```

### Return Dictionary Specification

| Field | Type | Description |
| :--- | :--- | :--- |
| `height_map` | `np.ndarray` (`float32`, shape `(H, W)`) | Metric elevation in metres if calibrated, or scale-agnostic relative rDSM $\in [0, 1]$ if uncalibrated. |
| `width` | `int` | Raster width in pixels ($W$). |
| `height` | `int` | Raster height in pixels ($H$). |
| `mode` | `str` (`"relative"` \| `"absolute"`) | `"absolute"` when valid calibration (GCP/DEM) is applied; `"relative"` otherwise. |
| `calibrated` | `bool` | `True` strictly when genuine GCP or DEM calibration was executed; `False` otherwise. |
| `georeferenced` | `bool` | `True` when input has valid CRS and non-identity affine transform; `False` otherwise. |
| `height_unit` | `str` (`"m"` \| `"rel"`) | `"m"` strictly when `calibrated=True`; `"rel"` for all uncalibrated rasters. |
| `rgb` | `np.ndarray` (`uint8`, shape `(H, W, 3)`) | Standard 3-channel 8-bit RGB image buffer. |
| `slope_map` | `np.ndarray` (`float32`, shape `(H, W)`) | Surface slope angle in physical degrees $[0, 90]^\circ$. |
| `confidence_map` | `np.ndarray` (`float32`, shape `(H, W)`) | Per-pixel reconstruction confidence score $\in [0, 1]$. |
| `crs` | `rasterio.crs.CRS` \| `None` | Coordinate Reference System (e.g. `EPSG:32633`) or `None` for non-geospatial images. |
| `transform` | `rasterio.transform.Affine` \| `None` | Affine transformation mapping pixel coordinates to geographic coordinates. |
| `metadata` | `dict` | Comprehensive metadata including calibration parameters, residuals, GSD, and shadow analysis. |

---

## 2. Four Frozen Calibration Modes

### Mode 1: Non-Georeferenced Image (PNG / JPG)
- **Inputs**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Output**: Relative Digital Surface Model ($\text{rDSM} \in [0, 1]$)
- **Contracts**:
  - `mode = "relative"`
  - `calibrated = False`
  - `georeferenced = False`
  - `height_unit = "rel"`
  - `crs = None`, `transform = None`, `gsd_x = None`, `gsd_y = None`
  - **Zero fabricated metric elevation values**.

### Mode 2: Uncalibrated Georeferenced Image (GeoTIFF)
- **Inputs**: `.tif`, `.tiff` with spatial metadata, no GCPs, no DEM.
- **Output**: Normalized relative surface representation ($\text{rDSM} \in [0, 1]$)
- **Contracts**:
  - `mode = "relative"`
  - `calibrated = False`
  - `georeferenced = True`
  - `height_unit = "rel"`
  - `crs` and `transform` are fully preserved.
  - **Zero hardcoded elevation priors**.

### Mode 3: Ground Control Point (GCP) Calibrated GeoTIFF
- **Inputs**: GeoTIFF + $K \ge 2$ valid Ground Control Points `[{"x": px_x, "y": px_y, "elevation": z_m}, ...]`.
- **Output**: Metric Digital Surface Model ($H(x, y) = a \cdot D(x, y) + b$)
- **Contracts**:
  - `mode = "absolute"`
  - `calibrated = True`
  - `georeferenced = True`
  - `height_unit = "m"`
  - Ordinary Least Squares (or regularized prior if $a_{\text{prior}}$ and $\lambda > 0$ are supplied).
  - Validation rejection on out-of-bounds, duplicate pixels, non-finite values, or degenerate depth distributions.

### Mode 4: DEM / SRTM Anchor Calibrated GeoTIFF
- **Inputs**: GeoTIFF + Coarse DEM GeoTIFF (`dem_path` or `dem_file`).
- **Output**: Metric Digital Surface Model ($H_{\text{pred}} = a \cdot D + b$)
- **Contracts**:
  - `mode = "absolute"`
  - `calibrated = True`
  - `georeferenced = True`
  - `height_unit = "m"`
  - DEM is reprojected via bilinear interpolation to match target grid.
  - Coarse terrain datum is extracted from candidate terrain pixels (`terrain_percentile`, default $25\%$) without overwriting high-frequency monocular building morphology.

---

## 3. Depth Sign Convention & Units

- **Monocular Relative Disparity**: Monocular depth backbones (*Depth Anything V2*) output relative disparity $D(x, y)$. Elevated structures (closer to sensor) have distinct disparity signatures.
- **Linear Transformation**: $H(x, y) = a \cdot D(x, y) + b$. The sign of $a$ is solved naturally via least-squares regression against physical ground truth anchors and is **never artificially forced positive or negative**.
- **Units**:
  - Uncalibrated: Unitless relative normalized scale $[0, 1]$ (`height_unit = "rel"`).
  - Calibrated: Physical elevation in metres above datum (`height_unit = "m"`).
  - Slope: Physical degrees $[0, 90]^\circ$.
  - Confidence: Unitless score $[0, 1]$.

---

## 4. Integration Hooks for Other Modules

- **M1 (Preprocessing / Super-Resolution / External Depth)**:
  - Can pass precomputed or enhanced depth via `external_depth: np.ndarray`. When present, M2 bypasses internal backbone inference and executes calibration on the supplied array.
- **M3 (Spatial Refinement / Edge Regularization)**:
  - Can pass configuration parameters via `refinement: Any`.
- **M4 (Shadow Cue Analysis)**:
  - Can pass structured shadow geometric constraints via `shadow_constraints: dict`.
  - M2 automatically calls `shadow.run_full_pipeline` from `shadow-cue/` when available.
- **M5 / M6 (Visualization & Dashboard Consoles)**:
  - M2 results serialize directly into standard JSON via `api_server.py` (`POST /process`), supporting multi-tab colormaps, 3D WebGL meshes, and 32-bit float GeoTIFF downloads (`GET /export/{session_id}`).

---

## 5. Scientific Boundaries & Non-Negotiable Rules

1. **Evaluation-Only Isolation**: Reference LiDAR DSMs (e.g. ISPRS Potsdam LiDAR) are strictly isolated for post-inference benchmarking and error analysis. They must **never** enter the runtime production calibration pipeline.
2. **Zero Magic Numbers**: Default values for calibration parameters in production are `a_prior = None` and `lambda_prior = 0.0`. No dataset-specific constants are hardcoded.

---

## 6. Known Limitations

1. **Monocular Ambiguity**: In the uncalibrated state, monocular disparity can only represent relative topological relief, not absolute metric heights.
2. **Coarse DEM Resolution**: SRTM DEMs ($30\,\text{m}$ GSD) provide broad terrain datum anchoring but cannot calibrate individual building roof heights without high-resolution GCPs.
3. **GCP Geometry**: At least 2 non-collocated GCPs spanning different elevations are required to solve scale $a$ and offset $b$.

---

## 7. Verified Regression Test Suites

All regression test suites pass with 100% success:

```bash
# 1. Dedicated Calibration Engine Test Suite (Tests A through G)
python scripts/test_m2_calibration.py

# 2. Official Problem Statement Compliance Suite (Tests 1 through 8)
python scripts/test_process_image.py

# 3. API Server Integration Suite (Tests 1 through 4)
python scripts/test_api_server.py

# 4. Final Integration Audit
python scripts/audit_m2_integration.py

# 5. Full Browser Acceptance Suite (Headless Chromium, 3D WebGL, 0 JS Errors)
python scripts/test_m6_full_features.py
```

---

## 8. Frozen Files (DO NOT MODIFY)

The following files constitute the frozen M2 core and must not be altered:

1. `process_image.py` — Core elevation estimation & calibration engine.
2. `depth/depth_model.py` — Depth Anything V2 monocular backbone loader.
3. `shadow_detection.py` — Shadow confidence and morphological filters.
4. `scripts/test_m2_calibration.py` — Calibration test suite.
5. `scripts/test_process_image.py` — Problem statement compliance test suite.
6. `scripts/audit_m2_integration.py` — Final integration audit suite.
7. `docs/M2_FREEZE.md` — This freeze specification.
