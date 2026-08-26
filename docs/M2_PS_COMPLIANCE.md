# DepthWizard M2 & M5 — Problem Statement Compliance Reference

**Project**: SIH26175 — Monocular Satellite/Aerial Elevation Reconstruction  
**Module**: M2 (Elevation Engine & Depth Pipeline) & M5 (3D WebGL Visualization Interface)  
**Status**: Problem Statement Compliant, Fully Refactored, Verified & Frozen  

---

## 1. Problem Statement Executive Summary

The official SIH26175 problem statement requires transforming single-view optical RGB remote-sensing imagery into elevation representations under strict geospatial and scientific principles:

1. **Non-Georeferenced Imagery (PNG / JPG / standard photo)**:
   - Must produce a **Relative Digital Surface Model (rDSM)** normalized to [0, 1].
   - Must **never** fabricate ground sampling distance (GSD), geospatial coordinates, or Coordinate Reference Systems (CRS).
   - Must **never** report elevation values in metres (`"m"`).
   - UI must explicitly state: *"Relative Surface Height — Values are not metric elevations."*

2. **Georeferenced Imagery (GeoTIFF / Geospatial TIFF)**:
   - CRS and affine transformation matrices provide **spatial reference only**, not vertical elevation.
   - Without an explicit calibration source, the engine must return an **uncalibrated relative DSM** with `calibrated: False`, `method: "none"`, and elevation unit `'rel'`.
   - The engine must **never** invent scale/offset parameters or use hardcoded empirical priors.
   - When the user supplies genuine Ground Control Points (GCPs) or a lower-resolution DEM, the engine fits an Ordinary Least Squares (OLS) linear calibration model ($H = a \cdot D + b$) to produce a genuine **Metric Absolute DSM** in metres.

---

## 2. Core Python Backend Contract (`process_image()`)

### Function Signature
```python
def process_image(
    path: str,
    gcps: Optional[List[Dict[str, float]]] = None,
    dem_path: Optional[str] = None,
    dem_file: Optional[str] = None,
    use_shadows: bool = True,
    model=None,
    processor=None,
    device=None
) -> Dict[str, Any]:
```

### Return Data Model
```python
{
    "height_map": np.ndarray,      # 2D float32 array: metric DSM (m) if calibrated=True, else rDSM in [0, 1]
    "width": int,                  # Original raster pixel width
    "height": int,                 # Original raster pixel height
    "mode": "relative" | "absolute", # "relative" (PNG/JPG) or "absolute" (GeoTIFF)
    "calibrated": bool,            # True ONLY if user supplied valid GCPs or DEM
    "rgb": np.ndarray,             # Original RGB array (H, W, 3) uint8
    "slope_map": np.ndarray,       # 2D float32 slope array: physical degrees if calibrated, else geometric slope
    "confidence_map": np.ndarray,  # 2D float32 confidence array in [0, 1]
    "crs": Optional[str],          # True CRS (e.g., "EPSG:32633") or None for non-georeferenced
    "transform": Optional[Affine], # True Affine transform or None for non-georeferenced
    "metadata": {
        "input_path": str,
        "input_format": str,
        "mode": "relative" | "absolute",
        "calibrated": bool,
        "model": "Depth Anything V2",
        "calibration": {
            "method": "none" | "gcp" | "dem",
            "gcp_count": int,
            "dem_source": Optional[str],
            "scale_a": Optional[float],
            "offset_b": Optional[float]
        },
        "width": int,
        "height": int,
        "crs": Optional[str],
        "gsd_x": Optional[float],
        "gsd_y": Optional[float]
    }
}
```

---

## 3. Operational Modes & Calibration Pathways

```
                      ┌────────────────────────────────────────┐
                      │          Input Image Raster            │
                      └──────────────────┬─────────────────────┘
                                         │
                         Is Geospatial Metadata Present?
                                         │
                    ┌────────────────────┴────────────────────┐
                   YES                                        NO
                    │                                          │
        ┌───────────▼────────────┐                ┌────────────▼───────────┐
        │  Georeferenced GeoTIFF │                │   PNG / JPG Photo      │
        │  mode = 'absolute'     │                │   mode = 'relative'    │
        └───────────┬────────────┘                └────────────┬───────────┘
                    │                                          │
        Did user provide GCPs or DEM?                          │
                    │                                          │
          ┌─────────┴─────────┐                                │
         YES                  NO                               │
          │                    │                               │
┌─────────▼────────┐  ┌────────▼─────────┐                     │
│ Metric DSM (m)   │  │ Uncalibrated     │                     │
│ calibrated=True  │  │ rDSM [0, 1]      │                     │
│ unit = 'm'       │  │ calibrated=False │                     │
│ Badge: "Metric   │  │ unit = 'rel'     │                     │
│ Absolute DSM"    │  │ Badge: "Metric   │                     │
│                  │  │ Calibration      │                     │
│                  │  │ Required"        │                     │
└──────────────────┘  └──────────────────┘                     │
                                                               │
                                                  ┌────────────▼───────────┐
                                                  │ Relative rDSM [0, 1]   │
                                                  │ calibrated=False       │
                                                  │ unit = 'rel'           │
                                                  │ CRS = None, GSD = None │
                                                  │ Badge: "Relative rDSM" │
                                                  └────────────────────────┘
```

---

## 4. Calibration Mathematics

### A. Ground Control Points (GCP)
When the user provides $K \ge 2$ GCPs with pixel coordinates $(x_k, y_k)$ and known metric elevations $Z_k$:
$$\min_{a, b} \sum_{k=1}^K \left( a \cdot D(x_k, y_k) + b - Z_k \right)^2$$
The solved parameters $a > 0$ and $b$ are applied across all pixels:
$$H(x, y) = a \cdot D(x, y) + b$$

### B. Digital Elevation Model (DEM)
When the user supplies a coarse DEM (e.g., SRTM / Copernicus DEM GeoTIFF):
1. The DEM is reprojected onto the target raster grid using `rasterio.reproject` with bilinear interpolation.
2. Low-slope terrain regions (ground pixels) are selected to avoid building occlusion errors.
3. Ordinary Least Squares regression solves for datum anchor $b$ and scale $a$.

### C. Uncalibrated Fallback
When neither GCPs nor a DEM is provided, the relative depth map $D(x, y)$ is min-max normalized to $[0.0, 1.0]$:
$$rDSM(x, y) = \frac{D(x, y) - \min(D)}{\max(D) - \min(D)}$$
No elevation in metres is exposed, and `calibrated` remains `False`.

---

## 5. M5 3D WebGL Viewer Three-State Machine

| State | Input Type | Calibrated | Mode Badge | Description Text | Inspector Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Relative Mode** | PNG / JPG | `False` | `Relative rDSM Mode` | `Relative Surface Height — Values are not metric elevations.` | `0.xxx (rel)` |
| **2. Uncalibrated Mode** | GeoTIFF (No GCP/DEM) | `False` | `Metric Calibration Required` | `Relative Surface Height — metric calibration unavailable` | `0.xxx (uncalibrated)` |
| **3. Metric Absolute Mode**| GeoTIFF + GCPs/DEM | `True` | `Metric Absolute DSM` | `Metric Surface Elevation (m)` | `XX.XX m` |

---

## 6. Verification and Test Results

### Automated Test Suites
1. **`scripts/test_process_image.py`** (7 compliance tests):
   - **Test 1**: PNG $\to$ `mode='relative'`, `calibrated=False`, `CRS=None`, `GSD=None`, span $[0.00, 1.00]$ `[PASS]`
   - **Test 2**: JPG $\to$ `mode='relative'`, `calibrated=False`, `CRS=None`, span $[0.00, 1.00]$ `[PASS]`
   - **Test 3**: Uncalibrated GeoTIFF $\to$ `mode='absolute'`, `calibrated=False`, `scale_a=None`, span $[0.00, 1.00]$ `[PASS]`
   - **Test 4**: GCP-Calibrated GeoTIFF $\to$ `mode='absolute'`, `calibrated=True`, $a=14.85$, $b=29.63\text{m}$, span $[39.24\text{m}, 65.00\text{m}]$ `[PASS]`
   - **Test 5**: DEM-Calibrated GeoTIFF $\to$ `mode='absolute'`, `calibrated=True`, span $[43.75\text{m}, 46.02\text{m}]$ `[PASS]`
   - **Test 6**: Slope Computation $\to$ metric slope $[0.0^\circ, 87.4^\circ]$, relative slope $[0.0^\circ, 13.0^\circ]$ `[PASS]`
   - **Test 7**: Raster Export $\to$ GeoTIFF with intact EPSG:32633 CRS & PNG rDSM `[PASS]`

2. **`scripts/test_api_server.py`** (4 endpoint integration tests):
   - `GET /` and `GET /health` online `[PASS]`
   - `POST /process` with uncalibrated GeoTIFF $\to$ `height_unit: "rel"`, `calibrated: False` `[PASS]`
   - `POST /process` with 5 GCPs $\to$ `height_unit: "m"`, `calibrated: True` `[PASS]`
   - `POST /process` with PNG photo $\to$ `mode: "relative"`, `crs: None`, `height_unit: "rel"` `[PASS]`

3. **`scripts/e2e_browser_qa.py`** (Automated Playwright Chromium validation):
   - Verified 3D WebGL rendering in headless browser.
   - Tested real GeoTIFF and PNG image uploads.
   - Verified DOM badges, metadata panels, and raycast inspector.
   - Verified 0 browser console JavaScript errors.
   - Saved unified multi-panel validation screenshot: `outputs/figures/m5_e2e_qa.png`.

---

## 7. Scope Boundaries & Preservation

- **M2 & M5**: Complete, fully aligned with the official problem statement, tested, and frozen.
- **M1, M3, M4, M6**: Strictly **NOT** implemented or modified, preserving clean modular separation for subsequent milestones.
