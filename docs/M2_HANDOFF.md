# DepthWizard M2 — Elevation Engine Handoff & Technical Reference

**Project**: SIH26175 — Single-View Optical Remote-Sensing Elevation Estimation  
**Module**: M2 (Core Algorithm / Elevation Estimation Backend)  
**Author**: M2 Engine Team  
**Status**: Problem-Statement Compliant, Fully Verified, Tested & Integration-Ready  

---

## 1. M2 Purpose & System Role

Module M2 provides the core elevation-estimation algorithm and processing engine for the DepthWizard system. It transforms single-view optical RGB satellite and aerial imagery into surface elevation models using a pretrained monocular depth backbone (**Depth Anything V2**) coupled with rigorous geospatial calibration.

M2 serves as the standalone, stateless computation backbone:
- **Inputs received from**:
  - M1: Image preprocessing, contrast enhancement, or external refined depth (`external_depth`).
  - M3: External geospatial DEM datums (`dem_path` / `dem_file`) or Ground Control Points (`gcps`).
  - M4: Validated solar/shadow geometric constraints (`shadow_constraints`).
- **Outputs provided to**:
  - M5: 3D WebGL terrain viewer (downsampled height arrays, vertex colors, slope maps, confidence metrics).
  - M6: Geospatial export pipelines (GeoTIFF rasters, shapefile extraction, analytical reporting).

---

## 2. Processing Pipeline

```
                     Input Raster (TIFF / PNG / JPG)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
         Image Ingestion                   Geospatial Metadata
      (RGB uint8 extraction)             (CRS, Affine Transform, GSD)
                 │                                 │
                 ▼                                 │
        Depth Anything V2                          │
    (Monocular Disparity/Depth)                    │
                 │                                 │
                 ▼                                 │
         Relative Depth D                          │
                 │                                 │
       Has Valid Calibration?                      │
                 │                                 │
      ┌──────────┴──────────┐                      │
     YES                    NO                     │
      │                      │                     │
┌─────▼─────────────┐ ┌──────▼─────────────┐       │
│ Metric Calibration│ │ Relative Min-Max   │       │
│ H = a*D + b (OLS) │ │ rDSM in [0.0, 1.0] │       │
└─────┬─────────────┘ └──────┬─────────────┘       │
      │                      │                     │
      ▼                      ▼                     │
  Metric DSM (m)        Relative rDSM              │
      │                      │                     │
      └──────────┬───────────┘                     │
                 ▼                                 │
        Slope & Confidence ◄───────────────────────┘
     (Physical / Geometric)
                 │
                 ▼
       Standard Return Object
```

---

## 3. Supported Input Formats

1. **Non-georeferenced Imagery (`.png`, `.jpg`, `.jpeg`, standard `.tif`)**:
   - Ordinary aerial or satellite photos without embedded projection/geotransform metadata.
2. **Georeferenced Imagery (`.tif`, `.tiff` GeoTIFF)**:
   - Contains a valid Coordinate Reference System (e.g., `EPSG:32633`) and non-identity affine transform.

---

## 4. Operational Modes

### Mode A: Non-Georeferenced Image (PNG/JPG)
- `mode = "relative"`, `calibrated = False`, `georeferenced = False`, `height_unit = "rel"`
- Produces a **Relative Digital Surface Model (rDSM)** normalized to $[0.0, 1.0]$.
- `crs = None`, `transform = None`, `gsd_x = None`, `gsd_y = None`.
- **Constraint**: Values are never presented as metric elevations; no fake GSD or CRS is fabricated.

### Mode B: Georeferenced Image WITHOUT Elevation Calibration (GeoTIFF Fallback)
- `mode = "relative"`, `calibrated = False`, `georeferenced = True`, `height_unit = "rel"`
- Preserves the true CRS and affine transform for 2D spatial referencing.
- Height values are returned as a normalized relative surface in $[0.0, 1.0]$.
- `calibration["method"] = "none"`, `scale_a = None`, `offset_b = None`.
- **Constraint**: Never invents scale/offset; CRS alone does NOT provide elevation.

### Mode C: Georeferenced Image WITH GCP Calibration
- `mode = "absolute"`, `calibrated = True`, `georeferenced = True`, `height_unit = "m"`
- Fits $H(x, y) = a \cdot D(x, y) + b$ using user-supplied Ground Control Points.
- Height values represent absolute metric surface elevations in metres.

### Mode D: Georeferenced Image WITH DEM/SRTM Calibration
- `mode = "absolute"`, `calibrated = True`, `georeferenced = True`, `height_unit = "m"`
- Aligns lower-resolution reference DEM with the target raster and anchors ground elevation.

---

## 5. Calibration Mathematics

### A. Ground Control Points (GCP)
Given $K \ge 2$ GCPs with pixel coordinates $(x_k, y_k)$ and known ground-truth elevations $Z_k$:
$$\min_{a, b} \sum_{k=1}^K \left( a \cdot D(x_k, y_k) + b - Z_k \right)^2$$
Using Ordinary Least Squares (OLS):
$$a = \frac{\sum (D_k - \bar{D})(Z_k - \bar{Z})}{\sum (D_k - \bar{D})^2}, \quad b = \bar{Z} - a \bar{D}$$
Applied globally:
$$H(x, y) = a \cdot D(x, y) + b$$

### B. DEM / SRTM Terrain Anchoring
1. The user-supplied DEM raster is dynamically reprojected to the target scene using `rasterio.reproject` with bilinear interpolation.
2. Ground-level terrain pixels are sampled, filtering out high-slope occlusion regions.
3. OLS linear regression solves for the vertical datum offset $b$ and scale $a$.

---

## 6. GCP & DEM Input Formats

### GCP Format (List of Dictionaries)
```python
gcps = [
    {"x": 100.0, "y": 150.0, "elevation": 45.20},  # Pixel coordinates & elevation in metres
    {"x": 450.0, "y": 300.0, "elevation": 58.40},
    {"x": 800.0, "y": 750.0, "elevation": 43.10}
]
```
*Note: If spatial coordinates (easting/northing) are passed in `x` and `y`, M2 automatically maps them to pixel indices using the inverse affine transform.*

### DEM Format
Filepath string to a valid raster (e.g., `dem_path="data/dem/srtm_potsdam.tif"` or `dem_file=...`).

---

## 7. Returned Dictionary Contract (`process_image()`)

```python
result = process_image(path="input.tif", ...)
```

| Key | Type | Description |
| :--- | :--- | :--- |
| `height_map` | `np.ndarray` (float32 [H, W]) | Metric DSM in metres if `calibrated=True`, else rDSM in $[0.0, 1.0]$ |
| `width` | `int` | Original image pixel width |
| `height` | `int` | Original image pixel height |
| `mode` | `str` | `"relative"` or `"absolute"` |
| `calibrated` | `bool` | `True` only if genuine GCP or DEM calibration was supplied |
| `georeferenced`| `bool` | `True` if image has valid CRS and affine transform |
| `height_unit` | `str` | `"m"` (metres) if calibrated, `"rel"` if relative |
| `rgb` | `np.ndarray` (uint8 [H, W, 3]) | Original RGB image array |
| `slope_map` | `np.ndarray` (float32 [H, W])| Surface slope angle in degrees $[0.0, 90.0]$ |
| `confidence_map`| `np.ndarray` (float32 [H, W])| Reconstruction confidence in $[0.0, 1.0]$ |
| `crs` | `rasterio.crs.CRS` or `None` | Coordinate Reference System |
| `transform` | `rasterio.Affine` or `None` | Affine geotransform matrix |
| `metadata` | `dict` | Diagnostic metadata (GSD, calibration parameters, raw depth stats) |

### Metadata Dictionary Structure
```python
{
    "input_path": str,
    "input_format": "tif" | "png" | "jpg",
    "mode": "relative" | "absolute",
    "calibrated": bool,
    "georeferenced": bool,
    "height_unit": "rel" | "m",
    "model": "Depth Anything V2",
    "calibration": {
        "method": "none" | "gcp" | "dem",
        "gcp_count": int,
        "dem_source": Optional[str],
        "scale_a": Optional[float],
        "offset_b": Optional[float]
    },
    "shadow_mode": "metric" | "structural" | "disabled",
    "shadow_constraints": Optional[dict],
    "width": int,
    "height": int,
    "crs": Optional[str],
    "gsd_x": Optional[float],
    "gsd_y": Optional[float],
    "raw_depth_stats": {
        "min": float,
        "max": float,
        "span": float
    }
}
```

---

## 8. Slope & Confidence Computation

### Slope Angle
- Computed via 2D spatial gradients:
  $$\theta(x, y) = \arctan\left( \sqrt{\left(\frac{\partial H}{\partial x}\right)^2 + \left(\frac{\partial H}{\partial y}\right)^2} \right) \times \frac{180}{\pi}$$
- **Metric Mode**: Uses physical Ground Sampling Distance (`gsd_x`, `gsd_y` in metres/pixel) $\to$ True physical slope in degrees $[0^\circ, 90^\circ]$.
- **Relative Mode**: Uses unit pixel spacing $\to$ Geometric surface slope in degrees $[0^\circ, 90^\circ]$.

### Heuristic Confidence
Reconstruction confidence $C(x, y) \in [0.0, 1.0]$ combines:
1. **Boundary attenuation**: Reduces confidence within 15 pixels of the image edge.
2. **Shadow penalty**: Lowers confidence in deep occluded shadow regions.
3. **Disparity gradient penalty**: Penalizes high-frequency noise spikes.

---

## 9. Geospatial & Non-Geospatial Export

### Function: `export_dsm(result, output_path)`
- **Calibrated GeoTIFF**: Writes single-band float32 GeoTIFF with original CRS, affine transform, and NoData value `-9999.0`.
- **Non-Georeferenced / rDSM**: Writes standard 8-bit grayscale PNG or unprojected float32 TIFF.

### Function: `export_slope(result, output_path)`
- Exports single-band slope raster in degrees $[0.0, 90.0]$.

---

## 10. Evaluation & Error Metrics

M2 supports standard geospatial verification metrics:
- **MAE** (Mean Absolute Error): $\frac{1}{N} \sum |H_{pred} - H_{ref}|$
- **RMSE** (Root Mean Square Error): $\sqrt{\frac{1}{N} \sum (H_{pred} - H_{ref})^2}$
- **Pearson Correlation ($r$)**: Linear correlation between predicted and LiDAR reference elevations.
- **$R^2$ Score**: Coefficient of determination.

*Note: In-sample calibration (evaluating on the same tile used to fit $a, b$) must always be reported separately from held-out spatial validation.*

---

## 11. Known Physical Limitations

1. **Monocular Depth Dynamic Compression**: Pretrained vision transformers compress relative disparity for very tall isolated structures (e.g., high-rise buildings).
2. **Roof Surface Flatness**: Monocular depth models may introduce minor dome or bowl curvatures on large flat rooftops.
3. **SRTM Vertical Resolution**: SRTM (30m GSD) represents coarse terrain elevation, not building-level roof structures. It anchors ground datum $b$, but does not replace GCPs for vertical building height calibration.

---

## 12. Integration Guide for M5 (3D Visualization) & M6 (Application)

### Quick Start Example
```python
from process_image import process_image, export_dsm

# Example 1: Non-georeferenced photo
res_photo = process_image("data/aerial_photo.png")
print(res_photo["mode"])         # "relative"
print(res_photo["height_unit"])  # "rel"

# Example 2: Georeferenced GeoTIFF with GCPs
user_gcps = [
    {"x": 120, "y": 250, "elevation": 44.5},
    {"x": 800, "y": 600, "elevation": 52.0}
]
res_metric = process_image("data/scene.tif", gcps=user_gcps)
print(res_metric["mode"])        # "absolute"
print(res_metric["height_unit"]) # "m"

# Export GeoTIFF for GIS (QGIS, ArcGIS)
export_dsm(res_metric, "outputs/final_metric_dsm.tif")
```

### M5 Integration Best Practices
- Read `result["height_map"]` (2D float32 array) to build Three.js plane vertex displacements.
- Read `result["height_unit"]` to format UI labels:
  - If `height_unit == "m"`: format as `"XX.XX m"`
  - If `height_unit == "rel"`: format as `"0.xxx (rel)"`
- Check `result["calibrated"]` and `result["georeferenced"]` to set UI mode badges.
- Use `result["slope_map"]` and `result["confidence_map"]` for vertex color shaders.
