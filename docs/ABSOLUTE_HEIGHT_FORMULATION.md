# DepthWizard — Complete Mathematical & Physical Formulation Handbook

This document provides the exhaustive mathematical, geodetic, optical, and statistical formulations implemented across the **DepthWizard** 3D surface reconstruction and elevation extraction engine.

---

## Table of Contents
1. [Monocular Depth & Tiled Spatial Blending](#1-monocular-depth--tiled-spatial-blending)
2. [Relative Digital Surface Model Normalization (rDSM)](#2-relative-digital-surface-model-normalization-rdsm)
3. [Metric Elevation Calibration Engine (Absolute Height)](#3-metric-elevation-calibration-engine-absolute-height)
   - [3.1 Linear Ordinary Least Squares (OLS)](#31-linear-ordinary-least-squares-ols)
   - [3.2 Single-Point Anchor Calibration (K = 1)](#32-single-point-anchor-calibration-k--1)
   - [3.3 Prior-Regularized Least Squares ($\lambda$-Ridge)](#33-prior-regularized-least-squares-lambda-ridge)
   - [3.4 Non-Linear Polynomial Calibration](#34-non-linear-polynomial-calibration)
   - [3.5 Monotonic Isotonic Regression Calibration](#35-monotonic-isotonic-regression-calibration)
   - [3.6 GCP Residuals & Goodness-of-Fit](#36-gcp-residuals--goodness-of-fit)
4. [Coarse DEM & SRTM Terrain Datum Anchoring](#4-coarse-dem--srtm-terrain-datum-anchoring)
   - [4.1 Terrarium RGB Elevation Decoding](#41-terrarium-rgb-elevation-decoding)
   - [4.2 Web Mercator & Slippy Map Projection](#42-web-mercator--slippy-map-projection)
   - [4.3 Ground Terrain Candidate Extraction](#43-ground-terrain-candidate-extraction)
   - [4.4 Robust Datum Offset Anchoring](#44-robust-datum-offset-anchoring)
5. [Shadow Geometry & Vertical Structure Relief Extraction](#5-shadow-geometry--vertical-structure-relief-extraction)
   - [5.1 Spectral Luminance & Rayleigh Skylight Ratios](#51-spectral-luminance--rayleigh-skylight-ratios)
   - [5.2 Continuous Shadow Confidence Formulation](#52-continuous-shadow-confidence-formulation)
   - [5.3 Shadow Length & Direction Extraction via PCA](#53-shadow-length--direction-extraction-via-pca)
   - [5.4 Building Height Trigonometry](#54-building-height-trigonometry)
   - [5.5 Absolute Building Rooftop Elevation](#55-absolute-building-rooftop-elevation)
   - [5.6 Multivariate Uncertainty & Error Propagation](#56-multivariate-uncertainty--error-propagation)
6. [Guided Filter Edge-Preserving Depth Refinement](#6-guided-filter-edge-preserving-depth-refinement)
   - [6.1 Grayscale Guided Filter](#61-grayscale-guided-filter)
   - [6.2 3-Channel Color Guided Filter](#62-3-channel-color-guided-filter)
   - [6.3 Color-Edge Gated Disparity Sharpening](#63-color-edge-gated-disparity-sharpening)
7. [Geomorphometry & Topographic Shaded Relief](#7-geomorphometry--topographic-shaded-relief)
   - [7.1 Surface Gradient & Slope Angle](#71-surface-gradient--slope-angle)
   - [7.2 Horn's Algorithm for Shaded Relief (Hillshade)](#72-horns-algorithm-for-shaded-relief-hillshade)
   - [7.3 Heuristic Reliability / Confidence Mapping](#73-heuristic-reliability--confidence-mapping)
8. [Statistical Verification & Benchmark Metrics](#8-statistical-verification--benchmark-metrics)
9. [Zero-Fabrication Safety Contract](#9-zero-fabrication-safety-contract)

---

## 1. Monocular Depth & Tiled Spatial Blending

*Implementation: [`depth/tiled_inference.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/depth/tiled_inference.py)*

For high-resolution satellite orthophotos exceeding single-pass GPU memory, imagery is partitioned into overlapping tiles of size $T \times T$ with stride $S = \lfloor T \cdot (1 - \text{overlap}) \rfloor$.

### 1.1 2D Hann Window Weighting Matrix
To eliminate seam boundaries across tile overlaps, a 2D Hann weighting window $W(y, x)$ is evaluated as the outer tensor product of two 1D windows:

$$w_y(i) = \sin^2\left(\frac{\pi (2i + 1)}{2H}\right), \quad i \in [0, H-1]$$

$$w_x(j) = \sin^2\left(\frac{\pi (2j + 1)}{2W}\right), \quad j \in [0, W-1]$$

$$W(i, j) = w_y(i) \otimes w_x(j) = w_y(i) \cdot w_x(j)$$

### 1.2 Multi-Tile Weighted Blending
For $M$ overlapping tile predictions $D_k(x, y)$ positioned at tile coordinate offsets $(x_{0,k}, y_{0,k})$:

$$D_{\text{accum}}(x, y) = \sum_{k=1}^{M} W_k(x, y) \cdot D_k(x, y)$$

$$W_{\text{accum}}(x, y) = \sum_{k=1}^{M} W_k(x, y)$$

$$D_{\text{final}}(x, y) = \frac{D_{\text{accum}}(x, y)}{W_{\text{accum}}(x, y) + \epsilon}$$

---

## 2. Relative Digital Surface Model Normalization (rDSM)

*Implementation: [`calibration/rdsm.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/calibration/rdsm.py)*

Relative disparity values $D(x, y)$ are normalized to an invariant scale $[0.0, 1.0]$ using robust percentile bounds to reject sensor anomalies and high-frequency edge outliers:

$$\text{rDSM}(x, y) = \text{clip}\left( \frac{D(x, y) - D_{p_{\text{low}}}}{D_{p_{\text{high}}} - D_{p_{\text{low}}} + \epsilon}, \; 0.0, \; 1.0 \right)$$

where:
- $D_{p_{\text{low}}} = \text{Percentile}(D, 0.5\%)$
- $D_{p_{\text{high}}} = \text{Percentile}(D, 99.5\%)$
- $\epsilon = 10^{-7}$ (regularization to prevent division by zero in flat terrain).

---

## 3. Metric Elevation Calibration Engine (Absolute Height)

*Implementation: [`calibration/metric.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/calibration/metric.py), [`calibration/gcp_calibration.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/calibration/gcp_calibration.py)*

### 3.1 Linear Ordinary Least Squares (OLS)
Given $K \ge 2$ Ground Control Points $\mathcal{P} = \{(x_i, y_i, H_i)\}_{i=1}^K$ with sampled predicted disparity $D_i = D(x_i, y_i)$:

$$H(x, y) = a \cdot D(x, y) + b$$

Cost function:
$$\min_{a, b} \mathcal{L}(a, b) = \sum_{i=1}^{K} (a D_i + b - H_i)^2$$

Closed-form parameter solution:
$$a = \frac{\text{Cov}(D, H)}{\text{Var}(D)} = \frac{\sum_{i=1}^K (D_i - \bar{D})(H_i - \bar{H})}{\sum_{i=1}^K (D_i - \bar{D})^2}$$

$$b = \bar{H} - a \bar{D}$$

where $\bar{D} = \frac{1}{K}\sum D_i$ and $\bar{H} = \frac{1}{K}\sum H_i$.

---

### 3.2 Single-Point Anchor Calibration (K = 1)
When only one GCP $(D_1, H_1)$ is available, the scale is fixed to a regional prior $a_{\text{prior}}$:

$$a = a_{\text{prior}}$$

$$b = H_1 - a_{\text{prior}} \cdot D_1$$

---

### 3.3 Prior-Regularized Least Squares ($\lambda$-Ridge)
When sparse GCPs ($K < 4$) are provided, a scale penalty regularizes $a$ toward the expected physical scale prior $a_{\text{prior}}$:

$$\min_{a, b} \left[ \sum_{i=1}^{K} (a D_i + b - H_i)^2 + \lambda (a - a_{\text{prior}})^2 \right]$$

The normal matrix equation is:

$$\begin{bmatrix} \sum_{i=1}^K D_i^2 + \lambda & \sum_{i=1}^K D_i \\ \sum_{i=1}^K D_i & K \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} \sum_{i=1}^K D_i H_i + \lambda a_{\text{prior}} \\ \sum_{i=1}^K H_i \end{bmatrix}$$

Determinant:
$$\Delta = K \left(\sum_{i=1}^K D_i^2 + \lambda\right) - \left(\sum_{i=1}^K D_i\right)^2$$

Analytical solutions:
$$a = \frac{K \left(\sum D_i H_i + \lambda a_{\text{prior}}\right) - \left(\sum D_i\right)\left(\sum H_i\right)}{\Delta}$$

$$b = \frac{\left(\sum D_i^2 + \lambda\right)\left(\sum H_i\right) - \left(\sum D_i\right)\left(\sum D_i H_i + \lambda a_{\text{prior}}\right)}{\Delta}$$

---

### 3.4 Non-Linear Polynomial Calibration
For non-linear sensor responses across large terrain reliefs ($K \ge d + 1$ GCPs):

$$H(x, y) = \sum_{k=0}^{d} c_k \cdot D(x, y)^k = c_d D^d + c_{d-1} D^{d-1} + \dots + c_1 D + c_0$$

Coefficients $\mathbf{c} = [c_d, \dots, c_0]^T$ are solved via:
$$\mathbf{c} = (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T \mathbf{H}$$
where $\mathbf{V}_{i, j} = D_i^{d - j}$ is the Vandermonde design matrix.

---

### 3.5 Monotonic Isotonic Regression Calibration
To enforce strict physical elevation monotonicity ($\frac{\partial H}{\partial D} \ge 0$) without imposing an arbitrary polynomial shape:

$$\min_{\hat{\mathbf{H}}} \sum_{i=1}^K \left(H_i - \hat{H}_i\right)^2 \quad \text{subject to } \hat{H}_i \le \hat{H}_j \text{ whenever } D_i \le D_j$$

Inference evaluation:
$$H(x, y) = \hat{H}_k + \frac{\hat{H}_{k+1} - \hat{H}_k}{D_{k+1} - D_k} \big(D(x, y) - D_k\big) \quad \text{for } D_k \le D(x, y) \le D_{k+1}$$

---

### 3.6 GCP Residuals & Goodness-of-Fit
$$\text{residual}_i = (a D_i + b) - H_i$$

$$\text{MAE}_{\text{GCP}} = \frac{1}{K} \sum_{i=1}^K |\text{residual}_i|$$

$$\text{RMSE}_{\text{GCP}} = \sqrt{\frac{1}{K} \sum_{i=1}^K (\text{residual}_i)^2}$$

---

## 4. Coarse DEM & SRTM Terrain Datum Anchoring

*Implementation: [`calibration/srtm_anchor.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/calibration/srtm_anchor.py)*

### 4.1 Terrarium RGB Elevation Decoding
Mapzen / AWS Terrain Open Data encodes global terrain elevation in 24-bit RGB Terrarium PNGs:

$$\text{Elevation (m)} = \left( R \cdot 256.0 + G + \frac{B}{256.0} \right) - 32768.0$$

### 4.2 Web Mercator & Slippy Map Projection
Tile indices at zoom level $z$:

$$x_{\text{tile}} = \left\lfloor \frac{\text{lon} + 180^\circ}{360^\circ} \cdot 2^z \right\rfloor$$

$$y_{\text{tile}} = \left\lfloor \left( 1 - \frac{\ln\left(\tan(\text{lat}_{\text{rad}}) + \sec(\text{lat}_{\text{rad}})\right)}{\pi} \right) \cdot \frac{2^z}{2} \right\rfloor$$

Web Mercator projected metric coordinates (EPSG:3857):
$$x_m = \text{lon} \cdot \frac{20037508.34}{180^\circ}$$

$$y_m = \frac{\ln\left(\tan\left(\frac{90^\circ + \text{lat}}{360^\circ} \pi\right)\right)}{\pi / 180^\circ} \cdot \frac{20037508.34}{180^\circ}$$

### 4.3 Ground Terrain Candidate Extraction
$$\mathcal{M}_{\text{terrain}} = \left\{ (x, y) \;\middle|\; \text{Percentile}(D, 1.0\%) \le D(x, y) \le \text{Percentile}(D, 25.0\%) \right\}$$

### 4.4 Robust Datum Offset Anchoring
The coarse DEM is reprojected and resampled to match the high-resolution RGB grid. The absolute vertical datum offset $b$ is:

$$b = \text{median}\Big(H_{\text{DEM}}(x, y) \;\Big|\; (x, y) \in \mathcal{M}_{\text{terrain}}\Big) - a \cdot \text{median}\Big(D(x, y) \;\Big|\; (x, y) \in \mathcal{M}_{\text{terrain}}\Big)$$

$$H(x, y) = a \cdot D(x, y) + b$$

---

## 5. Shadow Geometry & Vertical Structure Relief Extraction

*Implementation: [`shadow_detection.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/shadow_detection.py), [`shadow-cue/shadow/height.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/shadow-cue/shadow/height.py)*

### 5.1 Spectral Luminance & Rayleigh Skylight Ratios
Luminance intensity (ITU-R BT.601):
$$I = 0.299 R + 0.587 G + 0.114 B$$

Blue / Rayleigh scattering ratio:
$$\text{BlueRatio} = \frac{B}{\frac{R + G}{2.0} + 10^{-5}}$$

Normalized spectral blue-red differential:
$$\text{DiffBR} = \frac{B - R}{B + R + 10^{-4}}$$

### 5.2 Continuous Shadow Confidence Formulation
$$\text{DarkScore} = \text{clip}\left( \frac{1.5 \cdot I_{\text{thresh}} - I}{1.5 \cdot I_{\text{thresh}} + 10^{-5}}, \; 0.0, \; 1.0 \right)$$

$$\text{BlueScore} = \text{clip}\left( \frac{\text{BlueRatio} - 0.80}{0.50}, \; 0.0, \; 1.0 \right)$$

$$\text{SpectralScore} = \text{clip}\left( \frac{\text{DiffBR} + 0.10}{0.40}, \; 0.0, \; 1.0 \right)$$

$$S_{\text{conf}}(x, y) = \text{DarkScore} \times \Big(0.60 \cdot \text{BlueScore} + 0.40 \cdot \text{SpectralScore}\Big)$$

### 5.3 Shadow Length & Direction Extraction via PCA
For connected component shadow region $\Omega_k = \{(r_i, c_i)\}_{i=1}^N$:

$$\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i, \quad \mathbf{\Sigma} = \frac{1}{N} \sum_{i=1}^N (\mathbf{p}_i - \mathbf{\mu})(\mathbf{p}_i - \mathbf{\mu})^T$$

Eigendecomposition:
$$\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v} \implies \lambda_{\max}, \; \mathbf{v}_{\max}$$

$$L_{\text{pixel}} = 4.0 \cdot \sqrt{\max(1.0, \lambda_{\max})}$$

$$\phi_{\text{shadow}} = \text{atan2}(v_{\max, r}, \; v_{\max, c}) \times \frac{180^\circ}{\pi}$$

### 5.4 Building Height Trigonometry
Given ground sampling distance $\text{GSD} = \sqrt{|a_{\text{affine}} \cdot e_{\text{affine}}|}$ and solar elevation angle $\theta_{\text{elev}}$:

$$L_{\text{physical}} = L_{\text{pixel}} \cdot \text{GSD}$$

$$h_{\text{building}} = L_{\text{physical}} \cdot \tan(\theta_{\text{elev}}) = (L_{\text{pixel}} \cdot \text{GSD}) \cdot \tan(\theta_{\text{elev}})$$

### 5.5 Absolute Building Rooftop Elevation
$$H_{\text{rooftop}} = H_{\text{ground datum}} + h_{\text{building}} = H_{\text{ground datum}} + (L_{\text{pixel}} \cdot \text{GSD}) \cdot \tan(\theta_{\text{elev}})$$

### 5.6 Multivariate Uncertainty & Error Propagation
$$H(L_{\text{px}}, s, \theta) = L_{\text{px}} \cdot s \cdot \tan\theta$$

Partial derivatives:
$$\frac{\partial H}{\partial L_{\text{px}}} = s \cdot \tan\theta$$

$$\frac{\partial H}{\partial s} = L_{\text{px}} \cdot \tan\theta$$

$$\frac{\partial H}{\partial \theta} = L_{\text{px}} \cdot s \cdot \sec^2\theta = \frac{L_{\text{px}} \cdot s}{\cos^2\theta}$$

Total standard deviation $\sigma_H$ (assuming uncorrelated error components):
$$\sigma_H = \sqrt{\left(\frac{\partial H}{\partial L_{\text{px}}} \delta L_{\text{px}}\right)^2 + \left(\frac{\partial H}{\partial s} \delta s\right)^2 + \left(\frac{\partial H}{\partial \theta} \delta \theta_{\text{rad}}\right)^2}$$

$$\sigma_H = \sqrt{\big(s \tan\theta \cdot \delta L_{\text{px}}\big)^2 + \big(L_{\text{px}} \tan\theta \cdot \delta s\big)^2 + \left(\frac{L_{\text{px}} s}{\cos^2\theta} \cdot \frac{\pi}{180^\circ} \delta \theta_{\text{deg}}\right)^2}$$

---

## 6. Guided Filter Edge-Preserving Depth Refinement

*Implementation: [`shadow/guided_filter.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/shadow/guided_filter.py)*

### 6.1 Grayscale Guided Filter (He et al., TPAMI 2013)
Local linear model in window $\omega_k$:
$$q_i = a_k I_i + b_k \quad \forall i \in \omega_k$$

Minimization objective:
$$E(a_k, b_k) = \sum_{i \in \omega_k} \Big( (a_k I_i + b_k - p_i)^2 + \epsilon a_k^2 \Big)$$

Parameters:
$$a_k = \frac{\frac{1}{|\omega|} \sum_{i \in \omega_k} I_i p_i - \mu_I \mu_p}{\sigma_I^2 + \epsilon}$$

$$b_k = \mu_p - a_k \mu_I$$

Final filtered disparity:
$$q_i = \bar{a}_i I_i + \bar{b}_i$$
where $\bar{a}_i = \frac{1}{|\omega|} \sum_{k \in \omega_i} a_k$ and $\bar{b}_i = \frac{1}{|\omega|} \sum_{k \in \omega_i} b_k$.

### 6.2 3-Channel Color Guided Filter
$$q_i = \mathbf{a}_k^T \mathbf{I}_i + b_k = a_{r,k} R_i + a_{g,k} G_i + a_{b,k} B_i + b_k$$

$$\mathbf{a}_k = \left(\mathbf{\Sigma}_k + \epsilon \mathbf{I}_3\right)^{-1} \text{Cov}_{\mathbf{I}, p}$$

$$b_k = \mu_{p,k} - \mathbf{a}_k^T \mathbf{\mu}_{\mathbf{I},k}$$

$$q_i = \bar{\mathbf{a}}_i^T \mathbf{I}_i + \bar{b}_i$$

### 6.3 Color-Edge Gated Disparity Sharpening
Gradient magnitude in CIELAB color space:
$$\|\nabla \mathbf{I}_{\text{LAB}}\| = \sqrt{\left(\frac{\partial L^*}{\partial x}\right)^2 + \left(\frac{\partial L^*}{\partial y}\right)^2 + \left(\frac{\partial a^*}{\partial x}\right)^2 + \left(\frac{\partial a^*}{\partial y}\right)^2 + \left(\frac{\partial b^*}{\partial x}\right)^2 + \left(\frac{\partial b^*}{\partial y}\right)^2}$$

Edge gating weight:
$$W_{\text{edge}} = \text{clip}\left( \frac{\|\nabla \mathbf{I}_{\text{LAB}}\|}{\text{Percentile}_{95}(\|\nabla \mathbf{I}_{\text{LAB}}\|) + 10^{-5}}, \; 0.0, \; 1.0 \right)$$

Sharpened disparity:
$$D_{\text{sharp}} = D + \alpha_{\text{strength}} \cdot \big(D - \text{BoxFilter}(D, 3\times 3)\big) \cdot W_{\text{edge}}$$

---

## 7. Geomorphometry & Topographic Shaded Relief

*Implementation: [`process_image.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/process_image.py)*

### 7.1 Surface Gradient & Slope Angle
$$g_x = \frac{\partial H}{\partial x} \frac{1}{\text{GSD}_x}, \quad g_y = \frac{\partial H}{\partial y} \frac{1}{\text{GSD}_y}$$

$$\text{Slope (deg)} = \arctan\left(\sqrt{g_x^2 + g_y^2}\right) \times \frac{180^\circ}{\pi} \quad \in [0^\circ, 90^\circ]$$

### 7.2 Horn's Algorithm for Shaded Relief (Hillshade)
Terrain aspect angle:
$$\text{Aspect} = \text{atan2}(-g_y, -g_x)$$

Solar azimuth mathematical angle:
$$\phi_{\text{math}} = \text{radians}(360^\circ - \phi_{\text{azimuth, geo}} + 90^\circ)$$

Hillshade intensity in $[0, 255]$:
$$\text{Shaded} = \sin(\theta_{\text{altitude}}) \cos(\text{Slope}) + \cos(\theta_{\text{altitude}}) \sin(\text{Slope}) \cos(\phi_{\text{math}} - \text{Aspect})$$

$$\text{Hillshade} = \left\lfloor 255.0 \cdot \text{clip}(\text{Shaded}, 0.0, 1.0) \right\rfloor$$

### 7.3 Heuristic Reliability / Confidence Mapping
$$C_{\text{boundary}}(x, y) = \text{clip}\left(\frac{\min(x, W-1-x, y, H-1-y)}{\text{Margin}_{\text{border}}}, \; 0.35, \; 1.0\right)$$

$$C_{\text{shadow}}(x, y) = 1.0 - 0.35 \cdot S_{\text{conf}}(x, y)$$

$$C_{\text{gradient}}(x, y) = \text{clip}\left(1.0 - 0.20 \cdot \frac{\|\nabla D(x, y)\|}{2.0 \cdot \text{Percentile}_{95}(\|\nabla D\|)}, \; 0.50, \; 1.0\right)$$

$$C(x, y) = 0.90 \cdot C_{\text{boundary}}(x, y) \cdot C_{\text{shadow}}(x, y) \cdot C_{\text{gradient}}(x, y) \quad \in [0.0, 1.0]$$

---

## 8. Statistical Verification & Benchmark Metrics

*Implementation: [`evaluation/metrics.py`](file:///c:/Users/Meher/OneDrive/Desktop/DepthWizard/evaluation/metrics.py)*

For predicted elevation raster $\mathbf{P} = \{P_i\}_{i=1}^N$ and ground-truth LiDAR reference DSM $\mathbf{R} = \{R_i\}_{i=1}^N$:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |P_i - R_i|$$

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (P_i - R_i)^2}$$

3. **Pearson Correlation Coefficient ($r$):**
   $$r = \frac{\sum_{i=1}^N (P_i - \bar{P})(R_i - \bar{R})}{\sqrt{\sum_{i=1}^N (P_i - \bar{P})^2 \sum_{i=1}^N (R_i - \bar{R})^2}}$$

4. **Coefficient of Determination ($R^2$):**
   $$SS_{\text{tot}} = \sum_{i=1}^N (R_i - \bar{R})^2, \quad SS_{\text{res}} = \sum_{i=1}^N (P_i - R_i)^2$$
   $$R^2 = 1.0 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1.0 - \frac{\sum_{i=1}^N (P_i - R_i)^2}{\sum_{i=1}^N (R_i - \bar{R})^2}$$

5. **Pixel-wise Absolute Error Map:**
   $$E(x, y) = |P(x, y) - R(x, y)|$$

---

## 9. Zero-Fabrication Safety Contract

To maintain scientific integrity across remote sensing workflows:

$$\text{Output Model} = \begin{cases}
H(x, y) = a D(x, y) + b \in \mathbb{R} \text{ (metres)}, & \text{if GCP or DEM calibration is supplied} \\
H_{\text{roof}} = H_{\text{datum}} + L s \tan\theta \in \mathbb{R} \text{ (metres)}, & \text{if solar angle tags \& GSD are present} \\
\text{rDSM}(x, y) \in [0.0, 1.0] \text{ (unitless)}, & \text{if uncalibrated (Zero-Fabrication Mode)}
\end{cases}$$
