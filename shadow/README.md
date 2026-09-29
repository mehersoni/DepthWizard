# M4 Shadow Cue Module - Shadow Candidate Detector

## Overview

The `shadow.detector` module performs the initial stage of the M4 Shadow Cue pipeline:

$$\text{RGB Satellite Image} \longrightarrow \mathbf{\text{Candidate Shadow Mask}} \longrightarrow \text{Cleaned Regions} \longrightarrow \text{Shadow Geometry} \longrightarrow \text{Structured Output}$$

Its primary function, `detect_shadow_candidates()`, converts an input BGR satellite image into a binary mask isolating pixels likely belonging to cast shadow regions.

---

## Physical & Optical Reasoning

In high-resolution satellite remote sensing, cast shadows occur when solid elevated structures (e.g., buildings, terrain features) obstruct direct solar illumination. 

### 1. Illumination Disparity (Solar vs. Ambient Diffuse Radiation)
- **Sunlit Regions**: Receive both direct beam solar radiation and indirect diffuse skylight radiation.
- **Shadowed Regions**: Direct solar radiation is obstructed. Shadowed pixels receive primarily indirect diffuse skylight (Rayleigh scattering) and ambient environment reflections.
- As a result, shadow regions exhibit a severe drop in overall luminance (radiance intensity) compared to their surrounding sunlit backgrounds.

### 2. Why HSV Color Space?
In standard RGB (or OpenCV's BGR) color space, pixel values $R, G, B$ combine both intensity (brightness) and chromaticity (color information). This makes direct RGB thresholding unreliable across varying surface albedos (e.g., dark asphalt vs. brightly illuminated roofs).

Transforming the image to **HSV (Hue, Saturation, Value)** space decouples color information from luminance:
- **Value ($V$)**: Represents light intensity/brightness directly $[0, 255]$. The primary indicator of a shadow is a low $V$ value.
- **Hue ($H$)**: Represents color type $[0, 180^\circ]$. Secondary Rayleigh scattering often gives shadows a subtle blue/cyan skylight hue bias.
- **Saturation ($S$)**: Represents color purity $[0, 255]$. Shadows tend to exhibit lower to moderate saturation levels relative to sunlit vegetation or colorful urban surfaces.

By applying threshold filters primarily on $V$ while constraining $S$ and $H$, we isolate dark candidate shadow pixels while preventing false positives on dark chromatic objects.

---

## Parameter Configuration & Adaptability

Hard-coded threshold values fail when satellite images are captured under varying atmospheric conditions, solar zenith angles, or surface albedo variations.

### Function Parameters

| Parameter | Type | Default | Explanation & Purpose |
| :--- | :--- | :--- | :--- |
| `image` | `np.ndarray` | *Required* | Input image in OpenCV BGR format with shape `(H, W, 3)` and `uint8` data type. |
| `v_max` | `int` | `80` | Upper bound threshold for the Value (brightness) channel $[0, 255]$. Pixels with $V \le v\_max$ are flagged as candidate shadows. |
| `s_min` | `int` | `0` | Lower bound threshold for the Saturation channel $[0, 255]$. |
| `s_max` | `int` | `255` | Upper bound threshold for the Saturation channel $[0, 255]$. |
| `h_min` | `int` | `0` | Lower bound for the Hue channel $[0, 180]$. |
| `h_max` | `int` | `180` | Upper bound for the Hue channel $[0, 180]$. |
| `adaptive_v` | `bool` | `False` | When enabled, dynamically computes `v_max` based on the image's overall luminance distribution instead of using a fixed static threshold. |
| `v_percentile` | `float` | `25.0` | Percentile $[0, 100]$ of the V-channel intensity distribution used to set `v_max` when `adaptive_v=True` (e.g., lower quartile of brightness). |

---

## Output Specification

The function returns a binary mask `candidate_mask` as a `np.ndarray` (`uint8`) with the same height and width as the input image:
- **`255`**: Pixel identified as a candidate shadow region.
- **`0`**: Non-shadow background pixel.
