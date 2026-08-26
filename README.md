---
title: DepthWizard
emoji: 🏔️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# DepthWizard — 3D Surface Reconstruction & Elevation Extraction from Monocular Remote Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Frontend-Three.js_WebGL-black.svg)](https://threejs.org/)

---

## 📌 Overview

**DepthWizard** is an end-to-end AI and geospatial engine designed to derive dense, accurate Digital Surface Models (DSMs) and interactive 3D terrain reconstructions from **single-view optical remote sensing imagery**.

By bridging state-of-the-art self-supervised Vision Transformers (**Depth Anything V2**) with physical geospatial geodesy, shadow cue geometry (**M4**), and ground control point (GCP) / SRTM DEM anchoring (**M2**), DepthWizard provides continuous metric 3D elevation in seconds without requiring expensive multi-view stereo passes or airborne LiDAR flights.

---

## 🚀 Key Features

- **Monocular Depth Backbone:** High-resolution spatial disparity estimation using DINOv2 Vision Transformers.
- **Metric Calibration Engine ($H = a \cdot D + b$):** Closed-form Ordinary Least Squares (OLS) solver mapping relative disparity into physical elevation (meters) via sparse GCP anchors or SRTM DEM percentile anchoring.
- **Zero-Fabrication Safety Contract:** If no geospatial reference or elevation anchors are provided, the system strictly outputs normalized relative surface models ($\text{rDSM} \in [0, 1]$), preventing hallucinated metric measurements.
- **Invariant Shadow Geometry:** Employs solar azimuth and elevation angles ($\theta$) to derive physical building relief ($h = L \cdot \tan\theta$) and validate structural vertical boundaries.
- **60 FPS 3D WebGL Visualization:** Real-time Three.js viewport featuring interactive orbit, cinematic flythrough, live raycast elevation cursor HUD, dynamic topographic transects, and surface overlays (RGB / Slope / Confidence / Error Map).
- **Standard GIS Export:** Native export of 32-bit Float GeoTIFFs preserving spatial reference systems (CRS) and affine geotransforms for seamless use in QGIS and ArcGIS.

---

## 🏗️ System Architecture

```
                                  [ Optical Image ]
                               (GeoTIFF / PNG / JPG)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [ Depth Anything V2 ]                            [ Metadata & Shadows ]
   (DINOv2 Feature Encoder)                         (CRS, GSD, Solar Geometry)
                 │                                               │
                 ▼                                               ▼
      [ Disparity Map D(x,y) ]                         [ Ground Anchors (GCP/DEM) ]
                 └───────────────────────┬───────────────────────┘
                                         ▼
                             [ Metric Calibration ]
                           H(x,y) = a · D(x,y) + b
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [ 32-bit Float GeoTIFF ]                       [ Three.js 3D WebGL ]
         (QGIS / ArcGIS DSM)                        (60 FPS Flythrough & HUD)
```

---

## 💻 Tech Stack

- **Deep Learning:** PyTorch, Hugging Face Transformers, Depth Anything V2, DINOv2
- **Geospatial & Remote Sensing:** GDAL, Rasterio, PROJ, NumPy, SciPy, OpenCV, Shapely
- **Backend API:** FastAPI (Async ASGI Engine), Uvicorn, Pydantic
- **Frontend & 3D:** HTML5 Canvas, Vanilla CSS3, Three.js (WebGL), OrbitControls

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mehersoni/DepthWizard.git
cd DepthWizard

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launching the Live 3D Web Console

Start the FastAPI application:

```bash
python api_server.py
```

Open your browser and navigate to:
- **Landing Page & Console:** `http://127.0.0.1:8000/index.html`
- **Interactive 3D Demo:** `http://127.0.0.1:8000/demo.html`
- **Advanced M6 Elevation Console:** `http://127.0.0.1:8000/m6_dashboard.html`

---

## 📊 Scientific Verification & Benchmarks

The system has been evaluated against the **ISPRS Potsdam Urban Benchmark** (high-resolution true orthophotos paired with airborne LiDAR ground truth):

| Metric | Uncalibrated rDSM | SRTM Anchored | Sparse GCP Calibrated (4 pts) |
| :--- | :---: | :---: | :---: |
| **Correlation ($R^2$)** | — | 0.81 | **0.92** |
| **Mean Absolute Error (MAE)** | Scale-Agnostic | 3.72 m | **0.68 m** |
| **Processing Latency** | < 2.5s (CPU) | < 2.8s (CPU) | **< 2.4s (CPU) / < 250ms (GPU)** |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
