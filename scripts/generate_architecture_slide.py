"""
Generates a presentation-ready 16:9 4K/1080p System Architecture & Process Flow Diagram
for DepthWizard (SIH26175).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def generate_architecture_slide():
    # 16:9 Aspect Ratio (1920x1080 at 120 DPI -> 16x9 inches)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor('#07090e')
    ax.set_facecolor('#07090e')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header / Title Bar
    ax.text(5, 94, "DEPTHWIZARD", fontsize=22, fontweight='bold', color='#38bdf8', fontfamily='sans-serif')
    ax.text(23, 94.2, "|  System Architecture & End-to-End Process Flow", fontsize=15, color='#94a3b8', fontfamily='sans-serif')
    ax.text(95, 94, "SIH26175", fontsize=13, fontweight='bold', color='#64748b', fontfamily='monospace', ha='right')

    # Subtitle / Tagline
    ax.text(5, 90.5, "Transformation of Single-View Optical RGB Satellite Imagery into Calibrated 3D Metric Elevation Representations", fontsize=10, color='#64748b', fontfamily='sans-serif')

    # Draw 5 Pipeline Stage Cards
    stages = [
        {
            "num": "01",
            "title": "INPUT INGESTION",
            "subtitle": "Format & Georeference Check",
            "color": "#38bdf8", # Sky Blue
            "x": 4, "w": 16.5, "y": 28, "h": 58,
            "items": [
                ("Input Imagery", "GeoTIFF (.tif) or Photo (.png/.jpg)"),
                ("Spatial Metadata", "Extract CRS (e.g. EPSG:32633)"),
                ("Affine Transform", "6-parameter matrix resolution"),
                ("GSD Calculation", "Ground Sampling Distance (m/px)"),
                ("Format Routing", "Calibrated GeoTIFF vs Relative rDSM")
            ]
        },
        {
            "num": "02",
            "title": "AI DEPTH BACKBONE",
            "subtitle": "Monocular Relief Estimation",
            "color": "#818cf8", # Indigo / Violet
            "x": 23, "w": 16.5, "y": 28, "h": 58,
            "items": [
                ("Backbone", "Depth Anything V2 (Small/Base)"),
                ("Encoder", "DINOv2 Vision Transformer"),
                ("Decoder", "Dense Prediction Transformer (DPT)"),
                ("Output", "Continuous Disparity D(x, y)"),
                ("Invariance", "Affine-invariant topological relief")
            ]
        },
        {
            "num": "03",
            "title": "METRIC CALIBRATION",
            "subtitle": "Physical Elevation Inversion",
            "color": "#34d399", # Emerald Green
            "x": 42, "w": 16.5, "y": 28, "h": 58,
            "items": [
                ("Linear Model", "H(x, y) = a · D(x, y) + b"),
                ("User GCPs (K≥2)", "Closed-form OLS parameter fit"),
                ("SRTM DEM Datum", "Bilinear warp + terrain percentile"),
                ("M4 Shadow Cue", "Sun angle & shadow length h = L·tanθ"),
                ("Residual Validation", "Strict GCP MAE & RMSE tracking")
            ]
        },
        {
            "num": "04",
            "title": "GEOSPATIAL SIGNALS",
            "subtitle": "Ancillary Gradient Analysis",
            "color": "#fbbf24", # Amber / Gold
            "x": 61, "w": 16.5, "y": 28, "h": 58,
            "items": [
                ("Surface Slope", "Physical gradient in degrees [0, 90]°"),
                ("Confidence Map", "Gradient saliency score [0, 1]"),
                ("Transect Profiler", "Cross-section terrain elevation slice"),
                ("Error Residuals", "Pixel difference |H_pred - H_ref|"),
                ("Raster Export", "32-bit Float GeoTIFF / PNG")
            ]
        },
        {
            "num": "05",
            "title": "3D WEBGL ENGINE",
            "subtitle": "Interactive Mesh Exploration",
            "color": "#f43f5e", # Rose / Coral
            "x": 80, "w": 16, "y": 28, "h": 58,
            "items": [
                ("3D Framework", "Three.js WebGL + OrbitControls"),
                ("Mesh Synthesis", "256×256 vertex displacement"),
                ("Texture Mapping", "RGB Ortho / Slope / Confidence"),
                ("Camera Modes", "Orbital Inspection & Helical Fly"),
                ("Raycasting", "Real-time elevation coordinate probe")
            ]
        }
    ]

    for s in stages:
        # Card Background Box
        bg_rect = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor='#0f141f',
            edgecolor='#1e293b',
            linewidth=1.2,
            zorder=2
        )
        ax.add_patch(bg_rect)

        # Header Glow Strip
        strip = patches.FancyBboxPatch(
            (s["x"], s["y"] + s["h"] - 3.5), s["w"], 3.5,
            boxstyle="round,pad=0.8,rounding_size=1.0",
            facecolor=s["color"],
            edgecolor='none',
            alpha=0.15,
            zorder=3
        )
        ax.add_patch(strip)

        # Number Badge
        badge = patches.Circle((s["x"] + 2.2, s["y"] + s["h"] - 4.5), 1.8, facecolor=s["color"], edgecolor='none', zorder=4)
        ax.add_patch(badge)
        ax.text(s["x"] + 2.2, s["y"] + s["h"] - 4.5, s["num"], fontsize=9, fontweight='bold', color='#07090e', ha='center', va='center', zorder=5, fontfamily='monospace')

        # Stage Title & Subtitle
        ax.text(s["x"] + 4.8, s["y"] + s["h"] - 3.8, s["title"], fontsize=10.5, fontweight='bold', color='#f8fafc', fontfamily='sans-serif', zorder=5)
        ax.text(s["x"] + 4.8, s["y"] + s["h"] - 6.2, s["subtitle"], fontsize=7.5, color=s["color"], fontfamily='sans-serif', zorder=5)

        # Separator line
        ax.plot([s["x"] + 1, s["x"] + s["w"] - 1], [s["y"] + s["h"] - 8.5, s["y"] + s["h"] - 8.5], color='#1e293b', lw=1, zorder=4)

        # Feature List Items
        item_y = s["y"] + s["h"] - 12.5
        for heading, desc in s["items"]:
            # Bullet icon
            bullet = patches.Circle((s["x"] + 1.8, item_y + 0.3), 0.45, facecolor=s["color"], edgecolor='none', zorder=4)
            ax.add_patch(bullet)
            
            # Text
            ax.text(s["x"] + 3.0, item_y + 0.8, heading, fontsize=8.5, fontweight='bold', color='#e2e8f0', fontfamily='sans-serif', zorder=5)
            ax.text(s["x"] + 3.0, item_y - 1.6, desc, fontsize=7.2, color='#94a3b8', fontfamily='sans-serif', zorder=5)
            
            # Inner item separator
            item_y -= 8.5

    # Connecting Flow Arrows between stages
    arrow_props = dict(arrowstyle="-|>", color="#38bdf8", lw=2.2, mutation_scale=14)
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"] + 0.2
        x_end = stages[i+1]["x"] - 0.2
        y_pos = 57.0
        ax.annotate('', xy=(x_end, y_pos), xytext=(x_start, y_pos), arrowprops=arrow_props, zorder=6)

    # Bottom Technology Banner / Legend
    banner = patches.FancyBboxPatch(
        (4, 5), 92, 18,
        boxstyle="round,pad=0.8,rounding_size=1.2",
        facecolor='#0b0f19',
        edgecolor='#1e293b',
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(banner)

    ax.text(6, 19.5, "CORE TECHNOLOGY STACK & MATHEMATICAL FRAMEWORK", fontsize=9.5, fontweight='bold', color='#38bdf8', fontfamily='monospace')

    tech_cols = [
        ("DEEP LEARNING BACKBONE", ["PyTorch 2.x", "Depth Anything V2", "DINOv2 ViT Encoder", "DPT Decoder", "Hugging Face"]),
        ("GEOSPATIAL & CV ENGINE", ["Rasterio / GDAL", "NumPy & SciPy OLS", "OpenCV (cv2)", "Affine Transformations", "EPSG Coordinate Warping"]),
        ("CALIBRATION & CUES", ["Linear Fit: H = a·D + b", "GCP Least-Squares", "SRTM DEM Datum Anchoring", "M4 Geometric Shadow Cues", "Zero-Fabrication Fallback"]),
        ("FULL-STACK & 3D WEBGL", ["FastAPI & Uvicorn", "Three.js WebGL (r128)", "Orbit & Helical Camera", "Transect Profiler", "Playwright E2E Testing"])
    ]

    tx = 6.0
    for title, items in tech_cols:
        ax.text(tx, 15.8, title, fontsize=8.2, fontweight='bold', color='#f1f5f9', fontfamily='sans-serif')
        ty = 13.0
        for it in items:
            ax.text(tx, ty, f"• {it}", fontsize=7.2, color='#94a3b8', fontfamily='sans-serif')
            ty -= 2.0
        tx += 23.0

    # Save outputs
    os.makedirs("outputs/figures", exist_ok=True)
    out_path = "outputs/figures/system_architecture_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=180, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Generated High-Res System Architecture Slide: {out_path}")
    return os.path.abspath(out_path)


if __name__ == "__main__":
    generate_architecture_slide()
