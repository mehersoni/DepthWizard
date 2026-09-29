import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_flowchart_png():
    # Setup figure with 16:9 aspect ratio and dark theme
    fig = plt.figure(figsize=(22, 12), facecolor="#090d16")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#090d16")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color palette
    COLOR_BG_CARD = "#111827"
    COLOR_BORDER_DEFAULT = "#374151"
    COLOR_CYAN = "#06b6d4"
    COLOR_BLUE = "#3b82f6"
    COLOR_INDIGO = "#818cf8"
    COLOR_EMERALD = "#10b981"
    COLOR_AMBER = "#f59e0b"
    COLOR_PURPLE = "#c084fc"
    COLOR_ROSE = "#fb7185"
    COLOR_TEXT = "#f8fafc"
    COLOR_MUTED = "#94a3b8"

    # Header Title
    ax.text(50, 96.5, "DEPTHWIZARD: END-TO-END SYSTEM ARCHITECTURE", 
            ha='center', va='center', fontsize=22, fontweight='bold', color=COLOR_TEXT, family='sans-serif')
    ax.text(50, 93.8, "Single-View Optical Remote Sensing → Physical Metric 3D Surface Reconstruction & Geodetic Analytics", 
            ha='center', va='center', fontsize=12, color=COLOR_CYAN, family='sans-serif')

    def draw_box(x, y, w, h, title, subtitle="", bg=COLOR_BG_CARD, border=COLOR_BORDER_DEFAULT, title_color=COLOR_TEXT, rx=0.8):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.25,rounding_size={rx}",
                                     facecolor=bg, edgecolor=border, linewidth=1.5, zorder=3)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h/2 + 1.2, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=title_color, zorder=4)
            ax.text(x + w/2, y + h/2 - 1.3, subtitle, ha='center', va='center', fontsize=8.2, color=COLOR_MUTED, zorder=4)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=title_color, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color=COLOR_MUTED, lw=1.6, label="", label_pos=(0.5, 0.5)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=13, zorder=5))
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=8.2, fontweight='bold',
                    color=color, bbox=dict(boxstyle="round,pad=0.2", fc="#090d16", ec=color, lw=0.8, alpha=0.95), zorder=6)

    def draw_phase_container(x, y, w, h, phase_num, title, color_accent):
        container = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                           facecolor="#0f172a", edgecolor=color_accent, linewidth=1.3, linestyle="--", alpha=0.7, zorder=1)
        ax.add_patch(container)
        tag_text = f"PHASE {phase_num}: {title.upper()}"
        tag_w = min(w - 2.5, len(tag_text) * 0.42 + 2.0)
        tag_box = patches.FancyBboxPatch((x + 1.5, y + h - 2.4), tag_w, 2.4, boxstyle="round,pad=0.2,rounding_size=0.5",
                                        facecolor=color_accent, edgecolor="none", zorder=3)
        ax.add_patch(tag_box)
        ax.text(x + 2.5, y + h - 1.2, tag_text, ha='left', va='center', fontsize=8.5, fontweight='bold', color="#090d16", zorder=4)

    # ---------------- PHASE 1: INPUT & METADATA (Left Column) ----------------
    draw_phase_container(2.5, 48, 17, 42, 1, "Input & Ingestion", COLOR_CYAN)
    draw_box(4, 76, 14, 9.5, "Optical Aerial/Satellite Image", "GeoTIFF, PNG, JPG (RGB / Pan)", border=COLOR_CYAN, title_color="#38bdf8")
    draw_arrow(11, 76, 11, 68)
    draw_box(4, 52, 14, 16, "Geospatial Ingestion Engine", "• Affine Transform & CRS (EPSG)\n• Pixel GSD (m/px)\n• Solar Ephemeris (Azimuth, Elev θ)\n• Sensor Metadata Extraction", border=COLOR_CYAN, title_color="#38bdf8")

    # Connect Phase 1 to Phase 2
    draw_arrow(18, 62, 22.5, 78, color=COLOR_CYAN, label="Image Stream", label_pos=(0.35, 0.45))
    draw_arrow(11, 52, 11, 23, color=COLOR_AMBER)
    draw_arrow(11, 23, 44, 23, color=COLOR_AMBER, label="Solar θ, φ & GSD", label_pos=(0.4, 0.5))

    # ---------------- PHASE 2: MONOCULAR PERCEPTION (M1) ----------------
    draw_phase_container(22.5, 48, 18.5, 42, 2, "Monocular Perception (M1)", COLOR_BLUE)
    draw_box(24, 74, 15.5, 9.5, "Tiled Sliding Window", "1024×1024 with 25% Spatial Overlap\nDynamic Padding on Edges", border=COLOR_BLUE, title_color="#60a5fa")
    draw_arrow(31.75, 74, 31.75, 67)
    draw_box(24, 58, 15.5, 9, "Depth Anything V2 Backbone", "DINOv2 Feature Encoder\nMulti-scale Disparity Decoder", border=COLOR_BLUE, title_color="#60a5fa")
    draw_arrow(31.75, 58, 31.75, 51)
    draw_box(24, 40, 15.5, 11, "2D Hann Window Blending", "W(x, y) = w_x ⊗ w_y\nZero Boundary Seam Artifacts\nPercentile rDSM ∈ [0.0, 1.0]", border=COLOR_BLUE, title_color="#60a5fa")

    # Connect Phase 2 to Phase 3
    draw_arrow(39.5, 45.5, 44, 77, color=COLOR_BLUE, label="Disparity D(x, y)", label_pos=(0.35, 0.4))

    # ---------------- PHASE 3: CALIBRATION DECISION TREE (M2) ----------------
    draw_phase_container(44, 41, 25.5, 49, 3, "Metric Calibration (M2)", COLOR_INDIGO)
    
    # Diamond 1: GCP available?
    dia_gcp = patches.Polygon([[56.75, 84], [64, 78], [56.75, 72], [49.5, 78]], closed=True,
                              facecolor="#2e1065", edgecolor=COLOR_INDIGO, linewidth=1.5, zorder=3)
    ax.add_patch(dia_gcp)
    ax.text(56.75, 78, "Are Ground Control\nPoints (GCPs) Available?", ha='center', va='center', fontsize=8.2, fontweight='bold', color=COLOR_TEXT, zorder=4)

    # Path A: GCP OLS
    draw_arrow(64, 78, 68, 78, color=COLOR_EMERALD)
    draw_box(57.5, 59, 10.5, 8.5, "Closed-Form OLS Fit", "H(x, y) = a·D + b (m)\nOptimal K=10 Points", border=COLOR_EMERALD, title_color="#34d399")
    draw_arrow(68, 78, 62.75, 67.5, color=COLOR_EMERALD, label="YES (K≥1)", label_pos=(0.3, 0.4))

    # Diamond 2: DEM available?
    draw_arrow(56.75, 72, 56.75, 65, color=COLOR_ROSE, label="NO", label_pos=(0.5, 0.5))
    dia_dem = patches.Polygon([[56.75, 65], [64, 59], [56.75, 53], [49.5, 59]], closed=True,
                              facecolor="#2e1065", edgecolor=COLOR_INDIGO, linewidth=1.5, zorder=3)
    ax.add_patch(dia_dem)
    ax.text(56.75, 59, "Is Coarse Regional\nDEM (SRTM) Available?", ha='center', va='center', fontsize=8.2, fontweight='bold', color=COLOR_TEXT, zorder=4)

    # Path B: DEM Anchoring
    draw_arrow(49.5, 59, 45.5, 53, color=COLOR_EMERALD)
    draw_box(45.5, 43, 10.5, 8, "DEM Datum Anchor", "Percentile Terrain Filter\nb ≈ DEM_25 - a·D_25", border=COLOR_EMERALD, title_color="#34d399")

    # Path C: Zero-Fabrication
    draw_arrow(64, 59, 64, 53, color=COLOR_ROSE, label="NO", label_pos=(0.5, 0.5))
    draw_box(57.5, 43, 11, 8, "Zero-Fabrication Contract", "Refuse Hallucinated Meters\nOutput pure rDSM [0, 1]", border=COLOR_ROSE, title_color="#fb7185")

    # ---------------- PHASE 4: SHADOW & REFINE (M4 & M7) ----------------
    draw_phase_container(44, 4, 25.5, 34, 4, "Physical Photogrammetry & Refinement", COLOR_AMBER)
    
    draw_box(45.5, 18, 10.5, 15, "Shadow Mensuration (M4)", "• 1D Outward Raycasting\n• Solar Azimuth Vector φ\n• Trig Height: H = L·tan(θ)\n• Uncertainty Bounds σ_h", border=COLOR_AMBER, title_color="#fbbf24")
    draw_arrow(56, 25.5, 58, 25.5, color=COLOR_AMBER)
    draw_box(58, 18, 10.5, 15, "Edge & Roof Refiner", "• Guided Filter (M7, r=16)\n• sat2Map Planar RANSAC\n• Normal Vector Quantization\n• Eliminates Curved Roofs", border=COLOR_AMBER, title_color="#fbbf24")

    # Connect Phase 3 outputs down to Phase 4
    draw_arrow(62.75, 43, 62.75, 33, color=COLOR_CYAN, label="Elevation Surface", label_pos=(0.5, 0.5))
    draw_arrow(50.75, 43, 50.75, 33, color=COLOR_EMERALD)

    # ---------------- PHASE 5: GEOMORPHOMETRY & OUTPUTS (Right Column) ----------------
    draw_phase_container(72.5, 4, 25, 86, 5, "Geomorphometry & Multimodal Outputs", COLOR_PURPLE)
    
    draw_box(74.5, 71, 21, 14, "Geomorphometric Analytics Engine", "• Horn's Algorithm (Topographic Hillshade)\n• Surface Gradient & Slope Map (Degrees)\n• Multi-scale Reliability & Confidence Mapping", border=COLOR_PURPLE, title_color="#c084fc")

    draw_arrow(74.5, 78, 69.5, 80, color=COLOR_EMERALD) # Connect from Phase 3 Calibrated

    draw_arrow(85, 71, 85, 64)
    draw_box(74.5, 50, 21, 14, "Standard GIS GeoTIFF Exporter", "• 32-bit Floating Point Metric DSM / rDSM\n• Preserves Affine Geotransform & CRS Projection\n• Ready for QGIS, ArcGIS & Hydrological Analysis", border=COLOR_EMERALD, title_color="#34d399")

    draw_arrow(85, 50, 85, 44)
    draw_box(74.5, 30, 21, 14, "3D CAD & Mesh Exporter", "• Stanford Binary Point Cloud (.ply with Normals)\n• Wavefront Textured Mesh (.obj + .mtl)\n• Triangular Surface Decimation for CAD", border=COLOR_CYAN, title_color="#38bdf8")

    draw_arrow(85, 30, 85, 23)
    draw_box(74.5, 8, 21, 15, "60 FPS Interactive WebGL Viewport", "• Real-Time Three.js Orbit & Drone Flythrough\n• Live Raycasting Elevation Inspector HUD\n• Dynamic Topographic Transects (Profile Slicing)\n• Texture Switcher (RGB / Turbo / Hillshade)", border=COLOR_AMBER, title_color="#fbbf24")

    # Connect Phase 4 refined mesh to Phase 5
    draw_arrow(68.5, 25.5, 74.5, 37, color=COLOR_AMBER, label="Refined Facets", label_pos=(0.4, 0.45))
    draw_arrow(68.5, 20, 74.5, 14, color=COLOR_AMBER)

    # Save output
    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "depthwizard_pipeline_flowchart.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[Success] Polished Flowchart PNG generated at: {out_path}")

if __name__ == "__main__":
    create_flowchart_png()
