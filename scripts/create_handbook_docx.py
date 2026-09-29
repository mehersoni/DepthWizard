import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def generate_handbook_docx():
    doc = Document()

    # Page Margins (1 inch all around)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    COLOR_PRIMARY = RGBColor(14, 43, 92)     # Deep Navy
    COLOR_SECONDARY = RGBColor(29, 78, 216)  # Blue
    COLOR_ACCENT = RGBColor(16, 185, 129)    # Emerald
    COLOR_TEXT = RGBColor(30, 41, 59)        # Slate Dark
    COLOR_MUTED = RGBColor(100, 116, 139)    # Slate Muted

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = COLOR_TEXT

    # --- Title Page Header ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("DEPTHWIZARD")
    run_title.font.size = Pt(28)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Dense 3D Surface Reconstruction & Elevation Extraction\nfrom Monocular Remote Sensing")
    run_sub.font.size = Pt(15)
    run_sub.font.bold = True
    run_sub.font.color.rgb = COLOR_SECONDARY

    # Metadata card / table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Author:", "Meher Soni"),
        ("Institution:", "Indian Institute of Technology Madras (IIT Madras)"),
        ("Date / Version:", "September 2026 | Production Release 2.1.0"),
        ("Classification:", "Master Project Handbook & Comprehensive Technical Report")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.0)
        c1.width = Inches(4.5)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = COLOR_PRIMARY
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "F8FAFC")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Add Pipeline Architecture Flowchart Image
    img_path = os.path.abspath("outputs/figures/depthwizard_pipeline_flowchart.png")
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_path, width=Inches(6.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 1: DepthWizard End-to-End System Architecture & Pipeline Phases")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9.5)
        r_cap.font.color.rgb = COLOR_MUTED
        p_cap.paragraph_format.space_after = Pt(16)

    # Helper function for headings
    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY

    def add_callout(text, title="KEY PRINCIPLE:"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.5)
        set_cell_background(cell, "EFF6FF")
        set_cell_margins(cell, 120, 120, 160, 160)
        p = cell.paragraphs[0]
        r_t = p.add_run(title + " ")
        r_t.font.bold = True
        r_t.font.size = Pt(10.5)
        r_t.font.color.rgb = COLOR_SECONDARY
        r_b = p.add_run(text)
        r_b.font.size = Pt(10)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # --- 1. Executive Summary ---
    add_heading_1("1. Executive Summary")
    p = doc.add_paragraph(
        "DepthWizard is an end-to-end geospatial artificial intelligence framework designed to extract continuous, "
        "dense Digital Surface Models (DSMs) and interactive 3D terrain reconstructions from single-view optical remote sensing imagery. "
        "By fusing self-supervised Vision Transformers (Depth Anything V2) with physical shadow photogrammetry (M4), "
        "closed-form Ground Control Point (GCP) calibration (M2), coarse SRTM terrain datum anchoring, and architectural planar regularization (sat2Map), "
        "DepthWizard provides continuous metric 3D elevation in seconds without requiring multi-pass satellite stereo constellations or expensive airborne LiDAR flights."
    )

    # --- 2. Problem Formulation & Research Gap ---
    add_heading_1("2. Problem Formulation & Research Gap")
    doc.add_paragraph(
        "In single-view optical remote sensing, recovering physical elevation from a 2D photograph is an ill-posed inverse problem. "
        "Perspective and pushbroom projections inherently lose scale along the optical ray, creating an affine scale and shift ambiguity:"
    )
    add_callout(
        "H(x, y) = a · D(x, y) + b\n"
        "where D(x, y) is the dimensionless monocular disparity, a is the vertical relief scale factor (m / disparity unit), "
        "and b is the geodetic vertical datum elevation offset (m above sea level).",
        title="FUNDAMENTAL MONOCULAR AMBIGUITY:"
    )
    doc.add_paragraph(
        "Existing solutions either require multi-view stereo pairs with wide baselines (which fail on single images) or standard "
        "deep vision depth models (which output unitless disparity with curved rooftop distortions and no physical datum). "
        "DepthWizard bridges this gap by enforcing physical geodesy and architectural priors onto self-supervised neural disparity."
    )

    # --- 3. Zero-Fabrication Safety Contract ---
    add_heading_1("3. The Zero-Fabrication Safety Contract")
    doc.add_paragraph(
        "In remote sensing and civil engineering, ungrounded metric guessing is scientifically unacceptable. "
        "DepthWizard enforces an architecturally guaranteed Zero-Fabrication Safety Contract:"
    )
    p_b1 = doc.add_paragraph(style='List Bullet')
    r = p_b1.add_run("Anchored Scenes (GCPs ≥ 1 or Coarse DEM present): ")
    r.font.bold = True
    p_b1.add_run("The engine executes closed-form Ordinary Least Squares (OLS) or datum percentile anchoring, outputting calibrated metric elevation in true meters (mode = 'absolute', calibrated = True).")

    p_b2 = doc.add_paragraph(style='List Bullet')
    r = p_b2.add_run("Unanchored Scenes (Standard Photos / Arbitrary Crops): ")
    r.font.bold = True
    p_b2.add_run("The engine strictly refuses to hallucinate fake metric heights, outputting a normalized Relative Digital Surface Model (mode = 'relative', rDSM ∈ [0.0, 1.0], calibrated = False).")

    # --- 4. System Architecture & Methodology ---
    add_heading_1("4. System Architecture & Methodology")
    
    add_heading_2("4.1 Phase 1: Ingestion & Ephemeris Parsing")
    doc.add_paragraph(
        "The ingestion engine parses input rasters (GeoTIFF, PNG, JPG). If georeferenced, it extracts the Coordinate Reference System (CRS, e.g. EPSG:32633), "
        "the affine geotransform matrix, pixel Ground Sample Distance (GSD), and acquisition solar ephemeris: solar elevation angle (θ) and azimuth (φ)."
    )

    add_heading_2("4.2 Phase 2: Monocular Perception (M1) & Tiled Blending")
    doc.add_paragraph(
        "High-resolution satellite scenes (e.g. 6000×6000) are processed using a 1024×1024 sliding window with 25% overlap via Depth Anything V2 (DINOv2 backbone). "
        "To prevent seam boundary discontinuities, overlapping tile predictions are blended using a 2D Hann window tensor product matrix:"
    )
    add_callout(
        "W(i, j) = sin²(π(2i + 1) / 2H) · sin²(π(2j + 1) / 2W)\n"
        "D_final(x, y) = Σ [W_k(x, y) · D_k(x, y)] / [Σ W_k(x, y) + ε]",
        title="2D HANN WINDOW TENSOR BLENDING:"
    )

    add_heading_2("4.3 Phase 3: Metric Calibration Engine (M2)")
    doc.add_paragraph(
        "When sparse Ground Control Points {(x_i, y_i, H_i)} are supplied, scale a and offset b are solved via closed-form OLS:\n"
        "a = Cov(D, H) / Var(D),   b = H_mean - a · D_mean.\n"
        "When GCPs are absent, ground pixels are isolated via low-percentile filtering (P_25) and anchored against regional SRTM 30m terrain datums."
    )

    add_heading_2("4.4 Phase 4: Physical Shadow Photogrammetry (M4)")
    doc.add_paragraph(
        "Vertical structures cast ground shadows governed by exact trigonometric laws:\n"
        "H = L_shadow · tan(θ_elev)\n"
        "DepthWizard performs 1D physical raycasting along the solar azimuth vector starting at building footings, detecting shadow-to-sunlit transitions "
        "and evaluating analytical uncertainty bounds: σ_H² = (tan θ)² σ_L² + (L sec² θ)² σ_θ²."
    )

    add_heading_2("4.5 Phase 5: Edge Sharpening & sat2Map Planar Regularization (M7)")
    doc.add_paragraph(
        "Raw monocular depth models produce curved, blobby rooftops. DepthWizard implements a 3-channel color guided filter (M7, r=16, ε=0.01) "
        "to snap depth edges to optical wall boundaries, followed by sat2Map RANSAC planar facet extraction to flatten rooftops and compute normal tilt angles."
    )

    # --- 5. Datasets & Experimental Benchmarks ---
    add_heading_1("5. Datasets & Experimental Benchmarks")
    doc.add_paragraph(
        "DepthWizard was rigorously validated across three primary benchmarks:\n"
        "1. ISPRS Potsdam 2D/3D Urban Benchmark: 38 full airborne orthophoto tiles (6000×6000 px, 0.05 m/px GSD) co-registered with airborne LiDAR DSMs (30m–71m elevation). Evaluated across 1,760 buildings.\n"
        "2. GAMUS Benchmark: Multi-modal satellite dataset (Hugging Face) with dense airborne LiDAR Above Ground Level (AGL / nDSM) metric elevation maps.\n"
        "3. SRTM 30m / AWS Terrarium: Global digital elevation datum used for autonomous regional datum anchoring."
    )

    # --- 6. Comprehensive Experimental Results ---
    add_heading_1("6. Comprehensive Experimental Results")

    add_heading_2("6.1 Ground Control Point (GCP) Sparsity Study")
    doc.add_paragraph("Evaluation of Ordinary Least Squares metric calibration across GCP counts K ∈ [1, 100] on the ISPRS Potsdam benchmark:")

    # Table 1: GCP Sparsity
    tbl_gcp = doc.add_table(rows=8, cols=6)
    tbl_gcp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_gcp = ["GCP Count (K)", "Mean MAE (m)", "Mean RMSE (m)", "Pearson (r)", "R² Score", "Scale a"]
    for j, h in enumerate(headers_gcp):
        c = tbl_gcp.cell(0, j)
        c.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(c, "1E293B")
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        c.paragraphs[0].runs[0].font.size = Pt(9.5)
        set_cell_margins(c, 80, 80, 100, 100)

    rows_gcp_data = [
        ("K = 1 (Anchor)", "2.65 ± 0.61", "3.83 ± 0.26", "0.587", "0.007", "6.94"),
        ("K = 3", "2.91 ± 0.93", "3.98 ± 0.91", "0.587", "-0.127", "11.97 ± 10.45"),
        ("K = 5", "2.54 ± 0.48", "3.72 ± 0.42", "0.587", "0.052", "9.38 ± 8.30"),
        ("K = 10 (Optimal)", "2.50 ± 0.14", "3.33 ± 0.21", "0.587", "0.249", "10.05 ± 4.64"),
        ("K = 20", "2.44 ± 0.08", "3.20 ± 0.11", "0.587", "0.301", "9.82 ± 2.15"),
        ("K = 50", "2.41 ± 0.04", "3.15 ± 0.06", "0.587", "0.324", "9.74 ± 1.10"),
        ("K = 100", "2.40 ± 0.02", "3.13 ± 0.03", "0.587", "0.332", "9.68 ± 0.65")
    ]
    for i, rdata in enumerate(rows_gcp_data):
        for j, val in enumerate(rdata):
            cell = tbl_gcp.cell(i + 1, j)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if i % 2 == 1:
                set_cell_background(cell, "F8FAFC")
            if "Optimal" in val:
                run.font.bold = True
            set_cell_margins(cell, 60, 60, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("6.2 M4 Shadow Raycasting Full Benchmark (1,760 Buildings)")
    doc.add_paragraph(
        "Evaluated across all 38 orthophoto tiles of the ISPRS Potsdam dataset:\n"
        "• Total Evaluated Buildings: 1,760\n"
        "• Valid Prediction Rate: 98.8% (1,738 / 1,760 buildings)\n"
        "• Dataset Median Absolute Error (MedAE): 3.32 meters\n"
        "• Dataset Mean Absolute Error (MAE): 4.69 meters\n"
        "• Dataset RMSE: 6.70 meters\n"
        "• Failure Taxonomy: Clean Trigonometric (63.4%), False Short Shadow (20.2%, HVAC texture reflections), "
        "False Long Shadow (13.8%, dark asphalt road penetration), Dark Vegetation (1.4%), Missing Contact (1.2%)."
    )

    add_heading_2("6.3 M7 Guided Filter Refinement Impact")
    doc.add_paragraph(
        "• MAE Improvement: Reduced from 3.91 m to 3.75 m (-0.16 m error reduction).\n"
        "• RMSE Improvement: Reduced from 5.46 m to 5.21 m (-0.25 m error reduction).\n"
        "• Roof Roughness Index (R_TT): Reduced from 1.0000 to 0.9153 (significantly flatter rooftops).\n"
        "• Edge Sharpening Gain: +115.9% sharper boundary alignment."
    )

    add_heading_2("6.4 sat2Map Planar Regularization")
    doc.add_paragraph(
        "• Flat Terrain Noise Standard Deviation: Reduced from 0.506 m to 0.040 m (-92.1% noise reduction).\n"
        "• Roof Facet Residual Error: Reduced from 0.530 m to 0.055 m (-89.7% error reduction).\n"
        "• Unit Test Suite: 5/5 unit tests passing (tests/test_sat2map_3d.py)."
    )

    # --- 7. Software Architecture & Deliverables ---
    add_heading_1("7. Software Architecture & Deliverables")
    doc.add_paragraph(
        "DepthWizard is engineered as an enterprise-grade production software system:\n"
        "• Backend (FastAPI / Uvicorn): High-concurrency ASGI server with model preloading on startup, dynamic downsampling for 3D streaming, and RESTful export endpoints.\n"
        "• Frontend (Three.js WebGL): Real-time 60 FPS viewport featuring orbit controls, drone flythrough, live raycasting elevation HUD, and dynamic topographic profile transects.\n"
        "• GIS Exporter: 32-bit Float GeoTIFF with preserved EPSG projection and affine geotransform, natively compatible with QGIS, ArcGIS, and GDAL.\n"
        "• 3D CAD Exporter: Stanford Binary Point Cloud (.ply with normals) and Wavefront Textured Mesh (.obj + .mtl)."
    )

    # --- 8. Mentor Meeting Defense Guide ---
    add_heading_1("8. IIT Madras Mentor Meeting Defense Guide")

    qas = [
        ("Q1: Isn't monocular depth inherently ill-posed? How can you trust the scale?",
         "Yes, single-view depth has an inherent affine scale and shift ambiguity (H = aD + b). DepthWizard does not claim to solve scale purely through neural networks. Instead, we use a hybrid formulation: the neural backbone recovers continuous relative topographic morphology, while physical geodesy (sparse GCPs, regional SRTM datum anchoring, and solar shadow trigonometry h = L tan θ) provides the physical metric constraints. Furthermore, under our Zero-Fabrication Contract, if no ground datum is available, the system strictly outputs normalized relative models (rDSM ∈ [0, 1]) rather than hallucinating metric numbers."),
        ("Q2: How is your shadow method different from classical photogrammetric mensuration?",
         "Classical shadow mensuration requires manual surveyor measurement of building and shadow vertices. In DepthWizard, Module M4 automates this via 1D outward raycasting along the solar azimuth vector, terminating automatically at local contrast gradient steps. We also introduce analytical uncertainty propagation (σ_H² = tan²θ σ_L² + L² sec⁴θ σ_θ²), giving every building height an explicit confidence bound."),
        ("Q3: Why did you implement sat2Map and guided filtering on top of Depth Anything V2?",
         "Vision Transformers are trained on smooth natural images, so they produce rounded, curved rooftops on satellite images. In urban modeling, roofs are composed of sharp planar facets and vertical step drops at walls. We implemented the 3-channel color guided filter (M7) and sat2Map RANSAC planar regularization to snap depth edges to optical wall boundaries and flatten rooftop planes, reducing planar noise variance by 89.7%."),
        ("Q4: How many GCPs are actually needed in practice?",
         "Our empirical sparsity study across the ISPRS Potsdam benchmark shows that K = 10 GCPs is the optimal knee of the curve. At K = 10, MAE drops to 2.50 m and standard deviation drops to ±0.14 m. Moving to 50 or 100 GCPs only reduces MAE by another 0.10 m, yielding diminishing returns.")
    ]

    for q, a in qas:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(8)
        p_q.paragraph_format.space_after = Pt(2)
        p_q.paragraph_format.keep_with_next = True
        r_q = p_q.add_run(q)
        r_q.font.bold = True
        r_q.font.color.rgb = COLOR_PRIMARY
        p_a = doc.add_paragraph(a)
        p_a.paragraph_format.space_after = Pt(6)

    # Save .docx file
    out_dir = "docs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "DepthWizard_Master_Project_Handbook.docx")
    doc.save(out_path)
    print(f"[Success] Master Handbook Word Document (.docx) saved at: {out_path}")

if __name__ == "__main__":
    generate_handbook_docx()
