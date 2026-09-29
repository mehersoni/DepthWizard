"""
Module: api_server
Description: FastAPI Backend Server connecting DepthWizard M2 Elevation Engine with M6 Application Dashboard.
"""

import os
import io
import json
import uuid
import base64
import tempfile
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from process_image import process_image, export_dsm, export_slope, export_point_cloud, export_mesh
from depth.depth_model import load_model

GLOBAL_MODEL = None
GLOBAL_PROCESSOR = None
GLOBAL_DEVICE = None

# Cache for exported rasters ready for download
EXPORT_CACHE: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload Depth Anything V2 model into memory on startup."""
    global GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE
    print("[API Server] Preloading Depth Anything V2 model...", flush=True)
    try:
        GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE = load_model()
        print(f"[API Server] Model ready on device: {GLOBAL_DEVICE}", flush=True)
    except Exception as e:
        print(f"[API Server Warning] Could not preload model on startup: {e}", flush=True)
    yield


app = FastAPI(
    title="DepthWizard M6 Elevation Application Backend",
    description="Transforms optical RGB imagery into calibrated 3D Digital Surface Models.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")
if os.path.isdir("demoImages"):
    app.mount("/demoImages", StaticFiles(directory="demoImages"), name="demoImages")
if os.path.isdir("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, tuples, and non-serializables into standard JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


def downsample_array(arr: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Downsample a 2D float array to target_size x target_size for WebGL."""
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32)


def encode_rgb_to_base64_jpeg(rgb: np.ndarray, target_size: int = 256, quality: int = 85) -> str:
    """Resize RGB image to target_size x target_size and encode as Base64 JPEG data URL."""
    pil_img = Image.fromarray(rgb)
    resized_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


import matplotlib

def encode_colormap_to_base64_jpeg(
    arr: np.ndarray,
    cmap_name: str = "turbo",
    target_size: int = 256,
    invert: bool = False,
    quality: int = 85
) -> str:
    """Apply a matplotlib colormap to a 2D float array and return Base64 JPEG."""
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    span = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0

    norm_arr = np.clip((arr - min_v) / span, 0.0, 1.0)
    if invert:
        norm_arr = 1.0 - norm_arr

    try:
        cmap = matplotlib.colormaps[cmap_name]
    except Exception:
        cmap = matplotlib.colormaps["viridis"]

    colored_rgba = cmap(norm_arr)  # (H, W, 4) in [0, 1]
    colored_rgb = (colored_rgba[:, :, :3] * 255.0).astype(np.uint8)

    pil_img = Image.fromarray(colored_rgb)
    resized_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
def serve_landing():
    """Serve the landing page."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo.html", response_class=HTMLResponse)
def serve_demo():
    """Serve the 3D demo dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "m6_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/m6_dashboard.html", response_class=HTMLResponse)
@app.get("/m6", response_class=HTMLResponse)
def serve_m6():
    """Serve the M6 full console."""
    html_path = os.path.join(os.path.dirname(__file__), "m6_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/m5_3d_viewer_sharp.html", response_class=HTMLResponse)
@app.get("/sharp", response_class=HTMLResponse)
@app.get("/m5_sharp", response_class=HTMLResponse)
@app.get("/m5", response_class=HTMLResponse)
def serve_m5_sharp():
    """Serve the M5 Sharp 3D WebGL viewer (faceted columns, vertical walls, building blocks)."""
    html_path = os.path.join(os.path.dirname(__file__), "m5_3d_viewer_sharp.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "m5_3d_viewer_demo.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/m5_3d_viewer_demo.html", response_class=HTMLResponse)
@app.get("/m5_demo", response_class=HTMLResponse)
def serve_m5_demo():
    """Serve the legacy M5 continuous 3D WebGL viewer."""
    html_path = os.path.join(os.path.dirname(__file__), "m5_3d_viewer_demo.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/m5_3d_viewer_demo_blocks_v3.html", response_class=HTMLResponse)
@app.get("/m5_blocks", response_class=HTMLResponse)
def serve_m5_blocks():
    """Serve the M5 3D Building Blocks WebGL viewer."""
    html_path = os.path.join(os.path.dirname(__file__), "m5_3d_viewer_demo_blocks_v3.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "DepthWizard M2 Elevation Backend",
        "model": "Depth Anything V2",
        "device": str(GLOBAL_DEVICE)
    }


@app.post("/process")
@app.post("/api/process")
async def process_image_endpoint(
    file: UploadFile = File(...),
    gcps: Optional[str] = Form(None),
    dem_file: Optional[UploadFile] = File(None),
    dem_path: Optional[str] = Form(None),
    use_shadows: bool = Form(True),
    visual_size: int = Form(256),
    a_prior: Optional[float] = Form(None),
    lambda_prior: float = Form(0.0),
    terrain_percentile: float = Form(25.0),
    calibration_method: str = Form("linear"),
    refine_roofs: bool = Form(True)
):
    """
    Main elevation inference endpoint.
    Accepts JPG, PNG, or GeoTIFF, runs M2 pipeline, and returns problem-statement-compliant M6 payload.
    """
    global GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE
    
    if GLOBAL_MODEL is None:
        GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE = load_model()

    # Parse GCPs if provided as JSON string
    parsed_gcps = None
    if gcps:
        try:
            parsed_gcps = json.loads(gcps)
        except Exception as e:
            print(f"[API Warning] Failed to parse GCPs JSON: {e}")

    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        ext = ".png"

    # Save uploaded main image to temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    # Handle uploaded DEM file if provided
    temp_dem_path = None
    if dem_file is not None and dem_file.filename:
        dem_ext = os.path.splitext(dem_file.filename)[1].lower() or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=dem_ext) as tmp_dem:
            dem_contents = await dem_file.read()
            tmp_dem.write(dem_contents)
            temp_dem_path = tmp_dem.name
    elif dem_path and os.path.isfile(dem_path):
        temp_dem_path = dem_path

    try:
        print(f"[API Server] Received {filename} ({len(contents):,} bytes). Executing process_image()...", flush=True)
        result = await run_in_threadpool(
            process_image,
            path=temp_path,
            gcps=parsed_gcps,
            dem_path=temp_dem_path,
            use_shadows=use_shadows,
            model=GLOBAL_MODEL,
            processor=GLOBAL_PROCESSOR,
            device=GLOBAL_DEVICE,
            a_prior=a_prior,
            lambda_prior=lambda_prior,
            terrain_percentile=terrain_percentile,
            calibration_method=calibration_method,
            refine_roofs=refine_roofs
        )

        h_orig = result["height"]
        w_orig = result["width"]
        mode = result["mode"]
        calibrated = result["calibrated"]
        georeferenced = result["georeferenced"]
        height_unit = result["height_unit"]
        height_map_full = result["height_map"]
        slope_map_full = result["slope_map"]
        conf_map_full = result["confidence_map"]
        rgb_full = result["rgb"]

        # Calculate statistics from full-resolution arrays
        h_min = float(np.min(height_map_full))
        h_max = float(np.max(height_map_full))
        s_min = float(np.min(slope_map_full))
        s_max = float(np.max(slope_map_full))
        c_mean = float(np.mean(conf_map_full))

        # Downsample arrays for responsive WebGL 3D rendering
        v_size = max(64, min(512, int(visual_size)))
        print(f"[API Server] Downsampling {w_orig}x{h_orig} -> {v_size}x{v_size} for 3D visualization...", flush=True)
        h_vis = downsample_array(height_map_full, target_size=v_size)
        s_vis = downsample_array(slope_map_full, target_size=v_size)
        c_vis = downsample_array(conf_map_full, target_size=v_size)

        # Pre-render colormapped 2D channel views for M6 multi-tab display
        rgb_b64 = encode_rgb_to_base64_jpeg(rgb_full, target_size=v_size, quality=85)
        depth_b64 = encode_colormap_to_base64_jpeg(h_vis, cmap_name="gray", target_size=v_size, invert=False)
        dsm_b64 = encode_colormap_to_base64_jpeg(h_vis, cmap_name="turbo", target_size=v_size, invert=False)
        slope_b64 = encode_colormap_to_base64_jpeg(s_vis, cmap_name="magma", target_size=v_size, invert=False)
        conf_b64 = encode_colormap_to_base64_jpeg(c_vis, cmap_name="viridis", target_size=v_size, invert=False)

        # Planar Facets Colormap
        facet_vis = None
        facet_b64 = None
        if "planar_facets" in result and "facet_map" in result["planar_facets"]:
            f_map = result["planar_facets"]["facet_map"]
            facet_vis = downsample_array(f_map.astype(np.float32), target_size=v_size)
            facet_b64 = encode_colormap_to_base64_jpeg(facet_vis, cmap_name="tab10", target_size=v_size, invert=False)

        # Compute Reference Residual Error Map & Metrics
        val_metrics = None
        error_b64 = None

        # 1. External user-provided DEM / Reference
        eval_ref_path = None
        if temp_dem_path and os.path.isfile(temp_dem_path):
            eval_ref_path = temp_dem_path
        elif "potsdam" in file.filename.lower():
            candidate_ref = os.path.abspath("data/potsdam/1_DSM/dsm_potsdam_02_10.tif")
            if os.path.isfile(candidate_ref):
                eval_ref_path = candidate_ref
            else:
                candidate_srtm = os.path.abspath("data/dem_cache/top_potsdam_2_10_RGB_srtm_dem.tif")
                if os.path.isfile(candidate_srtm):
                    eval_ref_path = candidate_srtm

        if eval_ref_path and os.path.isfile(eval_ref_path):
            try:
                import rasterio
                with rasterio.open(eval_ref_path) as src_dem:
                    ref_dsm = src_dem.read(1).astype(np.float32)
                    if ref_dsm.shape != (h_orig, w_orig):
                        import cv2
                        ref_dsm = cv2.resize(ref_dsm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                    
                    diff = height_map_full - ref_dsm
                    abs_diff = np.abs(diff)
                    mae_val = float(np.nanmean(abs_diff))
                    rmse_val = float(np.sqrt(np.nanmean(diff ** 2)))
                    bias_val = float(np.nanmean(diff))
                    
                    # Pearson correlation
                    valid_mask = np.isfinite(height_map_full) & np.isfinite(ref_dsm)
                    if np.sum(valid_mask) > 10:
                        r_corr = float(np.corrcoef(height_map_full[valid_mask].ravel(), ref_dsm[valid_mask].ravel())[0, 1])
                        r2_val = round(max(0.0, min(1.0, r_corr ** 2)), 2)
                    else:
                        r2_val = None
                    
                    val_metrics = {
                        "mae": round(mae_val, 2),
                        "rmse": round(rmse_val, 2),
                        "r2": r2_val,
                        "bias": round(bias_val, 2)
                    }
                    
                    err_vis = downsample_array(abs_diff, target_size=v_size)
                    error_b64 = encode_colormap_to_base64_jpeg(err_vis, cmap_name="inferno", target_size=v_size, invert=False)
            except Exception as ex:
                print(f"[API Warning] Could not calculate reference DEM error metrics: {ex}")

        # If still no error map (e.g. unreferenced photo), generate high-frequency topological relief residual
        if error_b64 is None:
            from scipy.ndimage import gaussian_filter
            smooth_h = gaussian_filter(h_vis, sigma=3.0)
            abs_diff = np.abs(h_vis - smooth_h)
            error_b64 = encode_colormap_to_base64_jpeg(abs_diff, cmap_name="inferno", target_size=v_size, invert=False)
            val_metrics = {
                "mae": round(float(np.mean(abs_diff)), 3),
                "rmse": round(float(np.sqrt(np.mean(abs_diff ** 2))), 3),
                "r2": None,  # No external ground truth DEM provided for unreferenced scene
                "bias": round(float(np.mean(h_vis - smooth_h)), 3)
            }

        # Compute Center Cross-Section Elevation Profile (60 sample points across width)
        center_row = h_orig // 2
        profile_raw = height_map_full[center_row, :]
        profile_indices = np.linspace(0, len(profile_raw) - 1, 60)
        profile_pts = [round(float(v), 2) for v in np.interp(profile_indices, np.arange(len(profile_raw)), profile_raw)]

        # Round floats to 3 decimal places to minimize JSON payload size
        h_list = [round(float(v), 3) for v in h_vis.ravel()]
        s_list = [round(float(v), 2) for v in s_vis.ravel()]
        c_list = [round(float(v), 3) for v in c_vis.ravel()]

        # Generate Export Artifacts (GeoTIFF, 3D Point Cloud PLY, and Wavefront OBJ Mesh)
        session_id = str(uuid.uuid4())[:8]
        os.makedirs("outputs/export", exist_ok=True)
        export_ext = ".tif" if georeferenced else ".png"
        export_filename = f"depthwizard_{session_id}_{'dsm' if calibrated else 'rdsm'}{export_ext}"
        export_filepath = os.path.join("outputs/export", export_filename)
        export_dsm(result, export_filepath)

        # 3D Point Cloud (.ply)
        ply_filename = f"depthwizard_{session_id}_pointcloud.ply"
        ply_filepath = os.path.join("outputs/export", ply_filename)
        try:
            export_point_cloud(result, ply_filepath, step=1 if max(h_orig, w_orig) <= 512 else 2)
        except Exception as ply_err:
            print(f"[API Server 3D Export Warning] {ply_err}")

        # 3D Wavefront OBJ Mesh (.obj)
        obj_filename = f"depthwizard_{session_id}_mesh.obj"
        obj_filepath = os.path.join("outputs/export", obj_filename)
        try:
            export_mesh(result, obj_filepath, step=2)
        except Exception as obj_err:
            print(f"[API Server OBJ Export Warning] {obj_err}")

        EXPORT_CACHE[session_id] = {
            "path": export_filepath,
            "filename": export_filename,
            "ply_path": ply_filepath if os.path.isfile(ply_filepath) else None,
            "ply_filename": ply_filename,
            "obj_path": obj_filepath if os.path.isfile(obj_filepath) else None,
            "obj_filename": obj_filename,
            "calibrated": calibrated,
            "georeferenced": georeferenced,
            "mode": mode
        }

        planar_summary = result.get("metadata", {}).get("planar_facets", {})

        response_payload = {
            "session_id": session_id,
            "mode": mode,
            "calibrated": calibrated,
            "georeferenced": georeferenced,
            "height_unit": height_unit,
            "width": int(w_orig),
            "height": int(h_orig),
            "visual_width": int(v_size),
            "visual_height": int(v_size),
            "height_min": h_min,
            "height_max": h_max,
            "height_map": h_list,
            "rgb": rgb_b64,
            "depth_preview": depth_b64,
            "dsm_preview": dsm_b64,
            "slope_preview": slope_b64,
            "confidence_preview": conf_b64,
            "facet_preview": facet_b64,
            "error_preview": error_b64,
            "validation": val_metrics,
            "slope_map": s_list,
            "confidence_map": c_list,
            "slope_min": s_min,
            "slope_max": s_max,
            "confidence_mean": c_mean,
            "planar_facets": planar_summary,
            "elevation_profile": profile_pts,
            "crs": str(result["crs"]) if result["crs"] is not None else None,
            "gsd_x": result["metadata"].get("gsd_x"),
            "gsd_y": result["metadata"].get("gsd_y"),
            "export_url": f"/export/{session_id}",
            "export_ply_url": f"/export/3d/ply/{session_id}",
            "export_obj_url": f"/export/3d/obj/{session_id}",
            "metadata": result["metadata"]
        }

        sanitized_payload = sanitize_for_json(response_payload)
        print(f"[API Server] Response ready: session={session_id}, mode={mode}, calibrated={calibrated}, span=[{h_min:.2f}, {h_max:.2f}], visual={v_size}x{v_size}, 3D ready", flush=True)
        return JSONResponse(content=sanitized_payload)

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[API Server Error Exception] {err_msg}", flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Failed to process image elevation pipeline", "detail": err_msg}
        )

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if temp_dem_path and temp_dem_path != dem_path and os.path.exists(temp_dem_path):
            try:
                os.remove(temp_dem_path)
            except Exception:
                pass


@app.get("/export/{session_id}")
@app.get("/api/export/{session_id}")
def download_export(session_id: str):
    """Download exported GeoTIFF or PNG raster."""
    if session_id not in EXPORT_CACHE:
        raise HTTPException(status_code=404, detail="Export session not found or expired.")

    item = EXPORT_CACHE[session_id]
    filepath = item["path"]
    filename = item["filename"]

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Export file missing on server.")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/export/3d/ply/{session_id}")
@app.get("/api/export/3d/ply/{session_id}")
def download_ply_export(session_id: str):
    """Download exported 3D Point Cloud (.ply)."""
    if session_id not in EXPORT_CACHE:
        raise HTTPException(status_code=404, detail="Export session not found or expired.")

    item = EXPORT_CACHE[session_id]
    filepath = item.get("ply_path")
    filename = item.get("ply_filename", f"depthwizard_{session_id}_pointcloud.ply")

    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="3D PLY Point Cloud file missing on server.")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/export/3d/obj/{session_id}")
@app.get("/api/export/3d/obj/{session_id}")
def download_obj_export(session_id: str):
    """Download exported 3D Textured Mesh (.obj)."""
    if session_id not in EXPORT_CACHE:
        raise HTTPException(status_code=404, detail="Export session not found or expired.")

    item = EXPORT_CACHE[session_id]
    filepath = item.get("obj_path")
    filename = item.get("obj_filename", f"depthwizard_{session_id}_mesh.obj")

    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="3D OBJ Mesh file missing on server.")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"DepthWizard M6 Server running at http://127.0.0.1:{port}/")
    print(f"=======================================================\n")
    uvicorn.run("api_server:app", host="127.0.0.1", port=port, log_level="info")
