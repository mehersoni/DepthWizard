"""
DepthWizard Unified Server for Hugging Face Spaces (Gradio SDK + Native Custom 3D WebGL Dashboard)
"""

import os
import io
import json
import base64
import tempfile
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import gradio as gr
import spaces
import uvicorn

from process_image import process_image, export_dsm, export_slope
from depth.depth_model import load_model

GLOBAL_MODEL = None
GLOBAL_PROCESSOR = None
GLOBAL_DEVICE = None
EXPORT_CACHE: Dict[str, Dict[str, Any]] = {}

def get_model():
    global GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE
    if GLOBAL_MODEL is None:
        print("[API Server] Preloading Depth Anything V2 model...", flush=True)
        GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE = load_model()
        print(f"[API Server] Model ready on device: {GLOBAL_DEVICE}", flush=True)
    return GLOBAL_MODEL, GLOBAL_PROCESSOR, GLOBAL_DEVICE

@spaces.GPU
def run_depth_inference(
    path: str,
    gcps: Optional[list] = None,
    dem_path: Optional[str] = None,
    use_shadows: bool = True,
    a_prior: Optional[float] = None,
    lambda_prior: float = 0.0,
    terrain_percentile: float = 25.0,
    calibration_method: str = "linear"
) -> Dict[str, Any]:
    """Execute ML inference inside ZeroGPU isolated worker."""
    model, processor, device = get_model()
    return process_image(
        path=path,
        gcps=gcps,
        dem_path=dem_path,
        use_shadows=use_shadows,
        model=model,
        processor=processor,
        device=device,
        a_prior=a_prior,
        lambda_prior=lambda_prior,
        terrain_percentile=terrain_percentile,
        calibration_method=calibration_method
    )


def sanitize_for_json(obj: Any) -> Any:
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
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32)


def encode_rgb_to_base64_jpeg(rgb: np.ndarray, target_size: int = 256, quality: int = 85) -> str:
    pil_img = Image.fromarray(rgb)
    resized_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def encode_colormap_to_base64_jpeg(
    arr: np.ndarray,
    cmap_name: str = "turbo",
    target_size: int = 256,
    invert: bool = False,
    quality: int = 85,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> str:
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        arr_norm = np.zeros_like(arr, dtype=np.float32)
    else:
        v_min = vmin if vmin is not None else float(np.nanmin(arr[valid_mask]))
        v_max = vmax if vmax is not None else float(np.nanmax(arr[valid_mask]))
        span = v_max - v_min
        if span < 1e-7:
            arr_norm = np.zeros_like(arr, dtype=np.float32)
        else:
            arr_norm = np.clip((arr - v_min) / span, 0.0, 1.0)

    if invert:
        arr_norm = 1.0 - arr_norm

    cmap = cm.get_cmap(cmap_name)
    colored_rgba = cmap(arr_norm)
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

    pil_img = Image.fromarray(colored_rgb)
    resized_img = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


html_dashboard_path = os.path.join(os.path.dirname(__file__), "demo.html")
if not os.path.isfile(html_dashboard_path):
    html_dashboard_path = os.path.join(os.path.dirname(__file__), "m6_dashboard.html")

with open(html_dashboard_path, "r", encoding="utf-8") as f:
    custom_ui_html = f.read()

custom_css = """
body, html { margin: 0; padding: 0; min-height: 100vh; background: #08080a; }
.gradio-container { max-width: 100% !important; margin: 0 !important; padding: 0 !important; min-height: 100vh !important; background: #08080a; }
#custom-iframe-wrap, #custom-iframe-wrap iframe { width: 100% !important; min-height: 100vh !important; height: 100vh !important; border: none; }
"""

# -----------------------------------------------------------------------------
# FastAPI API Server with Integrated /process & /export routes
# -----------------------------------------------------------------------------
app = FastAPI(title="DepthWizard Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@spaces.GPU
def predict_gpu(image_path: str) -> str:
    """ZeroGPU entrypoint."""
    return image_path

with gr.Blocks(title="DepthWizard — 3D Elevation Platform", css=custom_css, fill_height=True) as demo:
    _hidden_in = gr.Textbox(visible=False)
    _hidden_out = gr.Textbox(visible=False)
    _hidden_btn = gr.Button(value="process", visible=False)
    _hidden_btn.click(fn=predict_gpu, inputs=[_hidden_in], outputs=[_hidden_out])

    gr.HTML(
        f"""
        <div id="custom-iframe-wrap" style="width:100%; height:100vh; overflow:auto;">
            <iframe srcdoc="{custom_ui_html.replace('"', '&quot;')}" style="width:100%; min-height:100vh; height:100%; border:none; display:block;"></iframe>
        </div>
        """
    )


@app.get("/health")
@app.post("/health")
def health_check():
    return {"status": "online", "service": "DepthWizard Engine"}


@app.get("/demo.html", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def serve_demo():
    html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "m6_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/index.html", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def serve_landing():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/export/{session_id}")
def download_export(session_id: str):
    if session_id not in EXPORT_CACHE:
        raise HTTPException(status_code=404, detail="Export session not found.")
    item = EXPORT_CACHE[session_id]
    filepath = item["path"]
    filename = item["filename"]
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Export file missing.")
    return FileResponse(path=filepath, filename=filename, media_type="application/octet-stream")


@app.post("/process")
@app.post("/api/process")
@app.post("/gradio/process")
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
    calibration_method: str = Form("linear")
):
    model, processor, device = get_model()

    parsed_gcps = None
    if gcps:
        try:
            parsed_gcps = json.loads(gcps)
        except Exception:
            pass

    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower() or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

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
        result = await run_in_threadpool(
            run_depth_inference,
            path=temp_path,
            gcps=parsed_gcps,
            dem_path=temp_dem_path,
            use_shadows=use_shadows,
            a_prior=a_prior,
            lambda_prior=lambda_prior,
            terrain_percentile=terrain_percentile,
            calibration_method=calibration_method
        )

        h_orig, w_orig = result["height_map"].shape
        v_size = min(int(visual_size), 512)

        h_sub = downsample_array(result["height_map"], target_size=v_size)
        s_sub = downsample_array(result["slope_map"], target_size=v_size)
        c_sub = downsample_array(result["confidence_map"], target_size=v_size)

        rgb_b64 = encode_rgb_to_base64_jpeg(result["rgb"], target_size=v_size)
        depth_b64 = encode_colormap_to_base64_jpeg(result["depth_map"], cmap_name="turbo", target_size=v_size)
        dsm_b64 = encode_colormap_to_base64_jpeg(result["height_map"], cmap_name="turbo", target_size=v_size)
        slope_b64 = encode_colormap_to_base64_jpeg(result["slope_map"], cmap_name="magma", target_size=v_size, vmin=0.0, vmax=45.0)
        conf_b64 = encode_colormap_to_base64_jpeg(result["confidence_map"], cmap_name="viridis", target_size=v_size, vmin=0.0, vmax=1.0)

        error_b64 = None
        val_metrics = None
        if result.get("error_map") is not None:
            err_arr = result["error_map"]
            error_b64 = encode_colormap_to_base64_jpeg(np.abs(err_arr), cmap_name="coolwarm", target_size=v_size)
            valid_err = err_arr[np.isfinite(err_arr)]
            if len(valid_err) > 0:
                val_metrics = {
                    "mae": float(np.mean(np.abs(valid_err))),
                    "rmse": float(np.sqrt(np.mean(valid_err ** 2))),
                    "max_error": float(np.max(np.abs(valid_err))),
                    "count": int(len(valid_err))
                }

        h_finite = result["height_map"][np.isfinite(result["height_map"])]
        h_min = float(np.min(h_finite)) if len(h_finite) > 0 else 0.0
        h_max = float(np.max(h_finite)) if len(h_finite) > 0 else 1.0

        s_finite = result["slope_map"][np.isfinite(result["slope_map"])]
        s_min = float(np.min(s_finite)) if len(s_finite) > 0 else 0.0
        s_max = float(np.max(s_finite)) if len(s_finite) > 0 else 90.0

        c_finite = result["confidence_map"][np.isfinite(result["confidence_map"])]
        c_mean = float(np.mean(c_finite)) if len(c_finite) > 0 else 1.0

        calibrated = bool(result.get("calibrated", False))
        georeferenced = bool(result.get("georeferenced", False))
        mode = str(result.get("mode", "rdsm"))
        height_unit = str(result.get("height_unit", "m" if calibrated else "relative"))

        mid_y = v_size // 2
        profile_pts = []
        for x_idx in range(v_size):
            val_h = float(h_sub[mid_y, x_idx])
            profile_pts.append({
                "x": x_idx,
                "elevation": None if np.isnan(val_h) else val_h
            })

        h_list = np.where(np.isnan(h_sub), None, h_sub).tolist()
        s_list = np.where(np.isnan(s_sub), None, s_sub).tolist()
        c_list = np.where(np.isnan(c_sub), None, c_sub).tolist()

        session_id = str(tempfile.NamedTemporaryFile().name).split(os.sep)[-1][:8]
        os.makedirs("outputs/export", exist_ok=True)
        export_ext = ".tif" if georeferenced else ".png"
        export_filename = f"depthwizard_{session_id}_{'dsm' if calibrated else 'rdsm'}{export_ext}"
        export_filepath = os.path.join("outputs/export", export_filename)
        export_dsm(result, export_filepath)

        EXPORT_CACHE[session_id] = {
            "path": export_filepath,
            "filename": export_filename
        }

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
            "error_preview": error_b64,
            "validation": val_metrics,
            "slope_map": s_list,
            "confidence_map": c_list,
            "slope_min": s_min,
            "slope_max": s_max,
            "confidence_mean": c_mean,
            "elevation_profile": profile_pts,
            "crs": str(result["crs"]) if result["crs"] is not None else None,
            "export_url": f"/export/{session_id}",
            "metadata": result["metadata"]
        }

        return JSONResponse(content=sanitize_for_json(response_payload))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(exc)}", "detail": str(exc)}
        )
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dem_path and os.path.exists(temp_dem_path) and temp_dem_path != dem_path:
                os.remove(temp_dem_path)
        except Exception:
            pass


# Mount all FastAPI routes onto Gradio's internal FastAPI app
demo.app.include_router(app.router)

# Also ensure /demo.html and / serve the custom Three.js interface
@demo.app.get("/", response_class=HTMLResponse)
@demo.app.get("/demo.html", response_class=HTMLResponse)
@demo.app.get("/demo", response_class=HTMLResponse)
def serve_root_demo():
    html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if not os.path.isfile(html_path):
        html_path = os.path.join(os.path.dirname(__file__), "m6_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    get_model()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        ssr_mode=False
    )
