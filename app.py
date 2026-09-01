"""
DepthWizard Unified Server for Hugging Face Spaces (Gradio SDK + Native Custom 3D WebGL Dashboard)

Architecture:
  - Custom routes (/process, /demo.html, /health, /export) are handled by a Starlette
    BaseHTTPMiddleware that intercepts requests BEFORE Gradio's internal router.
  - This bypasses the Gradio 5 route-table shadow problem where @demo.app.post() routes
    are registered but never reached due to Gradio's catch-all Mount.
  - ZeroGPU compliance: demo.queue().launch() is used; @spaces.GPU decorates the GPU worker.
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

import gradio as gr
import spaces

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, FileResponse, Response
from starlette.concurrency import run_in_threadpool

from process_image import process_image, export_dsm, export_slope
from depth.depth_model import load_model

# ---------------------------------------------------------------------------
# Global model cache & export store
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# JSON / array helpers
# ---------------------------------------------------------------------------

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
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


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
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


# ---------------------------------------------------------------------------
# Core process handler (called from middleware)
# ---------------------------------------------------------------------------

async def _handle_process(request: Request) -> Response:
    """Handle multipart /process POST — runs inference and returns JSON."""
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad multipart form: {e}"})

    file_field = form.get("file")
    if file_field is None:
        return JSONResponse(status_code=400, content={"error": "Missing 'file' field in form"})

    # Parse form parameters
    def _bool(v, default=True):
        if v is None:
            return default
        return str(v).lower() not in ("false", "0", "no")

    def _float_or_none(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    def _float(v, default):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _int(v, default):
        try:
            return int(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    gcps_raw = form.get("gcps")
    dem_file_field = form.get("dem_file")
    dem_path_raw = form.get("dem_path")
    use_shadows = _bool(form.get("use_shadows"), True)
    visual_size = _int(form.get("visual_size"), 256)
    a_prior = _float_or_none(form.get("a_prior"))
    lambda_prior = _float(form.get("lambda_prior"), 0.0)
    terrain_percentile = _float(form.get("terrain_percentile"), 25.0)
    calibration_method = str(form.get("calibration_method") or "linear")

    parsed_gcps = None
    if gcps_raw:
        try:
            parsed_gcps = json.loads(gcps_raw)
        except Exception:
            pass

    filename = getattr(file_field, "filename", None) or "upload.png"
    ext = os.path.splitext(filename)[1].lower() or ".png"

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file_field.read()
        tmp.write(contents)
        temp_path = tmp.name

    temp_dem_path = None
    if dem_file_field is not None and getattr(dem_file_field, "filename", None):
        dem_ext = os.path.splitext(dem_file_field.filename)[1].lower() or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=dem_ext) as tmp_dem:
            dem_contents = await dem_file_field.read()
            tmp_dem.write(dem_contents)
            temp_dem_path = tmp_dem.name
    elif dem_path_raw and os.path.isfile(dem_path_raw):
        temp_dem_path = dem_path_raw

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

        rgb_b64    = encode_rgb_to_base64_jpeg(result["rgb"], target_size=v_size)
        depth_b64  = encode_colormap_to_base64_jpeg(result["depth_map"],  cmap_name="turbo",  target_size=v_size)
        dsm_b64    = encode_colormap_to_base64_jpeg(result["height_map"], cmap_name="turbo",  target_size=v_size)
        slope_b64  = encode_colormap_to_base64_jpeg(result["slope_map"],  cmap_name="magma",  target_size=v_size, vmin=0.0, vmax=45.0)
        conf_b64   = encode_colormap_to_base64_jpeg(result["confidence_map"], cmap_name="viridis", target_size=v_size, vmin=0.0, vmax=1.0)

        error_b64 = None
        val_metrics = None
        if result.get("error_map") is not None:
            err_arr = result["error_map"]
            error_b64 = encode_colormap_to_base64_jpeg(np.abs(err_arr), cmap_name="coolwarm", target_size=v_size)
            valid_err = err_arr[np.isfinite(err_arr)]
            if len(valid_err) > 0:
                val_metrics = {
                    "mae":       float(np.mean(np.abs(valid_err))),
                    "rmse":      float(np.sqrt(np.mean(valid_err ** 2))),
                    "max_error": float(np.max(np.abs(valid_err))),
                    "count":     int(len(valid_err))
                }

        h_finite = result["height_map"][np.isfinite(result["height_map"])]
        h_min = float(np.min(h_finite)) if len(h_finite) > 0 else 0.0
        h_max = float(np.max(h_finite)) if len(h_finite) > 0 else 1.0

        s_finite = result["slope_map"][np.isfinite(result["slope_map"])]
        s_min = float(np.min(s_finite)) if len(s_finite) > 0 else 0.0
        s_max = float(np.max(s_finite)) if len(s_finite) > 0 else 90.0

        c_finite = result["confidence_map"][np.isfinite(result["confidence_map"])]
        c_mean = float(np.mean(c_finite)) if len(c_finite) > 0 else 1.0

        calibrated    = bool(result.get("calibrated", False))
        georeferenced = bool(result.get("georeferenced", False))
        mode          = str(result.get("mode", "rdsm"))
        height_unit   = str(result.get("height_unit", "m" if calibrated else "relative"))

        mid_y = v_size // 2
        profile_pts = []
        for x_idx in range(v_size):
            val_h = float(h_sub[mid_y, x_idx])
            profile_pts.append({"x": x_idx, "elevation": None if np.isnan(val_h) else val_h})

        h_list = np.where(np.isnan(h_sub), None, h_sub).tolist()
        s_list = np.where(np.isnan(s_sub), None, s_sub).tolist()
        c_list = np.where(np.isnan(c_sub), None, c_sub).tolist()

        session_id = str(tempfile.NamedTemporaryFile().name).split(os.sep)[-1][:8]
        os.makedirs("outputs/export", exist_ok=True)
        export_ext      = ".tif" if georeferenced else ".png"
        export_filename = f"depthwizard_{session_id}_{'dsm' if calibrated else 'rdsm'}{export_ext}"
        export_filepath = os.path.join("outputs/export", export_filename)
        export_dsm(result, export_filepath)
        EXPORT_CACHE[session_id] = {"path": export_filepath, "filename": export_filename}

        payload = {
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

        return JSONResponse(content=sanitize_for_json(payload))

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
            if temp_dem_path and os.path.exists(temp_dem_path) and temp_dem_path != (dem_path_raw or ""):
                os.remove(temp_dem_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Middleware — intercepts BEFORE Gradio's router
# ---------------------------------------------------------------------------

_HTML_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_HTML = os.path.join(_HTML_DIR, "demo.html")
if not os.path.isfile(_DEMO_HTML):
    _DEMO_HTML = os.path.join(_HTML_DIR, "m6_dashboard.html")


class DepthWizardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()

        # ── CORS preflight ────────────────────────────────────────────────
        if method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )

        # ── Health check ─────────────────────────────────────────────────
        if path == "/health":
            return JSONResponse(
                {"status": "online", "service": "DepthWizard Engine"},
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # ── Serve demo.html ───────────────────────────────────────────────
        if path in ("/demo.html", "/demo") and method == "GET":
            try:
                with open(_DEMO_HTML, "r", encoding="utf-8") as f:
                    html = f.read()
                return HTMLResponse(content=html, headers={"Access-Control-Allow-Origin": "*"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        # ── Export download ───────────────────────────────────────────────
        if path.startswith("/export/") and method == "GET":
            session_id = path[len("/export/"):].strip("/")
            if session_id not in EXPORT_CACHE:
                return JSONResponse(status_code=404, content={"error": "Export session not found"})
            item = EXPORT_CACHE[session_id]
            if not os.path.isfile(item["path"]):
                return JSONResponse(status_code=404, content={"error": "Export file missing"})
            return FileResponse(
                path=item["path"],
                filename=item["filename"],
                media_type="application/octet-stream",
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # ── Main inference endpoint ───────────────────────────────────────
        if path in ("/process", "/api/process") and method == "POST":
            resp = await _handle_process(request)
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

        # ── Everything else → Gradio ──────────────────────────────────────
        return await call_next(request)


# ---------------------------------------------------------------------------
# Gradio Blocks (ZeroGPU compliance — must use demo.launch())
# ---------------------------------------------------------------------------

@spaces.GPU
def predict_gpu(image_path: str) -> str:
    """ZeroGPU entrypoint — detected by HF Spaces AST checker."""
    return image_path


custom_css = """
body, html { margin: 0; padding: 0; height: 100vh; overflow: hidden; background: #08080a; }
.gradio-container { max-width: 100% !important; margin: 0 !important; padding: 0 !important;
                    height: 100vh !important; overflow: hidden; background: #08080a; }
footer { display: none !important; }
#dw-wrap, #dw-wrap iframe { width: 100% !important; height: 100vh !important; border: none; display: block; }
"""

with gr.Blocks(title="DepthWizard — 3D Elevation Platform", css=custom_css, fill_height=True) as demo:
    _hidden_in  = gr.Textbox(visible=False)
    _hidden_out = gr.Textbox(visible=False)
    _hidden_btn = gr.Button(value="process", visible=False)
    _hidden_btn.click(fn=predict_gpu, inputs=[_hidden_in], outputs=[_hidden_out])

    gr.HTML("""
        <div id="dw-wrap">
            <iframe src="/demo.html" style="width:100%;height:100vh;border:none;display:block;"></iframe>
        </div>
    """)

# Attach middleware AFTER demo is built (demo.app is now stable)
demo.app.add_middleware(DepthWizardMiddleware)


if __name__ == "__main__":
    get_model()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        ssr_mode=False
    )
