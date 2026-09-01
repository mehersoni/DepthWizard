"""
DepthWizard — Hugging Face Spaces (ZeroGPU compliant)

Architecture:
  1. demo.queue().launch(prevent_thread_lock=True)
       → Gradio creates the ACTUAL served app (demo.server_app), sets up ZeroGPU,
         and starts uvicorn in a background thread. Main thread continues.
  2. We insert our routes at index 0 of demo.server_app.router.routes
       → They appear BEFORE Gradio's catch-all, so /process is always matched first.
  3. threading.Event().wait()
       → Main thread blocks forever so the Space stays alive.

ZeroGPU compliance:
  • demo.launch() is called  ✓
  • @spaces.GPU decorates both inference functions  ✓
  • predict_gpu is wired to _hidden_btn.click() inside the Gradio Blocks  ✓
"""

import os
import io
import json
import base64
import tempfile
import threading
import time
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Optional, Dict, Any

import gradio as gr
import spaces

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.routing import APIRoute
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


@spaces.GPU
def predict_gpu(image_path: str) -> str:
    """ZeroGPU entrypoint — wired into Gradio event graph below."""
    return image_path


# ---------------------------------------------------------------------------
# JSON / array helpers
# ---------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
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
    return np.array(img.resize((target_size, target_size), Image.Resampling.BILINEAR), dtype=np.float32)


def encode_rgb_to_base64_jpeg(rgb: np.ndarray, sz: int = 256, q: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((sz, sz), Image.Resampling.BILINEAR).save(buf, format="JPEG", quality=q)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def encode_colormap_to_base64_jpeg(
    arr: np.ndarray, cmap_name: str = "turbo", sz: int = 256,
    invert: bool = False, q: int = 85,
    vmin: Optional[float] = None, vmax: Optional[float] = None
) -> str:
    valid = np.isfinite(arr)
    if not np.any(valid):
        arr_norm = np.zeros_like(arr, dtype=np.float32)
    else:
        lo = vmin if vmin is not None else float(np.nanmin(arr[valid]))
        hi = vmax if vmax is not None else float(np.nanmax(arr[valid]))
        span = hi - lo
        arr_norm = np.zeros_like(arr, dtype=np.float32) if span < 1e-7 else np.clip((arr - lo) / span, 0.0, 1.0)
    if invert:
        arr_norm = 1.0 - arr_norm
    rgb = (cm.get_cmap(cmap_name)(arr_norm)[:, :, :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((sz, sz), Image.Resampling.BILINEAR).save(buf, format="JPEG", quality=q)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_HTML = os.path.join(_DIR, "demo.html")
if not os.path.isfile(_DEMO_HTML):
    _DEMO_HTML = os.path.join(_DIR, "m6_dashboard.html")


# ---------------------------------------------------------------------------
# Endpoint functions (plain async functions — added via add_api_route later)
# ---------------------------------------------------------------------------

def health_check():
    return {"status": "online", "service": "DepthWizard Engine"}


def serve_demo():
    with open(_DEMO_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache"})


def download_export(session_id: str):
    if session_id not in EXPORT_CACHE:
        raise HTTPException(status_code=404, detail="Export session not found.")
    item = EXPORT_CACHE[session_id]
    if not os.path.isfile(item["path"]):
        raise HTTPException(status_code=404, detail="Export file missing.")
    return FileResponse(path=item["path"], filename=item["filename"], media_type="application/octet-stream")


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
    parsed_gcps = None
    if gcps:
        try:
            parsed_gcps = json.loads(gcps)
        except Exception:
            pass

    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower() or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    temp_dem_path = None
    if dem_file is not None and dem_file.filename:
        dem_ext = os.path.splitext(dem_file.filename)[1].lower() or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=dem_ext) as tmp_dem:
            tmp_dem.write(await dem_file.read())
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

        h_sub = downsample_array(result["height_map"], v_size)
        s_sub = downsample_array(result["slope_map"], v_size)
        c_sub = downsample_array(result["confidence_map"], v_size)

        rgb_b64   = encode_rgb_to_base64_jpeg(result["rgb"], v_size)
        depth_b64 = encode_colormap_to_base64_jpeg(result["depth_map"],      "turbo",   v_size)
        dsm_b64   = encode_colormap_to_base64_jpeg(result["height_map"],     "turbo",   v_size)
        slope_b64 = encode_colormap_to_base64_jpeg(result["slope_map"],      "magma",   v_size, vmin=0.0, vmax=45.0)
        conf_b64  = encode_colormap_to_base64_jpeg(result["confidence_map"], "viridis", v_size, vmin=0.0, vmax=1.0)

        error_b64 = val_metrics = None
        if result.get("error_map") is not None:
            err = result["error_map"]
            error_b64 = encode_colormap_to_base64_jpeg(np.abs(err), "coolwarm", v_size)
            valid_err = err[np.isfinite(err)]
            if len(valid_err) > 0:
                val_metrics = {
                    "mae": float(np.mean(np.abs(valid_err))),
                    "rmse": float(np.sqrt(np.mean(valid_err ** 2))),
                    "max_error": float(np.max(np.abs(valid_err))),
                    "count": int(len(valid_err))
                }

        def _stat(arr, fn, default):
            f = arr[np.isfinite(arr)]
            return float(fn(f)) if len(f) > 0 else default

        h_min  = _stat(result["height_map"], np.min, 0.0)
        h_max  = _stat(result["height_map"], np.max, 1.0)
        s_min  = _stat(result["slope_map"],  np.min, 0.0)
        s_max  = _stat(result["slope_map"],  np.max, 90.0)
        c_mean = _stat(result["confidence_map"], np.mean, 1.0)

        calibrated    = bool(result.get("calibrated", False))
        georeferenced = bool(result.get("georeferenced", False))
        mode          = str(result.get("mode", "rdsm"))
        height_unit   = str(result.get("height_unit", "m" if calibrated else "relative"))

        mid_y = v_size // 2
        profile_pts = [
            {"x": x, "elevation": (None if np.isnan(float(h_sub[mid_y, x])) else float(h_sub[mid_y, x]))}
            for x in range(v_size)
        ]

        h_flat = h_sub.flatten()
        s_flat = s_sub.flatten()
        c_flat = c_sub.flatten()
        h_list = [None if np.isnan(v) else float(v) for v in h_flat]
        s_list = [None if np.isnan(v) else float(v) for v in s_flat]
        c_list = [None if np.isnan(v) else float(v) for v in c_flat]

        session_id = tempfile.NamedTemporaryFile().name.split(os.sep)[-1][:8]
        os.makedirs("outputs/export", exist_ok=True)
        export_ext      = ".tif" if georeferenced else ".png"
        export_filename = f"depthwizard_{session_id}_{'dsm' if calibrated else 'rdsm'}{export_ext}"
        export_filepath = os.path.join("outputs/export", export_filename)
        export_dsm(result, export_filepath)
        EXPORT_CACHE[session_id] = {"path": export_filepath, "filename": export_filename}

        payload = {
            "session_id": session_id, "mode": mode,
            "calibrated": calibrated, "georeferenced": georeferenced,
            "height_unit": height_unit,
            "width": int(w_orig), "height": int(h_orig),
            "visual_width": int(v_size), "visual_height": int(v_size),
            "height_min": h_min, "height_max": h_max,
            "height_map": h_list,
            "rgb": rgb_b64, "depth_preview": depth_b64,
            "dsm_preview": dsm_b64, "slope_preview": slope_b64,
            "confidence_preview": conf_b64, "error_preview": error_b64,
            "validation": val_metrics,
            "slope_map": s_list, "confidence_map": c_list,
            "slope_min": s_min, "slope_max": s_max, "confidence_mean": c_mean,
            "elevation_profile": profile_pts,
            "crs": str(result["crs"]) if result["crs"] is not None else None,
            "export_url": f"/export/{session_id}",
            "metadata": result["metadata"]
        }
        return JSONResponse(content=sanitize_for_json(payload))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc), "detail": str(exc)})
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dem_path and os.path.exists(temp_dem_path) and temp_dem_path != (dem_path or ""):
                os.remove(temp_dem_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Gradio Blocks — ZeroGPU requires demo.launch()
# ---------------------------------------------------------------------------

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
    # Wiring @spaces.GPU function into the Gradio event graph — required for ZeroGPU
    _hidden_btn.click(fn=predict_gpu, inputs=[_hidden_in], outputs=[_hidden_out])

    # /demo.html is served by our custom route inserted below
    gr.HTML("""
        <div id="dw-wrap">
            <iframe src="/demo.html"
                    style="width:100%;height:100vh;border:none;display:block;"
                    allow="cross-origin-isolated">
            </iframe>
        </div>
    """)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    get_model()

    # Launch with prevent_thread_lock=True so main thread continues.
    # This starts uvicorn in a background thread AND sets demo.server_app
    # to the actual FastAPI instance that's being served.
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
        show_api=False,
        ssr_mode=False
    )

    # demo.server_app is the real served FastAPI app — NOT demo.app (which is
    # a separate pre-launch instance that gets replaced by launch()).
    # Insert our routes at index 0, BEFORE Gradio's catch-all route, so they
    # are matched first by Starlette's order-dependent router.
    server_app = demo.server_app
    print(f"[DepthWizard] Injecting custom routes into server_app: {type(server_app)}", flush=True)

    # Build our route objects
    custom_routes = [
        APIRoute("/health",               health_check,             methods=["GET", "POST"]),
        APIRoute("/demo.html",            serve_demo,               methods=["GET"]),
        APIRoute("/demo",                 serve_demo,               methods=["GET"]),
        APIRoute("/export/{session_id}",  download_export,          methods=["GET"]),
        APIRoute("/api/process",          process_image_endpoint,   methods=["POST"]),
        APIRoute("/process",              process_image_endpoint,   methods=["POST"]),
    ]

    # Insert in reverse order so /process ends up at index 0
    for route in reversed(custom_routes):
        server_app.router.routes.insert(0, route)

    # Rebuild the routing table so the new routes are included
    server_app.router.on_startup = server_app.router.on_startup  # no-op touch
    # Force Starlette to rebuild its compiled middleware/route stack on next request
    if hasattr(server_app, 'middleware_stack'):
        server_app.middleware_stack = None

    print("[DepthWizard] Custom routes injected. /process is at index 0.", flush=True)

    # Block main thread forever — the server runs in the background thread
    threading.Event().wait()
