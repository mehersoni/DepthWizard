"""
Automated Browser End-to-End Acceptance QA Suite for DepthWizard M5.
Uses Playwright to launch a headless Chromium browser, upload real imagery,
interact with Three.js 3D WebGL terrain, and capture multi-panel evidence screenshots.
"""

import os
import sys
import time
import threading
import uvicorn
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from api_server import app


class ServerThread:
    """Helper to start and stop FastAPI server in a background thread."""
    def __init__(self, host="127.0.0.1", port=8005):
        self.host = host
        self.port = port
        self.config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        # Wait until server is listening
        time.sleep(1.5)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=3)


def run_m5_browser_qa():
    print("=" * 85)
    print("DepthWizard M5 — Automated Browser End-to-End Acceptance QA Suite")
    print("=" * 85)

    os.makedirs("outputs/figures", exist_ok=True)

    # 1. Start Server on port 8005 to avoid conflicting with active servers
    server = ServerThread(port=8005)
    server.start()
    print(f"\n[1/7] Started FastAPI server on http://127.0.0.1:8005 (Thread ID: {server.thread.ident})")

    with sync_playwright() as p:
        print("-> Launching Headless Chromium Browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # Capture console errors and messages
        console_errors = []
        page.on("pageerror", lambda err: console_errors.append(str(err)))
        page.on("console", lambda msg: print(f"      [Browser Console] {msg.type}: {msg.text}", flush=True))

        print("-> Navigating to http://127.0.0.1:8005/ ...")
        page.goto("http://127.0.0.1:8005/")
        page.wait_for_selector("#canvas-holder", timeout=10000)
        time.sleep(1.0)

        # Verify Initial Startup State
        title = page.title()
        badge_text = page.locator("#mode-badge").inner_text()
        print(f"      [PASS] Initial Load: Title='{title}', Badge='{badge_text}'")
        assert "Startup Preview" in badge_text, f"Unexpected initial badge: {badge_text}"

        page.screenshot(path="outputs/figures/m5_startup_browser.png")

        # Health check verification
        health_resp = page.request.get("http://127.0.0.1:8005/health")
        assert health_resp.status == 200, "Health check failed"
        print(f"      [PASS] GET /health -> {health_resp.json()}")

        # Uncalibrated GeoTIFF upload test
        potsdam_path = os.path.abspath("data/potsdam_sample_1024.tif")
        print(f"\n-> Testing Potsdam GeoTIFF Upload (Uncalibrated): {potsdam_path}...", flush=True)
        page.locator("#file-input").set_input_files(potsdam_path)

        print("      Waiting for inference & rendering...", flush=True)
        page.wait_for_selector("#status-dot.complete", timeout=60000)
        time.sleep(1.5)

        badge_mode = page.locator("#mode-badge").inner_text()
        dims_val = page.locator("#m-dims").inner_text()
        crs_val = page.locator("#m-crs").inner_text()
        gsd_val = page.locator("#m-gsd").inner_text()
        elev_val = page.locator("#m-elev").inner_text()
        slope_val = page.locator("#m-slope").inner_text()
        conf_val = page.locator("#m-conf").inner_text()

        elev_clean = elev_val.replace("→", "->").replace("°", " deg")
        print(f"      [PASS] Browser DOM: Mode={badge_mode}, Dims={dims_val}, CRS={crs_val}, GSD={gsd_val}, Span={elev_clean}")
        assert "Calibration Required" in badge_mode, f"Expected 'Metric Calibration Required', got: {badge_mode}"
        assert "1024" in dims_val, f"Expected 1024 in dimensions, got {dims_val}"
        assert "EPSG:32633" in crs_val, "Invalid CRS"
        assert "0.050" in gsd_val, "Invalid GSD"
        assert "rel" in elev_clean, "Expected relative height unit"

        # Raycast Inspection
        box = page.locator("#canvas-holder").bounding_box()
        cx = box["x"] + box["width"] / 2
        cy = box["y"]  + box["height"] / 2

        page.mouse.move(cx, cy)
        time.sleep(0.3)
        h1 = page.locator("#h-val").inner_text().replace("→", "->").replace("°", " deg")
        s1 = page.locator("#s-val").inner_text().replace("→", "->").replace("°", " deg")
        c1 = page.locator("#c-val").inner_text().replace("→", "->").replace("°", " deg")
        print(f"      [PASS] Inspection @ Center: Elev={h1}, Slope={s1}, Conf={c1}")
        assert "uncalibrated" in h1 or "rel" in h1, "Inspector must not show metric meters for uncalibrated GeoTIFF!"

        page.locator("#btn-slope").click()
        time.sleep(0.5)
        page.locator("#btn-texture").click()
        time.sleep(0.5)

        page.screenshot(path="outputs/figures/m5_potsdam_browser.png")

        # PNG Upload test
        png_path = os.path.abspath("data/test_sample_photo.png")
        print(f"\n-> Testing PNG Upload: {png_path}...", flush=True)
        page.locator("#file-input").set_input_files(png_path)
        page.wait_for_selector("#status-dot.complete", timeout=30000)
        time.sleep(1.5)

        badge_png = page.locator("#mode-badge").inner_text()
        crs_png = page.locator("#m-crs").inner_text()
        gsd_png = page.locator("#m-gsd").inner_text()

        print(f"      [PASS] Browser DOM: Mode={badge_png}, CRS={crs_png}, GSD={gsd_png}")
        assert "Relative" in badge_png, "Invalid relative mode badge"
        assert "None" in crs_png, "Expected None for CRS"
        assert "N/A" in gsd_png, "Expected N/A for GSD"

        page.screenshot(path="outputs/figures/m5_png_browser.png")

        # Verify no console errors occurred during the entire run
        assert len(console_errors) == 0, f"Encountered browser JS errors: {console_errors}"
        print("      [PASS] Browser console audit: 0 JavaScript errors detected")

        browser.close()

    server.stop()
    print("\n-> Stopped background server.", flush=True)

    # Assemble unified 3-panel figure
    create_unified_qa_figure()


def create_unified_qa_figure():
    """Stitch 3 browser screenshots into a unified multi-panel validation image."""
    img_startup = Image.open("outputs/figures/m5_startup_browser.png")
    img_potsdam = Image.open("outputs/figures/m5_potsdam_browser.png")
    img_png = Image.open("outputs/figures/m5_png_browser.png")

    target_w, target_h = 600, 375
    s_startup = img_startup.resize((target_w, target_h), Image.Resampling.BILINEAR)
    s_potsdam = img_potsdam.resize((target_w, target_h), Image.Resampling.BILINEAR)
    s_png = img_png.resize((target_w, target_h), Image.Resampling.BILINEAR)

    canvas_w = target_w * 3 + 40
    canvas_h = target_h + 80
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(11, 13, 19))

    canvas.paste(s_startup, (10, 60))
    canvas.paste(s_potsdam, (target_w + 20, 60))
    canvas.paste(s_png, (target_w * 2 + 30, 60))

    canvas.save("outputs/figures/m5_e2e_qa.png")
    print("Saved unified figure to: outputs/figures/m5_e2e_qa.png")


if __name__ == "__main__":
    run_m5_browser_qa()
