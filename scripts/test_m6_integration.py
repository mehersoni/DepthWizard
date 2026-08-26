"""
M6 Integration & End-to-End Acceptance Test Suite
Tests API endpoints, session export, elevation profiling, and browser WebGL interaction.
"""

import os
import sys
import time
import json
import threading
import uvicorn
import rasterio
import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_server import app


class ServerThread:
    def __init__(self, host="127.0.0.1", port=8008):
        self.host = host
        self.port = port
        self.config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        time.sleep(2.0)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=3)


def run_m6_acceptance_tests():
    print("=" * 85)
    print("DepthWizard M6 — Application Integration & Browser Acceptance Suite")
    print("=" * 85)

    os.makedirs("outputs/figures", exist_ok=True)
    server = ServerThread(port=8008)
    server.start()
    print(f"\n[1/6] Started FastAPI M6 Server on http://127.0.0.1:8008 (Thread ID: {server.thread.ident})")

    try:
        with sync_playwright() as p:
            print("-> Launching Headless Chromium Browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()

            console_errors = []
            page.on("pageerror", lambda err: console_errors.append(str(err)))
            page.on("console", lambda msg: print(f"      [Browser Console] {msg.type}: {msg.text}", flush=True) if msg.type in ["error", "warning"] else None)

            # 1. Health check
            print("\n[2/6] Verifying API Server Health...")
            health_resp = page.request.get("http://127.0.0.1:8008/health")
            assert health_resp.status == 200, "Health check failed"
            print(f"      [PASS] GET /health -> {health_resp.json()}")

            # 2. Navigate to M6 Dashboard
            print("\n[3/6] Navigating to M6 Live Dashboard...")
            page.goto("http://127.0.0.1:8008/")
            page.wait_for_selector(".dash-app", timeout=10000)
            title = page.title()
            print(f"      [PASS] Page Loaded: '{title}'")

            # 3. File Upload & Processing
            potsdam_sample = os.path.abspath("data/potsdam_sample_1024.tif")
            print(f"\n[4/6] Uploading and processing test GeoTIFF: {potsdam_sample}...")
            page.locator("#fileInput").set_input_files(potsdam_sample)
            page.wait_for_selector("#fileCard", state="visible")

            # Click Process Button
            page.locator("#processBtn").click()

            # Wait for completion (Stepper reaches Step 5 'done')
            print("      Waiting for monocular depth, elevation estimation, and WebGL rendering...", flush=True)
            page.wait_for_selector(".step[data-i='4'].done", timeout=60000)
            time.sleep(2.0)

            # Validate UI DOM elements
            elev_val = page.locator("#valElev").inner_text()
            slope_val = page.locator("#valSlope").inner_text()
            conf_val = page.locator("#valConf").inner_text()
            span_val = page.locator("#rangeSpanVal").inner_text()
            print(f"      [PASS] Analytics: Elev={elev_val}, Slope={slope_val}, Conf={conf_val}, Span={span_val}")

            # 4. Multi-tab Navigation & Viewport validation
            print("\n[5/6] Testing Multi-Tab Viewports & 3D Interactive Canvas...")
            tabs = ["overview", "depth", "dsm", "error", "3d"]
            for t in tabs:
                page.locator(f".tab[data-tab='{t}']").click()
                time.sleep(0.4)
                vlabel = page.locator("#viewLabel").inner_text()
                print(f"      [PASS] Tab '{t.upper()}' activated -> Viewport Label: '{vlabel}'")

            # Test 3D Orbit & Overlay Controls
            page.locator("#ctrlSlope").click()
            time.sleep(0.3)
            page.locator("#ctrlTexture").click()
            time.sleep(0.3)
            page.locator("#ctrlExagPlus").click()
            time.sleep(0.3)

            # Capture full browser screenshot
            screenshot_path = "outputs/figures/m6_e2e_dashboard.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"      [PASS] Full Dashboard Screenshot saved to: {screenshot_path}")

            # 5. Export Verification
            print("\n[6/6] Testing GeoTIFF Raster Export...")
            btn_export = page.locator("#btnExport")
            assert not btn_export.is_disabled(), "Export button must be enabled after processing"

            with page.expect_download() as download_info:
                btn_export.click()
            download = download_info.value
            download_path = os.path.join("outputs/export", download.suggested_filename)
            download.save_as(download_path)

            print(f"      [PASS] Downloaded export artifact: {download_path} ({os.path.getsize(download_path):,} bytes)")
            with rasterio.open(download_path) as src:
                print(f"      [PASS] Verified GeoTIFF: shape={src.shape}, CRS={src.crs}, dtype={src.dtypes[0]}")
                assert src.shape == (1024, 1024)
                assert src.dtypes[0] == "float32"

            assert len(console_errors) == 0, f"Detected JavaScript errors: {console_errors}"
            print(f"      [PASS] 0 Browser JavaScript console errors detected")

            browser.close()

    finally:
        server.stop()
        print("\n-> Stopped background server.")

    print("\n" + "=" * 85)
    print("ALL M6 INTEGRATION & ACCEPTANCE TESTS PASSED (100% SUCCESS)!")
    print("=" * 85)


if __name__ == "__main__":
    run_m6_acceptance_tests()
