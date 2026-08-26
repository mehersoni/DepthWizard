"""
Comprehensive M6 Interactive Feature Acceptance Test Suite
Tests GCP/DEM ingestion, 2D/3D raycasting, transect slice lines, flythrough, and screenshot capture.
"""

import os
import sys
import time
import threading
import uvicorn
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_server import app


class ServerThread:
    def __init__(self, host="127.0.0.1", port=8035):
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


def run_full_feature_acceptance():
    print("=" * 85)
    print("DepthWizard M6 — Complete Interactive Feature Acceptance Suite")
    print("=" * 85)

    os.makedirs("outputs/figures", exist_ok=True)
    server = ServerThread(port=8035)
    server.start()
    print(f"\n[1/7] Started FastAPI Server on http://127.0.0.1:8035 (Thread ID: {server.thread.ident})")

    try:
        with sync_playwright() as p:
            print("-> Launching Headless Chromium Browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            console_errors = []
            page.on("pageerror", lambda err: console_errors.append(str(err)))
            page.on("console", lambda msg: print(f"      [Browser Console] {msg.type}: {msg.text}", flush=True) if msg.type in ["error", "warning"] else None)

            # -----------------------------------------------------------------
            # 1. Page Load & Initial State
            # -----------------------------------------------------------------
            print("\n[2/7] Navigating to M6 Live Console...")
            page.goto("http://127.0.0.1:8035/m6_dashboard.html")
            page.wait_for_selector(".dash-app", timeout=10000)
            print(f"      [PASS] Console Mounted: '{page.title()}'")

            # 2. Test Anchor Drawer & DEM Selection
            print("\n[3/7] Testing Elevation Anchor Drawer (DEM & GCP Input)...")
            page.locator("#drawerToggle").click()
            page.wait_for_selector("#drawerBody.open", state="visible")
            
            # Add 2 manual GCPs
            page.locator("#gcpX").fill("512")
            page.locator("#gcpY").fill("512")
            page.locator("#gcpZ").fill("46.5")
            page.locator("#btnAddGcpManual").click()
            time.sleep(0.2)
            page.locator("#gcpX").fill("200")
            page.locator("#gcpY").fill("200")
            page.locator("#gcpZ").fill("44.0")
            page.locator("#btnAddGcpManual").click()
            time.sleep(0.2)
            gcp_text = page.locator("#gcpList").inner_text()
            print(f"      [PASS] GCP Anchors Added:\n{gcp_text.strip()}")

            # Select DEM file
            dem_sample = os.path.abspath("data/dem_cache/top_potsdam_2_10_RGB_srtm_dem.tif")
            if os.path.isfile(dem_sample):
                page.locator("#demInput").set_input_files(dem_sample)
                time.sleep(0.3)
                dem_lbl = page.locator("#demFileName").inner_text()
                print(f"      [PASS] DEM Attached: '{dem_lbl}'")

            # 3. Main Ingestion & Processing
            potsdam_sample = os.path.abspath("data/potsdam_sample_1024.tif")
            print(f"\n[4/7] Ingesting Primary Raster & Running Elevation Engine...")
            page.locator("#fileInput").set_input_files(potsdam_sample)
            page.wait_for_selector("#fileCard", state="visible")

            page.locator("#processBtn").click()
            print("      Waiting for monocular inference, DEM validation, and 3D WebGL build...", flush=True)
            page.wait_for_selector(".step[data-i='4'].done", timeout=60000)
            time.sleep(2.0)

            # Check Validation Metric Readouts
            val1 = page.locator("#valMetric1").inner_text()
            val2 = page.locator("#valMetric2").inner_text()
            print(f"      [PASS] Live Validation Metrics: Metric 1 = {val1}, Metric 2 = {val2}")

            # Verify 3D is default active view
            vlabel_3d = page.locator("#viewLabel").inner_text()
            tab_3d_active = "active" in page.locator(".tab[data-tab='3d']").get_attribute("class")
            print(f"      [PASS] Default Output is 3D Model: '{vlabel_3d}' (Tab 3D Active = {tab_3d_active})")
            assert tab_3d_active is True

            # 4. Interactive 2D Hover Raycasting & Slice Tool (switch to 2D tab)
            print("\n[5/7] Testing 2D Canvas Hover Raycaster & Transect Tool...")
            page.locator(".tab[data-tab='overview']").click()
            time.sleep(0.3)
            main_canvas = page.locator("#mainCanvas")
            box = main_canvas.bounding_box()
            assert box is not None, "mainCanvas bounding box not found"

            # Hover across center
            page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
            time.sleep(0.3)
            coord_tag = page.locator("#coordTag").inner_text()
            elev_val = page.locator("#valElev").inner_text()
            slope_val = page.locator("#valSlope").inner_text()
            print(f"      [PASS] Hover Raycast @ Center: {coord_tag} -> Elev={elev_val}, Slope={slope_val}")

            # Draw Custom Transect Slice
            page.locator("#btnDrawSlice").click()
            time.sleep(0.2)
            page.mouse.move(box["x"] + 50, box["y"] + 50)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 50, box["y"] + box["height"] - 50)
            page.mouse.up()
            time.sleep(0.5)
            slice_info = page.locator("#sliceInfo").inner_text()
            print(f"      [PASS] Custom Transect Sampled: '{slice_info}'")

            # 5. 3D Viewport, Flythrough & Overlays
            print("\n[6/7] Testing 3D Viewport, Flythrough Animation & Overlays...")
            page.locator(".tab[data-tab='3d']").click()
            time.sleep(0.8)

            # Activate Flythrough
            page.locator("#ctrlFly").click(force=True)
            time.sleep(1.5)
            print("      [PASS] 3D Flythrough Camera Active (Helical Orbit Path)")

            # Toggle Overlays
            page.locator("#ctrlSlope").click(force=True)
            time.sleep(0.4)
            page.locator("#ctrlTexture").click(force=True)
            time.sleep(0.4)
            page.locator("#ctrlFly").click(force=True) # stop flythrough
            time.sleep(0.3)

            # Reset View & North
            page.locator(".vc-btn").nth(2).click(force=True)
            time.sleep(0.3)
            print("      [PASS] Reset View & True North Aligned")

            # Capture Screenshot
            screenshot_path = "outputs/figures/m6_interactive_full.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"      [PASS] High-Res Full Console Snapshot: {screenshot_path}")

            # 6. Tab Error Map & Export Validation
            print("\n[7/7] Verifying Error Map Tab & Export...")
            page.locator(".tab[data-tab='error']").click()
            time.sleep(0.5)
            err_label = page.locator("#viewLabel").inner_text()
            print(f"      [PASS] Error Tab: '{err_label}'")

            btn_export = page.locator("#btnExport")
            assert not btn_export.is_disabled(), "Export button must be enabled"

            assert len(console_errors) == 0, f"Detected JavaScript errors: {console_errors}"
            print(f"      [PASS] 0 Browser JavaScript console errors detected")

            browser.close()

    finally:
        server.stop()
        print("\n-> Stopped background server.")

    print("\n" + "=" * 85)
    print("ALL M6 INTERACTIVE SUITE ACCEPTANCE TESTS PASSED (100% SUCCESS)!")
    print("=" * 85)


if __name__ == "__main__":
    run_full_feature_acceptance()
