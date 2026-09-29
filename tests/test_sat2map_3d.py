"""
Unit & Integration Tests for sat2Map 3D Refiner & Exporters.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from PIL import Image

from depth.sat2map_refiner import (
    regularize_roof_planes,
    extract_planar_segments,
    export_point_cloud_ply,
    export_mesh_obj
)
from process_image import process_image, export_point_cloud, export_mesh


class TestSat2Map3DSuite(unittest.TestCase):

    def setUp(self):
        # Create a synthetic synthetic building rooftop with gabled planes and ground
        self.H, self.W = 128, 128
        self.height_map = np.zeros((self.H, self.W), dtype=np.float32)

        # Flat ground = 10m
        self.height_map[:, :] = 10.0

        # Gabled building in center [32:96, 32:96] rising to 25m ridge
        for r in range(32, 96):
            for c in range(32, 96):
                # Ridge at col 64
                dist_to_ridge = abs(c - 64)
                self.height_map[r, c] = 25.0 - dist_to_ridge * 0.35

        # Synthetic RGB
        self.rgb = np.full((self.H, self.W, 3), 120, dtype=np.uint8)
        self.rgb[32:96, 32:96] = [180, 80, 60]  # Terracotta roof color

        self.test_dir = "outputs/test_3d"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_roof_regularization(self):
        # Add high-frequency noise
        noisy_h = self.height_map + np.random.normal(0, 0.5, self.height_map.shape).astype(np.float32)
        regularized = regularize_roof_planes(noisy_h, rgb=self.rgb, spatial_sigma=2.0)

        self.assertEqual(regularized.shape, (self.H, self.W))
        self.assertTrue(np.all(np.isfinite(regularized)))

        # Noise variance should decrease after regularization
        orig_noise = np.std(noisy_h[:30, :30])
        reg_noise = np.std(regularized[:30, :30])
        self.assertLessEqual(reg_noise, orig_noise)

        # Roof facet residual error should also decrease
        orig_roof_err = np.std((noisy_h - self.height_map)[40:80, 40:60])
        reg_roof_err = np.std((regularized - self.height_map)[40:80, 40:60])
        self.assertLessEqual(reg_roof_err, orig_roof_err)

    def test_extract_planar_segments(self):
        planar_info = extract_planar_segments(self.height_map, gsd=0.5, min_plane_size=50)

        self.assertIn("num_planes", planar_info)
        self.assertIn("plane_tilts", planar_info)
        self.assertIn("mean_tilt_deg", planar_info)
        self.assertGreaterEqual(planar_info["num_planes"], 2)

    def test_export_point_cloud_ply(self):
        ply_path = os.path.join(self.test_dir, "test_pointcloud.ply")
        res_path = export_point_cloud_ply(self.height_map, self.rgb, ply_path, gsd_x=0.5, gsd_y=0.5, step=2)

        self.assertTrue(os.path.isfile(res_path))
        self.assertGreater(os.path.getsize(res_path), 500)

        # Verify binary PLY header
        with open(res_path, "rb") as f:
            header_start = f.read(60)
            self.assertTrue(header_start.startswith(b"ply") and b"format binary_little_endian" in header_start)

    def test_export_mesh_obj(self):
        obj_path = os.path.join(self.test_dir, "test_mesh.obj")
        res_obj, res_mtl, res_tex = export_mesh_obj(self.height_map, self.rgb, obj_path, gsd_x=0.5, gsd_y=0.5, step=2)

        self.assertTrue(os.path.isfile(res_obj))
        self.assertTrue(os.path.isfile(res_mtl))
        self.assertTrue(os.path.isfile(res_tex))
        self.assertGreater(os.path.getsize(res_obj), 1000)

        with open(res_obj, "r") as f:
            first_lines = "".join([f.readline() for _ in range(5)])
            self.assertIn("mtllib", first_lines)
            self.assertIn("usemtl", first_lines)

    def test_process_image_3d_pipeline(self):
        # Save sample image
        img_path = os.path.join(self.test_dir, "sample_building.png")
        Image.fromarray(self.rgb).save(img_path)

        # Run process_image with external depth hook
        res = process_image(img_path, external_depth=self.height_map, refine_roofs=True)

        self.assertIn("planar_facets", res)
        self.assertIn("num_planes", res["planar_facets"])
        self.assertIn("planar_facets", res["metadata"])

        # Test wrapper exports
        ply_out = export_point_cloud(res, os.path.join(self.test_dir, "wrapper_pc.ply"))
        self.assertTrue(os.path.isfile(ply_out))

        obj_out, _, _ = export_mesh(res, os.path.join(self.test_dir, "wrapper_mesh.obj"))
        self.assertTrue(os.path.isfile(obj_out))


if __name__ == "__main__":
    unittest.main()
