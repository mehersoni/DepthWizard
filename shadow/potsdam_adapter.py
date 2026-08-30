"""
M4 Shadow Cue Module - Potsdam Dataset Adapter & Ground-Truth Loader

Provides a modular dataset adapter for ISPRS Potsdam high-resolution satellite imagery.
Decouples dataset discovery, file pairing, and ground-truth elevation loading from the
shadow detection pipeline.

Design Principles:
1. Primary Input for Shadow Detection: RGB Image ONLY (+ GSD = 0.05 m/px).
2. Reference Inputs for Validation: DSM & nDSM loaded strictly for ground-truth comparison.
3. No Absolute Paths: Uses clean relative paths dynamically resolved from repository root.
4. Preserves Existing Pipeline: Does not alter Phase 1-4 detection/pairing/height logic.
"""

import os
import re
from typing import Dict, Any, List, Optional
import cv2 as cv
import numpy as np
from PIL import Image


def parse_potsdam_tile_id(filename: str) -> Optional[str]:
    """
    Extracts Potsdam tile ID (e.g., '2_10') from filename patterns.
    Handles 'top_potsdam_2_10_RGB.tif', 'dsm_potsdam_02_10.tif', etc.
    """
    m = re.search(r'potsdam_0?(\d+)_0?(\d+)', filename, re.IGNORECASE)
    if m:
        row, col = int(m.group(1)), int(m.group(2))
        return f"{row}_{col}"
    return None


class PotsdamTileSample:
    """
    Encapsulates a single Potsdam satellite tile and its paired ground-truth rasters.
    """
    def __init__(
        self,
        tile_id: str,
        rgb_path: str,
        dsm_path: Optional[str] = None,
        ndsm_lastools_path: Optional[str] = None,
        ndsm_ownapproach_path: Optional[str] = None,
        gsd_m: float = 0.05
    ):
        self.tile_id = tile_id
        r, c = tile_id.split("_")
        self.row = int(r)
        self.col = int(c)
        self.gsd_m = gsd_m
        
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path
        self.ndsm_lastools_path = ndsm_lastools_path
        self.ndsm_ownapproach_path = ndsm_ownapproach_path

    def load_rgb_image(self, as_bgr: bool = True) -> np.ndarray:
        """
        Loads the RGB satellite image for input to the shadow pipeline.
        
        Returns:
            np.ndarray: uint8 array of shape (6000, 6000, 3) in BGR (OpenCV format) or RGB format.
        """
        if not os.path.exists(self.rgb_path):
            raise FileNotFoundError(f"RGB image not found at path: {self.rgb_path}")
        
        img = cv.imread(self.rgb_path, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load RGB image at path: {self.rgb_path}")
        
        if not as_bgr:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def load_dsm(self) -> np.ndarray:
        """
        Loads the 32-bit floating point Digital Surface Model (DSM) raster.
        FOR GROUND-TRUTH COMPARISON ONLY — NOT FOR SHADOW DETECTOR INPUT.
        
        Returns:
            np.ndarray: float32 array of shape (6000, 6000) containing elevation in meters.
        """
        if not self.dsm_path or not os.path.exists(self.dsm_path):
            raise FileNotFoundError(f"DSM file not found for tile {self.tile_id} at {self.dsm_path}")
        
        with Image.open(self.dsm_path) as img:
            dsm_arr = np.array(img, dtype=np.float32)
        return dsm_arr

    def load_ndsm(self, approach: str = "lastools") -> np.ndarray:
        """
        Loads the normalized Digital Surface Model (nDSM) raster.
        FOR GROUND-TRUTH COMPARISON ONLY — NOT FOR SHADOW DETECTOR INPUT.
        
        Parameters:
            approach (str): 'lastools' or 'ownapproach'
            
        Returns:
            np.ndarray: uint8 array of shape (6000, 6000) representing height above ground.
        """
        path = self.ndsm_lastools_path if approach.lower() == "lastools" else self.ndsm_ownapproach_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"nDSM ({approach}) file not found for tile {self.tile_id} at {path}")
        
        ndsm_arr = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if ndsm_arr is None:
            with Image.open(path) as img:
                ndsm_arr = np.array(img, dtype=np.uint8)
        return ndsm_arr

    def get_pipeline_input(self) -> Dict[str, Any]:
        """
        Returns input arguments for the existing shadow height detection pipeline.
        Only provides RGB image path and validated GSD metadata.
        """
        return {
            "image_path": self.rgb_path,
            "meters_per_pixel": self.gsd_m,
            "tile_id": self.tile_id
        }

    def get_ground_truth_reference(self) -> Dict[str, Any]:
        """
        Returns reference paths and handles for ground-truth height comparison.
        """
        return {
            "tile_id": self.tile_id,
            "dsm_path": self.dsm_path,
            "ndsm_lastools_path": self.ndsm_lastools_path,
            "ndsm_ownapproach_path": self.ndsm_ownapproach_path,
            "loader_dsm": self.load_dsm,
            "loader_ndsm": self.load_ndsm
        }

    def __repr__(self) -> str:
        return (
            f"<PotsdamTileSample Tile={self.tile_id} GSD={self.gsd_m}m/px "
            f"RGB='{self.rgb_path}' DSM='{self.dsm_path}'>"
        )


class PotsdamDatasetAdapter:
    """
    Adapter class to discover, pair, and load ISPRS Potsdam dataset tiles using relative paths.
    """
    DEFAULT_GSD_M = 0.05

    def __init__(
        self,
        root_dir: Optional[str] = None,
        demo_dir: str = "demoImages",
        dataset_dir: str = os.path.join("Dataset", "Potsdam")
    ):
        if root_dir is None:
            # Default to workspace root (one level up from shadow/)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.root_dir = root_dir
        self.demo_dir = os.path.join(self.root_dir, demo_dir)
        self.dataset_dir = os.path.join(self.root_dir, dataset_dir)

    def discover_tiles(self) -> List[PotsdamTileSample]:
        """
        Discovers Potsdam RGB tiles in demoImages/, finds matching DSM & nDSM rasters,
        and constructs PotsdamTileSample instances.
        """
        if not os.path.exists(self.demo_dir):
            raise FileNotFoundError(f"demoImages directory not found: {self.demo_dir}")

        demo_files = os.listdir(self.demo_dir)
        
        # Discover RGB Potsdam images
        rgb_files = [
            f for f in sorted(demo_files)
            if "potsdam" in f.lower() and "rgb" in f.lower() and f.lower().endswith(".tif")
        ]

        discovered_samples = []

        for f in rgb_files:
            tid = parse_potsdam_tile_id(f)
            if not tid:
                continue

            r, c = tid.split("_")
            row_z, col_z = f"{int(r):02d}", f"{int(c):02d}"

            # RGB Path
            rgb_rel = os.path.relpath(os.path.join(self.demo_dir, f), self.root_dir)

            # DSM Search Paths (prefer demoImages, fallback to Dataset/Potsdam/)
            dsm_demo = os.path.join(self.demo_dir, f"dsm_potsdam_{row_z}_{col_z}.tif")
            dsm_data = os.path.join(self.dataset_dir, "1_DSM", "1_DSM", f"dsm_potsdam_{row_z}_{col_z}.tif")
            
            dsm_path = dsm_demo if os.path.exists(dsm_demo) else (dsm_data if os.path.exists(dsm_data) else None)
            dsm_rel = os.path.relpath(dsm_path, self.root_dir) if dsm_path else None

            # nDSM lastools Search Paths
            ndsm_las_demo = os.path.join(self.demo_dir, f"dsm_potsdam_{row_z}_{col_z}_normalized_lastools.jpg")
            ndsm_las_data = os.path.join(self.dataset_dir, "1_DSM_normalisation", "1_DSM_normalisation", f"dsm_potsdam_{row_z}_{col_z}_normalized_lastools.jpg")
            
            ndsm_las_path = ndsm_las_demo if os.path.exists(ndsm_las_demo) else (ndsm_las_data if os.path.exists(ndsm_las_data) else None)
            ndsm_las_rel = os.path.relpath(ndsm_las_path, self.root_dir) if ndsm_las_path else None

            # nDSM ownapproach Search Paths
            ndsm_own_demo = os.path.join(self.demo_dir, f"dsm_potsdam_{row_z}_{col_z}_normalized_ownapproach.jpg")
            ndsm_own_data = os.path.join(self.dataset_dir, "1_DSM_normalisation", "1_DSM_normalisation", f"dsm_potsdam_{row_z}_{col_z}_normalized_ownapproach.jpg")
            
            ndsm_own_path = ndsm_own_demo if os.path.exists(ndsm_own_demo) else (ndsm_own_data if os.path.exists(ndsm_own_data) else None)
            ndsm_own_rel = os.path.relpath(ndsm_own_path, self.root_dir) if ndsm_own_path else None

            sample = PotsdamTileSample(
                tile_id=tid,
                rgb_path=rgb_rel,
                dsm_path=dsm_rel,
                ndsm_lastools_path=ndsm_las_rel,
                ndsm_ownapproach_path=ndsm_own_rel,
                gsd_m=self.DEFAULT_GSD_M
            )
            discovered_samples.append(sample)

        return discovered_samples

    def get_tile(self, tile_id: str) -> PotsdamTileSample:
        """
        Retrieves a specific Potsdam tile by ID (e.g. '2_10').
        """
        samples = self.discover_tiles()
        for s in samples:
            if s.tile_id == tile_id:
                return s
        raise KeyError(f"Potsdam tile ID '{tile_id}' not found in discovered dataset.")
