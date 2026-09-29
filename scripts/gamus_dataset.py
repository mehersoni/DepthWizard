"""
DepthWizard - GAMUS Dataset Adapter (Hugging Face)
Repository: https://huggingface.co/datasets/earthflow/GAMUS

GAMUS (Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data)
provides co-registered optical RGB remote sensing images, LiDAR-derived Above Ground Level
(AGL / nDSM) metric elevation maps, and pixel-wise semantic classification masks.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import h5py
import numpy as np
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "earthflow/GAMUS"

class GAMUSDataset:
    """
    Lightweight interface to query, download, and load sample tiles from earthflow/GAMUS.
    """
    def __init__(self, repo_id: str = REPO_ID, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self._file_cache: Optional[List[str]] = None

    def list_files(self) -> List[str]:
        if self._file_cache is None:
            self._file_cache = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        return self._file_cache

    def list_tile_ids(self, split: str = "test") -> List[str]:
        """
        List all tile IDs available for a specific split ('train', 'val', 'test').
        Example ID: 'DC_03_26'
        """
        prefix = f"images/{split}/"
        files = self.list_files()
        tile_ids = []
        for f in files:
            if f.startswith(prefix) and f.endswith("_RGB.h5"):
                tile_id = os.path.basename(f).replace("_RGB.h5", "")
                tile_ids.append(tile_id)
        return sorted(tile_ids)

    def load_tile(
        self,
        tile_id: str,
        split: str = "test",
        include_class: bool = True
    ) -> Dict[str, Any]:
        """
        Download and read a GAMUS triple into numpy arrays:
          - rgb: (H, W, 3) uint8
          - height: (H, W) float32 (physical height in meters, AGL / nDSM)
          - cls: (H, W) float32 / uint8 (semantic class labels 0-6), optional
        """
        img_rel = f"images/{split}/{tile_id}_RGB.h5"
        agl_rel = f"heights/{split}/{tile_id}_AGL.h5"
        cls_rel = f"classes/{split}/{tile_id}_CLS.h5"

        img_path = hf_hub_download(repo_id=self.repo_id, filename=img_rel, repo_type="dataset")
        agl_path = hf_hub_download(repo_id=self.repo_id, filename=agl_rel, repo_type="dataset")

        with h5py.File(img_path, "r") as f_img:
            rgb = f_img["image"][:]

        with h5py.File(agl_path, "r") as f_agl:
            height = f_agl["image"][:].astype(np.float32)

        data = {
            "tile_id": tile_id,
            "split": split,
            "rgb": rgb,
            "height": height,
            "img_path": img_path,
            "height_path": agl_path,
        }

        if include_class:
            try:
                cls_path = hf_hub_download(repo_id=self.repo_id, filename=cls_rel, repo_type="dataset")
                with h5py.File(cls_path, "r") as f_cls:
                    cls_mask = f_cls["image"][:]
                data["class_mask"] = cls_mask
                data["class_path"] = cls_path
            except Exception as e:
                data["class_mask"] = None
                data["class_error"] = str(e)

        return data

def inspect_gamus_tile(tile_id: str = "DC_03_26", split: str = "test"):
    print("=" * 65)
    print(f"GAMUS Dataset Inspection: Tile {tile_id} ({split} split)")
    print("=" * 65)
    ds = GAMUSDataset()
    tile = ds.load_tile(tile_id, split=split)

    rgb = tile["rgb"]
    height = tile["height"]
    cls_mask = tile.get("class_mask")

    valid_h = np.isfinite(height) & (height > -999)

    print("[Optical RGB Image]")
    print(f"  Shape:       {rgb.shape}")
    print(f"  Datatype:    {rgb.dtype}")
    print(f"  Value Range: [{rgb.min()}, {rgb.max()}]")

    print("\n[Ground Truth Height / nDSM (AGL in meters)]")
    print(f"  Shape:       {height.shape}")
    print(f"  Datatype:    {height.dtype}")
    print(f"  Elevation Min:  {float(np.min(height[valid_h])):.2f} m")
    print(f"  Elevation Max:  {float(np.max(height[valid_h])):.2f} m")
    print(f"  Elevation Mean: {float(np.mean(height[valid_h])):.2f} m")
    print(f"  Elevation Std:  {float(np.std(height[valid_h])):.2f} m")

    if cls_mask is not None:
        print("\n[Semantic Class Mask]")
        print(f"  Shape:       {cls_mask.shape}")
        print(f"  Datatype:    {cls_mask.dtype}")
        unique_classes = np.unique(cls_mask)
        print(f"  Unique Classes: {unique_classes}")

    print("=" * 65)

if __name__ == "__main__":
    inspect_gamus_tile()
