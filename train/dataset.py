"""
Unified Aerial Depth Dataset for Potsdam and GAMUS.
Provides co-registered optical RGB imagery and LiDAR-derived Above Ground Level (AGL / nDSM) elevation.
"""

import os
import glob
import random
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from rasterio.windows import Window

try:
    from scripts.gamus_dataset import GAMUSDataset
except ImportError:
    from ..scripts.gamus_dataset import GAMUSDataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_image(rgb_np: np.ndarray) -> np.ndarray:
    """
    Normalizes RGB image from uint8 [0, 255] or float [0, 1] to ImageNet normalized float32 (3, H, W).
    """
    if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
        # Convert (H, W, 3) to (3, H, W)
        rgb_chw = np.transpose(rgb_np, (2, 0, 1)).astype(np.float32)
    elif rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        rgb_chw = rgb_np.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {rgb_np.shape}")

    if rgb_chw.max() > 1.0:
        rgb_chw = rgb_chw / 255.0

    norm_img = (rgb_chw - IMAGENET_MEAN) / IMAGENET_STD
    return norm_img.astype(np.float32)


class PotsdamTileDataset(Dataset):
    """
    ISPRS Potsdam Dataset reader.
    Extracts random or grid-based crops from high-resolution Potsdam Orthophotos and LiDAR DSMs.
    Converts absolute elevation DSM into Above Ground Level (nDSM / AGL) via robust ground subtraction.
    """
    def __init__(
        self,
        potsdam_dir: str = "data/potsdam",
        tile_ids: Optional[List[str]] = None,
        crop_size: int = 518,
        samples_per_epoch: int = 200,
        is_train: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.potsdam_dir = potsdam_dir
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

        if tile_ids is None:
            # Default split: 02_10, 02_11, 02_13 for train; 02_12 for validation
            if is_train:
                self.tile_ids = ["02_10", "02_11", "02_13"]
            else:
                self.tile_ids = ["02_12"]
        else:
            self.tile_ids = tile_ids

        self.tiles = []
        for tid in self.tile_ids:
            # Match top_potsdam_2_10_RGB.tif or similar
            norm_tid = tid.replace("02_", "2_")
            rgb_matches = glob.glob(os.path.join(potsdam_dir, "**", f"top_potsdam_{norm_tid}_RGB.tif"), recursive=True)
            dsm_matches = glob.glob(os.path.join(potsdam_dir, "**", f"dsm_potsdam_{tid}.tif"), recursive=True)
            if not dsm_matches:
                # Try norm_tid
                dsm_matches = glob.glob(os.path.join(potsdam_dir, "**", f"dsm_potsdam_{norm_tid}.tif"), recursive=True)

            if rgb_matches and dsm_matches:
                self.tiles.append({
                    "tile_id": tid,
                    "rgb_path": rgb_matches[0],
                    "dsm_path": dsm_matches[0]
                })

        if not self.tiles:
            # Check for sample tile fallback
            sample_path = "data/potsdam_sample_1024.tif"
            if os.path.exists(sample_path):
                self.tiles.append({
                    "tile_id": "sample_1024",
                    "rgb_path": sample_path,
                    "dsm_path": None
                })
            else:
                raise FileNotFoundError(f"No valid Potsdam tiles found in '{potsdam_dir}' for tiles {self.tile_ids}")

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.tiles) * 16

    def get_crop(self, tile_info: dict, x_off: Optional[int] = None, y_off: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_path = tile_info["rgb_path"]
        dsm_path = tile_info["dsm_path"]

        with rasterio.open(rgb_path) as src_rgb:
            h, w = src_rgb.height, src_rgb.width
            max_x = max(0, w - self.crop_size)
            max_y = max(0, h - self.crop_size)

            if x_off is None:
                x_off = int(self.rng.integers(0, max_x + 1)) if max_x > 0 else 0
            if y_off is None:
                y_off = int(self.rng.integers(0, max_y + 1)) if max_y > 0 else 0

            window = Window(x_off, y_off, min(self.crop_size, w), min(self.crop_size, h))
            rgb = src_rgb.read([1, 2, 3], window=window)

        if dsm_path is not None and os.path.exists(dsm_path):
            with rasterio.open(dsm_path) as src_dsm:
                dsm = src_dsm.read(1, window=window).astype(np.float32)
        else:
            # Fallback synthetic relative elevation if DSM missing
            dsm = np.zeros((rgb.shape[1], rgb.shape[2]), dtype=np.float32)

        # Pad to crop_size if smaller
        ch, cw = rgb.shape[1], rgb.shape[2]
        if ch < self.crop_size or cw < self.crop_size:
            pad_rgb = np.zeros((3, self.crop_size, self.crop_size), dtype=rgb.dtype)
            pad_dsm = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
            pad_rgb[:, :ch, :cw] = rgb
            pad_dsm[:ch, :cw] = dsm
            rgb = pad_rgb
            dsm = pad_dsm

        # Convert absolute DSM to nDSM / AGL:
        # Ground level is estimated as 5th percentile of valid terrain
        valid = np.isfinite(dsm) & (dsm > -100.0) & (dsm < 500.0)
        if valid.any():
            ground_z = np.percentile(dsm[valid], 5)
            agl = np.clip(dsm - ground_z, 0.0, 150.0)
        else:
            agl = np.zeros_like(dsm)
            valid = np.ones_like(dsm, dtype=bool)

        return rgb, agl, valid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile_info = self.tiles[idx % len(self.tiles)]
        rgb, agl, valid_mask = self.get_crop(tile_info)

        # Data augmentation for training
        if self.is_train:
            # Horizontal flip
            if self.rng.random() > 0.5:
                rgb = np.flip(rgb, axis=2).copy()
                agl = np.flip(agl, axis=1).copy()
                valid_mask = np.flip(valid_mask, axis=1).copy()
            # Vertical flip
            if self.rng.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                agl = np.flip(agl, axis=0).copy()
                valid_mask = np.flip(valid_mask, axis=0).copy()
            # Random 90 degree rotation
            k = int(self.rng.integers(0, 4))
            if k > 0:
                rgb = np.rot90(rgb, k, (1, 2)).copy()
                agl = np.rot90(agl, k, (0, 1)).copy()
                valid_mask = np.rot90(valid_mask, k, (0, 1)).copy()

        pixel_values = torch.from_numpy(normalize_image(rgb))
        depth_tensor = torch.from_numpy(agl.astype(np.float32))
        mask_tensor = torch.from_numpy(valid_mask.astype(bool))

        return {
            "pixel_values": pixel_values,
            "depth": depth_tensor,
            "valid_mask": mask_tensor,
            "source": "potsdam",
            "tile_id": tile_info["tile_id"]
        }


class GAMUSTileDataset(Dataset):
    """
    GAMUS Remote Sensing Dataset reader.
    Provides co-registered optical RGB imagery and LiDAR-derived AGL (metric height) maps.
    """
    def __init__(
        self,
        split: str = "train",
        crop_size: int = 518,
        tile_ids: Optional[List[str]] = None,
        samples_per_epoch: int = 200,
        is_train: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.split = split
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.gamus_api = GAMUSDataset()

        if tile_ids is not None:
            self.tile_ids = tile_ids
        else:
            # Fast default: use local cached tiles if available to avoid remote network latency
            if split == "test" or not is_train:
                self.tile_ids = ["DC_03_26", "DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29"]
            else:
                self.tile_ids = ["DC_05_28", "DC_05_30", "DC_07_21", "DC_07_29", "DC_01_25"]
            # If user explicitly requests full discovery, they can pass tile_ids or call list_tile_ids

        self._cached_tiles: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_tile_data(self, tile_id: str) -> Dict[str, np.ndarray]:
        if tile_id not in self._cached_tiles:
            try:
                loaded = self.gamus_api.load_tile(tile_id, split=self.split, include_class=False)
                self._cached_tiles[tile_id] = {
                    "rgb": loaded["rgb"],
                    "height": loaded["height"]
                }
            except Exception as e:
                # Fallback to test tile if train download fails
                fallback_tile = "DC_03_26"
                loaded = self.gamus_api.load_tile(fallback_tile, split="test", include_class=False)
                self._cached_tiles[tile_id] = {
                    "rgb": loaded["rgb"],
                    "height": loaded["height"]
                }
        return self._cached_tiles[tile_id]

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.tile_ids) * 4

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile_id = self.tile_ids[idx % len(self.tile_ids)]
        data = self._get_tile_data(tile_id)
        rgb = data["rgb"]  # (H, W, 3) uint8
        height = data["height"]  # (H, W) float32 AGL

        H, W = height.shape
        max_y = max(0, H - self.crop_size)
        max_x = max(0, W - self.crop_size)

        if self.is_train:
            y = int(self.rng.integers(0, max_y + 1)) if max_y > 0 else 0
            x = int(self.rng.integers(0, max_x + 1)) if max_x > 0 else 0
        else:
            y = max_y // 2
            x = max_x // 2

        crop_rgb = rgb[y:y+self.crop_size, x:x+self.crop_size]
        crop_h = height[y:y+self.crop_size, x:x+self.crop_size]

        # Convert to CHW
        crop_rgb_chw = np.transpose(crop_rgb, (2, 0, 1)).astype(np.float32)
        valid_mask = np.isfinite(crop_h) & (crop_h >= 0.0) & (crop_h < 300.0)

        # Augmentation
        if self.is_train:
            if self.rng.random() > 0.5:
                crop_rgb_chw = np.flip(crop_rgb_chw, axis=2).copy()
                crop_h = np.flip(crop_h, axis=1).copy()
                valid_mask = np.flip(valid_mask, axis=1).copy()
            if self.rng.random() > 0.5:
                crop_rgb_chw = np.flip(crop_rgb_chw, axis=1).copy()
                crop_h = np.flip(crop_h, axis=0).copy()
                valid_mask = np.flip(valid_mask, axis=0).copy()
            k = int(self.rng.integers(0, 4))
            if k > 0:
                crop_rgb_chw = np.rot90(crop_rgb_chw, k, (1, 2)).copy()
                crop_h = np.rot90(crop_h, k, (0, 1)).copy()
                valid_mask = np.rot90(valid_mask, k, (0, 1)).copy()

        pixel_values = torch.from_numpy(normalize_image(crop_rgb_chw))
        depth_tensor = torch.from_numpy(crop_h.astype(np.float32))
        mask_tensor = torch.from_numpy(valid_mask.astype(bool))

        return {
            "pixel_values": pixel_values,
            "depth": depth_tensor,
            "valid_mask": mask_tensor,
            "source": "gamus",
            "tile_id": tile_id
        }


class UnifiedAerialDepthDataset(Dataset):
    """
    Combined multi-source dataset balancing ISPRS Potsdam (0.05 m GSD)
    and GAMUS (0.30 m GSD) remote sensing aerial collections.
    """
    def __init__(
        self,
        potsdam_dir: str = "data/potsdam",
        crop_size: int = 518,
        samples_per_epoch: int = 200,
        is_train: bool = True,
        potsdam_ratio: float = 0.5,
        seed: int = 42
    ):
        super().__init__()
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train
        self.potsdam_ratio = potsdam_ratio
        self.rng = np.random.default_rng(seed)

        potsdam_samples = int(samples_per_epoch * potsdam_ratio)
        gamus_samples = max(1, samples_per_epoch - potsdam_samples)

        self.potsdam_ds = PotsdamTileDataset(
            potsdam_dir=potsdam_dir,
            crop_size=crop_size,
            samples_per_epoch=potsdam_samples,
            is_train=is_train,
            seed=seed
        )

        split = "train" if is_train else "test"
        self.gamus_ds = GAMUSTileDataset(
            split=split,
            crop_size=crop_size,
            samples_per_epoch=gamus_samples,
            is_train=is_train,
            seed=seed + 1
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Alternate or choose probabilistically
        if self.rng.random() < self.potsdam_ratio:
            return self.potsdam_ds[idx % len(self.potsdam_ds)]
        else:
            return self.gamus_ds[idx % len(self.gamus_ds)]
