
"""
Module: depth.tiled_inference
Description: High-resolution overlapping tiled monocular depth inference using Depth Anything V2.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple, Dict, Any
from depth.depth_model import load_model, get_device


def _create_2d_window(height: int, width: int, window_type: str = "hann") -> np.ndarray:
    if window_type == "hann":
        wy = np.sin(np.linspace(np.pi / (2 * height), np.pi - np.pi / (2 * height), height)) ** 2
        wx = np.sin(np.linspace(np.pi / (2 * width), np.pi - np.pi / (2 * width), width)) ** 2
        w2d = np.outer(wy, wx)
    elif window_type == "pyramid":
        wy = 1.0 - np.abs(np.linspace(-1, 1, height))
        wx = 1.0 - np.abs(np.linspace(-1, 1, width))
        w2d = np.outer(wy, wx)
    else:
        w2d = np.ones((height, width), dtype=np.float32)

    w2d = np.clip(w2d, 1e-4, 1.0).astype(np.float32)
    return w2d


def estimate_depth_tiled(
    image: Union[str, np.ndarray, Image.Image],
    tile_size: int = 512,
    overlap: float = 0.25,
    batch_size: int = 8,
    window_type: str = "hann",
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    start_time = time.time()

    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        pil_img = Image.open(image).convert("RGB")
        rgb_arr = np.array(pil_img, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        arr = np.squeeze(image)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        rgb_arr = arr
    elif isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported input image type: {type(image)}")

    H, W = rgb_arr.shape[:2]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    y_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if len(y_starts) == 0 or (y_starts[-1] + tile_size < H):
        y_starts.append(max(0, H - tile_size))

    x_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if len(x_starts) == 0 or (x_starts[-1] + tile_size < W):
        x_starts.append(max(0, W - tile_size))

    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    tiles = []
    for y0 in y_starts:
        y1 = min(H, y0 + tile_size)
        y0_eff = max(0, y1 - tile_size)
        for x0 in x_starts:
            x1 = min(W, x0 + tile_size)
            x0_eff = max(0, x1 - tile_size)
            tiles.append((y0_eff, y1, x0_eff, x1))

    num_tiles = len(tiles)

    model, processor, target_device = load_model(device=device)

    base_window = _create_2d_window(tile_size, tile_size, window_type=window_type)

    accum_depth = np.zeros((H, W), dtype=np.float64)
    accum_weight = np.zeros((H, W), dtype=np.float64)

    if torch.cuda.is_available() and target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    for b_idx in range(0, num_tiles, batch_size):
        batch_coords = tiles[b_idx : b_idx + batch_size]
        batch_pil = [
            Image.fromarray(rgb_arr[y0:y1, x0:x1])
            for (y0, y1, x0, x1) in batch_coords
        ]

        inputs = processor(images=batch_pil, return_tensors="pt").to(target_device)

        with torch.no_grad():
            outputs = model(**inputs)
            post_processed = processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(y1 - y0, x1 - x0) for (y0, y1, x0, x1) in batch_coords]
            )

        for (y0, y1, x0, x1), pred in zip(batch_coords, post_processed):
            t_depth = pred["predicted_depth"].cpu().numpy().astype(np.float64)
            th, tw = y1 - y0, x1 - x0
            if (th, tw) == (tile_size, tile_size):
                w = base_window
            else:
                w = _create_2d_window(th, tw, window_type=window_type)

            accum_depth[y0:y1, x0:x1] += t_depth * w
            accum_weight[y0:y1, x0:x1] += w

    valid_weight = accum_weight > 0
    final_depth = np.zeros((H, W), dtype=np.float32)
    final_depth[valid_weight] = (accum_depth[valid_weight] / accum_weight[valid_weight]).astype(np.float32)

    elapsed = time.time() - start_time

    peak_gpu_mb = 0.0
    if torch.cuda.is_available() and target_device.type == "cuda":
        peak_gpu_mb = float(torch.cuda.max_memory_allocated(target_device) / (1024 * 1024))

    info = {
        "image_shape": [H, W, 3],
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "grid_y": len(y_starts),
        "grid_x": len(x_starts),
        "total_tiles": num_tiles,
        "batch_size": batch_size,
        "window_type": window_type,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "device": str(target_device)
    }

    return final_depth, info
