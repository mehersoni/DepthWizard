"""
Module: depth.depth_model
Description: Monocular Relative Depth Estimation using Depth Anything V2.

Depth Convention:
    Depth Anything V2 outputs scale- and shift-invariant relative inverse depth (disparity).
    - Higher numerical values correspond to regions closer to the optical sensor (i.e. elevated building rooftops).
    - Lower numerical values correspond to regions farther from the optical sensor (i.e. ground-level terrain).
    - Output is a 2D float32 NumPy array preserving the exact spatial dimensions (H, W) of the input RGB image.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
ENV_CHECKPOINT_VAR = "DEPTH_MODEL_CHECKPOINT"

_CACHED_MODEL = None
_CACHED_PROCESSOR = None
_CACHED_DEVICE = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[AutoModelForDepthEstimation, AutoImageProcessor, torch.device]:
    global _CACHED_MODEL, _CACHED_PROCESSOR, _CACHED_DEVICE

    if device is None:
        target_device = get_device()
    else:
        target_device = torch.device(device)

    if checkpoint_path is None:
        checkpoint_path = os.environ.get(ENV_CHECKPOINT_VAR, DEFAULT_HF_MODEL_ID)

    if _CACHED_MODEL is not None and _CACHED_DEVICE == target_device:
        return _CACHED_MODEL, _CACHED_PROCESSOR, _CACHED_DEVICE

    if checkpoint_path != DEFAULT_HF_MODEL_ID and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Depth Anything V2 checkpoint not found at: '{checkpoint_path}'.\n"
            f"Please place the required weights in '{checkpoint_path}' or set the "
            f"environment variable '{ENV_CHECKPOINT_VAR}' to a valid local path or HF model ID."
        )

    print(f"[DepthModel] Loading Depth Anything V2 from: '{checkpoint_path}' on device: {target_device}...")
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint_path).to(target_device)
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Depth Anything V2 from '{checkpoint_path}': {e}\n"
            f"Ensure dependencies are installed and the checkpoint path is valid."
        ) from e

    _CACHED_MODEL = model
    _CACHED_PROCESSOR = processor
    _CACHED_DEVICE = target_device

    return model, processor, target_device


def estimate_depth(
    image_input: Union[str, np.ndarray, Image.Image],
    model: Optional[AutoModelForDepthEstimation] = None,
    processor: Optional[AutoImageProcessor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> np.ndarray:
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: '{image_input}'")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        arr = np.squeeze(image_input)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        pil_img = Image.fromarray(arr).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    target_w, target_h = pil_img.size

    if model is None or processor is None:
        model, processor, device = load_model(device=device)
    elif device is None:
        device = next(model.parameters()).device

    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(target_h, target_w)]
        )
        depth_tensor = post_processed[0]["predicted_depth"]

    depth_map = depth_tensor.cpu().numpy().astype(np.float32)

    if depth_map.shape != (target_h, target_w):
        raise ValueError(
            f"Output depth shape {depth_map.shape} does not match input dimensions ({target_h}, {target_w})"
        )

    return depth_map
