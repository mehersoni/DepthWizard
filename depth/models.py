"""
M1 Perception Module - Monocular Depth Inference Engine
"""
import cv2
import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageOps
from transformers import pipeline

class DepthEstimator:
    def __init__(self):
        self.pipe = None
        self.is_initialized = False

    def initialize(self, model_size: str = "Base") -> bool:
        """Loads the pretrained model weights into memory."""
        # Map size strings to HuggingFace repo names
        self.model_size = model_size
        model_map = {
            "Small": "depth-anything/Depth-Anything-V2-Small-hf",
            "Base": "depth-anything/Depth-Anything-V2-Base-hf",
            "Large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        repo_id = model_map.get(model_size, model_map["Base"])
        logging.info(f"[M1 MODEL] Initializing Depth Anything V2 {model_size} ({repo_id})...")
        
        try:
            self.pipe = pipeline(task="depth-estimation", model=repo_id)
            self.is_initialized = True
            return True
        except Exception as e:
            logging.error(f"[M1 MODEL] Failed to initialize model: {str(e)}")
            return False

    def predict_with_confidence(self, image: Image.Image, tta_mode: str = "2-view") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Executes depth inference with configurable Rotational Test-Time Augmentation (TTA).
        tta_mode: 'none' (1x), '2-view' (0° + 180°), or '4-view' (0°, 90°, 180°, 270°)
        Returns: (final_depth_array, confidence_map_array)
        """
        if not self.is_initialized:
            logging.error("[M1 MODEL] Cannot predict: Model not initialized.")
            return None

        try:
            import time
            start_time = time.time()
            img_np = np.array(image)
            
            # Determine rotation steps based on mode
            if tta_mode == "4-view":
                rotations = [0, 1, 2, 3]
            elif tta_mode == "2-view":
                rotations = [0, 2]
            else:
                rotations = [0]

            preds = []
            for k in rotations:
                # Rotate input image (k * 90 degrees)
                rotated_input = np.rot90(img_np, k)
                pil_rot = Image.fromarray(rotated_input)
                
                # Predict and extract the tensor 
                res = self.pipe(pil_rot)["predicted_depth"]
                depth_arr = res.squeeze().cpu().numpy().astype(np.float32)
                
                # Rotate predicted depth back to canonical orientation
                canonical_depth = np.rot90(depth_arr, -k)
                preds.append(canonical_depth)

            # Consensus and variance mapping
            stacked = np.stack(preds, axis=0)
            final_depth = np.median(stacked, axis=0)
            
            # The standard deviation across rotations represents structural uncertainty
            if len(preds) > 1:
                uncertainty_map = np.std(stacked, axis=0)
                u_max = np.max(uncertainty_map) if np.max(uncertainty_map) > 0 else 1.0
                confidence_map = 1.0 - (uncertainty_map / u_max)
            else:
                # Default to 1.0 confidence for single-pass inference
                confidence_map = np.ones_like(final_depth)
            
            # ---------------------------------------------------------
            # Calculate Terrain Bias B(D) using Morphological Opening
            # ---------------------------------------------------------
            # A 55x55 kernel is large enough to "erase" building peaks 
            # while preserving the underlying slowly-varying terrain slopes.
            kernel_size = 55
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            # B(D): The smoothed terrain (hills/valleys without buildings)
            terrain_bias = cv2.morphologyEx(final_depth, cv2.MORPH_OPEN, kernel)
            
            # H_rel = D - B(D): Isolate just the buildings
            # We use np.maximum to prevent negative values from float math artifacts
            height_map = np.maximum(final_depth - terrain_bias, 0.0)
            # ---------------------------------------------------------
            # Calculate latency in milliseconds
            inference_ms = round((time.time() - start_time) * 1000, 2)
            
            # Determine execution device from HuggingFace pipeline
            device_used = str(self.pipe.device)

            return {
                "relative_depth": final_depth,      
                "relative_height": height_map,      
                "quality_map": confidence_map,      
                "metadata": {
                    "model": f"Depth-Anything-V2-{self.model_size}",
                    "tta_mode": tta_mode,
                    "device": device_used,
                    "inference_ms": inference_ms,
                    "input_size": (img_np.shape[0], img_np.shape[1]),
                    "dtype": str(final_depth.dtype)
                }
            }
            
        except Exception as e:
            logging.error(f"[M1 MODEL] Inference failed: {str(e)}")
            return None