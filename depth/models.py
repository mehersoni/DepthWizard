"""
M1 Perception Module - Monocular Depth Inference Engine
"""
import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageOps
from transformers import pipeline

class DepthEstimator:
    def __init__(self):
        self.pipe = None
        self.is_initialized = False

    def initialize(self, model_size: str = "Small") -> bool:
        """Loads the pretrained model weights into memory."""
        # Map size strings to HuggingFace repo names
        model_map = {
            "Small": "depth-anything/Depth-Anything-V2-Small-hf",
            "Base": "depth-anything/Depth-Anything-V2-Base-hf",
            "Large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        repo_id = model_map.get(model_size, model_map["Small"])
        logging.info(f"[M1 MODEL] Initializing Depth Anything V2 {model_size} ({repo_id})...")
        
        try:
            self.pipe = pipeline(task="depth-estimation", model=repo_id)
            self.is_initialized = True
            return True
        except Exception as e:
            logging.error(f"[M1 MODEL] Failed to initialize model: {str(e)}")
            return False

    def predict_with_confidence(self, image: Image.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Executes depth inference with Test-Time Augmentation (TTA) 
        Returns: (final_depth_array, confidence_map_array)
        """
        if not self.is_initialized:
            logging.error("[M1 MODEL] Cannot predict: Model not initialized.")
            return None

        try:
            # 1. Standard Inference
            res_standard = self.pipe(image)["predicted_depth"]
            depth_std = res_standard.squeeze().cpu().numpy().astype(np.float32)
            
            # 2. Flipped Inference (TTA)
            img_flipped = ImageOps.mirror(image)
            res_flipped = self.pipe(img_flipped)["predicted_depth"]
            depth_flip = res_flipped.squeeze().cpu().numpy().astype(np.float32)
            
            # Un-flip to align with standard
            depth_flip_aligned = np.fliplr(depth_flip)
            
            # 3. Calculate Mean Depth and Confidence
            final_depth = (depth_std + depth_flip_aligned) / 2.0
            
            # The difference between predictions represents uncertainty
            uncertainty_map = np.abs(depth_std - depth_flip_aligned)
            u_max = np.max(uncertainty_map) if np.max(uncertainty_map) > 0 else 1.0
            
            # Invert uncertainty to get confidence (0.0 = Low, 1.0 = High)
            confidence_map = 1.0 - (uncertainty_map / u_max)
            
            return final_depth, confidence_map
            
        except Exception as e:
            logging.error(f"[M1 MODEL] Inference failed: {str(e)}")
            return None