"""
M1 Perception Module - Monocular Depth Inference Engine
"""
import logging
import numpy as np
from typing import Optional
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    """
    Encapsulates the Depth Anything V2 Small pipeline.
    """
    def __init__(self):
        self.pipe = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Loads the pretrained model weights into memory."""
        logging.info("[M1 MODEL] Initializing Depth Anything V2 Small (HF Pipeline)...")
        try:
            self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
            self.is_initialized = True
            return True
        except Exception as e:
            logging.error(f"[M1 MODEL] Failed to initialize model: {str(e)}")
            return False

    def predict(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Executes depth inference on a validated RGB image.
        
        Returns:
        --------
        raw_tensor : np.ndarray (float32)
            The dense relative-depth array, or None if inference fails.
        """
        if not self.is_initialized:
            logging.error("[M1 MODEL] Cannot predict: Model not initialized.")
            return None

        try:
            result = self.pipe(image)
            raw_tensor = result["predicted_depth"]
            
            # Squeeze batch dimension if necessary [1, H, W] -> [H, W]
            if len(raw_tensor.shape) == 3:
                raw_tensor = raw_tensor.squeeze(0)
                
            return raw_tensor.cpu().numpy().astype(np.float32)
        except Exception as e:
            logging.error(f"[M1 MODEL] Inference failed: {str(e)}")
            return None