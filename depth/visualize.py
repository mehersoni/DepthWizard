"""
M1 Perception Module - Diagnostic Visualization
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI crashes in headless environments
import matplotlib.pyplot as plt
from PIL import Image

def generate_diagnostic_plot(image: Image.Image, depth_array: np.ndarray, output_dir: str):
    """Generates a side-by-side presentation figure."""
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(image)
    axs[0].set_title("Input RGB Satellite Imagery", pad=10)
    axs[0].axis("off")
    
    im = axs[1].imshow(depth_array, cmap="magma")
    axs[1].set_title("M1 Monocular Relative Depth", pad=10)
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "m1_depth_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()