import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth.depth_model import load_model, estimate_depth

def test_depth_model():
    rgb_path = os.path.join('data', 'synthetic', 'rgb.png')
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f'Test image not found: {rgb_path}')

    pil_img = Image.open(rgb_path)
    img_w, img_h = pil_img.size
    rgb_arr = np.array(pil_img)

    model, processor, device = load_model()
    depth_map = estimate_depth(rgb_path, model=model, processor=processor, device=device)

    d_min = float(np.min(depth_map))
    d_max = float(np.max(depth_map))
    d_mean = float(np.mean(depth_map))

    print('='*50)
    print('Depth Anything V2 Inference Results')
    print('='*50)
    print(f'image dimensions:  {rgb_arr.shape} (HxWxC)')
    print(f'depth dimensions:  {depth_map.shape} (HxW)')
    print(f'depth min:        {d_min:.4f}')
    print(f'depth max:        {d_max:.4f}')
    print(f'depth mean:       {d_mean:.4f}')
    print('='*50)

    os.makedirs('outputs/figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    axes[0].imshow(rgb_arr)
    axes[0].set_title(f'Input RGB ({img_w}x{img_h})', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(depth_map, cmap='inferno')
    axes[1].set_title('Depth Anything V2 Relative Depth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Relative Depth / Disparity', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join('outputs', 'figures', 'depth_test.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved depth test figure to: {fig_path}')

if __name__ == '__main__':
    test_depth_model()
