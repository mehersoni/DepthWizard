import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

def generate_synthetic_scene(
    height=512,
    width=512,
    resolution_m=0.5,
    output_dir='data/synthetic',
    figure_dir='outputs/figures'
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    # 1. Coordinate Grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # 2. Smooth Terrain (10.0m base with smooth undulations)
    base_terrain_elevation = 10.0
    terrain_wave_1 = 2.0 * np.sin(2.0 * np.pi * x_coords / width) * np.cos(np.pi * y_coords / height)
    terrain_wave_2 = 1.2 * np.cos(3.0 * np.pi * (x_coords + y_coords) / (width + height))
    smooth_terrain = base_terrain_elevation + terrain_wave_1 + terrain_wave_2
    
    # Initialize DSM with terrain
    dsm = smooth_terrain.copy().astype(np.float32)
    
    # Initialize RGB canvas (terrain color: lush green/earth tone base)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Procedural grass/ground texture
    np.random.seed(42)
    ground_noise = np.random.normal(0, 5, (height, width))
    rgb[..., 0] = np.clip(70 + ground_noise + terrain_wave_1 * 4, 40, 100).astype(np.uint8)   # Red
    rgb[..., 1] = np.clip(135 + ground_noise + terrain_wave_2 * 4, 100, 180).astype(np.uint8) # Green
    rgb[..., 2] = np.clip(60 + ground_noise, 30, 90).astype(np.uint8)                           # Blue
    
    # Add an asphalt road network across the scene
    road_y = height // 2
    road_w = 24
    dsm[road_y - road_w//2 : road_y + road_w//2, :] = smooth_terrain[road_y - road_w//2 : road_y + road_w//2, :]
    rgb[road_y - road_w//2 : road_y + road_w//2, :] = [75, 75, 80]
    
    # Road markings
    for x in range(0, width, 40):
        rgb[road_y-1:road_y+1, x:x+20] = [230, 230, 230]
        
    # Vertical road
    road_x = width // 2
    rgb[:, road_x - road_w//2 : road_x + road_w//2] = [75, 75, 80]
    for y in range(0, height, 40):
        rgb[y:y+20, road_x-1:road_x+1] = [230, 230, 230]
    
    # 3. Define Buildings with Distinct Heights
    buildings = [
        {
            'name': 'Building A (Low-rise Warehouse)',
            'rel_height_m': 5.0,
            'bbox': [60, 60, 180, 200],  # ymin, xmin, ymax, xmax
            'roof_color': [180, 170, 160], # light beige / gravel
            'border_color': [130, 120, 110]
        },
        {
            'name': 'Building B (Medium-rise Office)',
            'rel_height_m': 10.0,
            'bbox': [60, 300, 200, 440],
            'roof_color': [120, 140, 165], # slate blue / metallic
            'border_color': [80, 95, 115]
        },
        {
            'name': 'Building C (Taller Residential Complex)',
            'rel_height_m': 18.0,
            'bbox': [300, 60, 440, 200],
            'roof_color': [175, 105, 90],  # terracotta / brick
            'border_color': [120, 70, 60]
        },
        {
            'name': 'Building D (High-rise Commercial Tower)',
            'rel_height_m': 25.0,
            'bbox': [300, 300, 450, 450],
            'roof_color': [210, 215, 220], # modern white/glass concrete
            'border_color': [150, 155, 160]
        }
    ]
    
    building_metadata = []
    
    # Apply buildings to DSM and RGB
    for b in buildings:
        ymin, xmin, ymax, xmax = b['bbox']
        h_rel = b['rel_height_m']
        
        # In DSM: building elevation = local terrain + relative height
        dsm[ymin:ymax, xmin:xmax] = smooth_terrain[ymin:ymax, xmin:xmax] + h_rel
        
        # In RGB: draw rooftop and parapet border
        rgb[ymin:ymax, xmin:xmax] = b['roof_color']
        # Draw border/parapet
        bw = 3
        rgb[ymin:ymin+bw, xmin:xmax] = b['border_color']
        rgb[ymax-bw:ymax, xmin:xmax] = b['border_color']
        rgb[ymin:ymax, xmin:xmin+bw] = b['border_color']
        rgb[ymin:ymax, xmax-bw:xmax] = b['border_color']
        
        # Add rooftop features (HVAC / skylights)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        rgb[cy-8:cy+8, cx-12:cx+12] = [80, 80, 80]
        
        # Record metadata
        building_metadata.append({
            'name': b['name'],
            'relative_height_m': float(h_rel),
            'base_terrain_mean_m': float(np.mean(smooth_terrain[ymin:ymax, xmin:xmax])),
            'total_elevation_mean_m': float(np.mean(dsm[ymin:ymax, xmin:xmax])),
            'bbox_pixel': [ymin, xmin, ymax, xmax]
        })
        
    # Cast directional optical shadows to enhance realism (Sun from North-West: dx=+1, dy=+1)
    shadow_overlay = np.zeros((height, width), dtype=bool)
    for b in buildings:
        ymin, xmin, ymax, xmax = b['bbox']
        h_rel = b['rel_height_m']
        s_len = int(h_rel * 1.5)
        pts = np.array([
            [xmax, ymin],
            [xmax + s_len, ymin + s_len],
            [xmax + s_len, ymax + s_len],
            [xmin + s_len, ymax + s_len],
            [xmin, ymax],
            [xmax, ymax]
        ], dtype=np.int32)
        cv2.fillPoly(shadow_overlay, [pts], True)
        
    # Keep buildings unshadowed
    for b in buildings:
        ymin, xmin, ymax, xmax = b['bbox']
        shadow_overlay[ymin:ymax, xmin:xmax] = False
        
    # Darken shadowed RGB pixels
    rgb[shadow_overlay] = (rgb[shadow_overlay].astype(np.float32) * 0.45).astype(np.uint8)
    
    # 4. Save RGB Image
    rgb_path = os.path.join(output_dir, 'rgb.png')
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f'Saved synthetic RGB to: {rgb_path}')
    
    # 5. Save Reference DSM as GeoTIFF
    dsm_path = os.path.join(output_dir, 'reference_dsm.tif')
    crs_epsg = 32633 # UTM zone 33N
    transform = from_origin(500000.0, 5800000.0, resolution_m, resolution_m)
    
    with rasterio.open(
        dsm_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=f'EPSG:{crs_epsg}',
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(dsm.astype(np.float32), 1)
    print(f'Saved synthetic reference DSM GeoTIFF to: {dsm_path}')
    
    # 6. Save Metadata JSON
    metadata = {
        'scene_name': 'synthetic_urban_scene_01',
        'image_dimensions': {
            'width': width,
            'height': height,
            'channels': 3,
            'resolution_meters_per_pixel': resolution_m
        },
        'crs': f'EPSG:{crs_epsg}',
        'transform': [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        'elevation_statistics_meters': {
            'terrain_min': float(np.min(smooth_terrain)),
            'terrain_max': float(np.max(smooth_terrain)),
            'dsm_min': float(np.min(dsm)),
            'dsm_max': float(np.max(dsm)),
            'dsm_mean': float(np.mean(dsm))
        },
        'buildings': building_metadata
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved metadata to: {metadata_path}')
    
    # 7. Generate Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # RGB Subplot
    axes[0].imshow(rgb)
    axes[0].set_title('Synthetic Optical RGB (512x512, 0.5m GSD)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Annotate buildings on RGB
    for b in buildings:
        ymin, xmin, ymax, xmax = b['bbox']
        h = b['rel_height_m']
        axes[0].text(
            (xmin + xmax) / 2, (ymin + ymax) / 2,
            f'+{h:.0f}m',
            color='yellow',
            fontsize=10,
            fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
        )
        
    # DSM Subplot
    im = axes[1].imshow(dsm, cmap='terrain', vmin=float(np.min(dsm)), vmax=float(np.max(dsm)))
    axes[1].set_title('Synthetic Reference DSM (Elevation in Meters)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (m)', rotation=270, labelpad=15, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(figure_dir, 'synthetic_reference.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved visualization to: {fig_path}')
    
    return metadata

if __name__ == '__main__':
    generate_synthetic_scene()
