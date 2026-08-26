import os
import sys
import rasterio
import numpy as np

def inspect_potsdam_tile(
    rgb_path='data/potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif',
    dsm_path='data/potsdam/1_DSM/dsm_potsdam_02_10.tif'
):
    print('='*65)
    print('DepthWizard M2 - ISPRS Potsdam Tile Inspection Report')
    print('='*65)
    
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f'RGB file not found: {rgb_path}')
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f'DSM file not found: {dsm_path}')

    with rasterio.open(rgb_path) as src_rgb:
        rgb_h, rgb_w = src_rgb.height, src_rgb.width
        rgb_count = src_rgb.count
        rgb_dtype = src_rgb.dtypes[0]
        rgb_crs = src_rgb.crs
        rgb_transform = src_rgb.transform
        rgb_nodata = src_rgb.nodata

    with rasterio.open(dsm_path) as src_dsm:
        dsm_h, dsm_w = src_dsm.height, src_dsm.width
        dsm_count = src_dsm.count
        dsm_dtype = src_dsm.dtypes[0]
        dsm_crs = src_dsm.crs
        dsm_transform = src_dsm.transform
        dsm_nodata = src_dsm.nodata
        dsm_data = src_dsm.read(1)
        valid = np.isfinite(dsm_data) & (dsm_data != (dsm_nodata or -9999.0))
        dsm_min = float(np.min(dsm_data[valid]))
        dsm_max = float(np.max(dsm_data[valid]))
        dsm_mean = float(np.mean(dsm_data[valid]))

    print('[RGB Image}')
    print(f'  Path:         {rgb_path}')
    print(f'  Dimensions:   (H={rgb_h}, W={rgb_w}, Bands={rgb_count})')
    print(f'  Datatype:     {rgb_dtype}')
    print(f'  CRS:         {rgb_crs}')
    print(f'  Transform:    {rgb_transform}')
    print(f'  NoData:       {rgb_nodata}')

    print('\n[DSM Reference]')
    print(f'  Path:         {dsm_path}')
    print(f'  Dimensions:   (H={dsm_h}, W={dsm_w}, Bands={dsm_count})')
    print(f'  Datatype:     {dsm_dtype}')
    print(f'  CRS:         {dsm_crs}')
    print(f'  Transform:    {dsm_transform}')
    print(f'  NoData:       {dsm_nodata}')
    print(f'  Elevation Min: {dsm_min:.3f} m')
    print(f'  Elevation Max: {dsm_max:.3f} m')
    print(f'  Elevation Mean:{dsm_mean:.3f} m')

    matches = (rgb_h, rgb_w) == (dsm_h, dsm_w)
    print('\n[Verification]')
    print(f'  Spatial Dimensions Match: {matches}')
    print(f'  CRS Match:             {rgb_crs == dsm_crs}')
    print('='*65)

if __name__ == '__main__':
    inspect_potsdam_tile()
