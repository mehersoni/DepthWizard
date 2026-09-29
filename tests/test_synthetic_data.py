import os
import json
import numpy as np
import rasterio
from PIL import Image

def test_rasterio_open_dsm():
    dsm_path = os.path.join('data', 'synthetic', 'reference_dsm.tif')
    assert os.path.exists(dsm_path), 'Missing reference_dsm.tif'
    with rasterio.open(dsm_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == 'float32'
        assert src.width == 512
        assert src.height == 512
        assert src.crs is not None
        assert src.transform is not None
        data = src.read(1)
        assert data.shape == (512, 512)
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()
        print('DSM verified: min =', np.min(data), 'max =', np.max(data), 'CRS =', src.crs)

def test_synthetic_rgb():
    rgb_path = os.path.join('data', 'synthetic', 'rgb.png')
    assert os.path.exists(rgb_path), 'Missing rgb.png'
    img = Image.open(rgb_path)
    assert img.size == (512, 512)
    arr = np.array(img)
    assert arr.shape == (512, 512, 3)
    assert arr.dtype == np.uint8
    print('RGB verified: shape =', arr.shape, 'dtype =', arr.dtype)

def test_metadata_json():
    meta_path = os.path.join('data', 'synthetic', 'metadata.json')
    assert os.path.exists(meta_path), 'Missing metadata.json'
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    assert meta['image_dimensions']['width'] == 512
    assert meta['image_dimensions']['height'] == 512
    assert len(meta['buildings']) >= 4
    print('Metadata verified:', len(meta['buildings']), 'buildings documented.')

def test_visualization():
    fig_path = os.path.join('outputs', 'figures', 'synthetic_reference.png')
    assert os.path.exists(fig_path), 'Missing synthetic_reference.png'
    assert os.path.getsize(fig_path) > 5000
    print('Visualization verified: size =', os.path.getsize(fig_path), 'bytes.')

if __name__ == '__main__':
    test_rasterio_open_dsm()
    test_synthetic_rgb()
    test_metadata_json()
    test_visualization()
    print('ALL SYNTHETIC DATA TESTS PASSED!')
