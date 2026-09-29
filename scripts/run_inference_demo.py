import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_image import process_image, export_dsm, export_point_cloud, export_mesh

def main():
    image_path = "data/potsdam_sample_1024.tif"
    dsm_ref_path = "data/potsdam_sample_1024_dsm.tif"
    out_dsm = "outputs/dsm/sample_run_predicted_dsm.tif"

    os.makedirs("outputs/dsm", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)

    print("=" * 70)
    print(" DepthWizard - Full Model Pipeline Execution")
    print(f" Input Image  : {image_path}")
    print(f" Reference DEM: {dsm_ref_path}")
    print("=" * 70)

    t0 = time.time()
    result = process_image(
        image_path,
        dem_path=dsm_ref_path,
        use_shadows=True
    )
    total_time = time.time() - t0
    export_dsm(result, out_dsm)

    h_map = result["height_map"]
    print(f"\n[Execution Summary]")
    print(f" Total Elapsed Time      : {total_time:.2f} seconds")
    print(f" Output Dimensions       : {h_map.shape[0]} x {h_map.shape[1]} px")
    print(f" Elevation Range         : {h_map.min():.2f} m to {h_map.max():.2f} m")
    print(f" Mean Elevation          : {h_map.mean():.2f} m")
    print(f" Elevation Std Dev       : {h_map.std():.2f} m")

    if "validation" in result and result["validation"]:
        val = result["validation"]
        print(f"\n[Validation vs True LiDAR Reference]")
        print(f" Mean Absolute Error (MAE): {val.get('mae', 'N/A')} m")
        print(f" Root Mean Square Error   : {val.get('rmse', 'N/A')} m")
        print(f" Linear Correlation (R2)  : {val.get('r2', 'N/A')}")
        print(f" Mean Bias                : {val.get('bias', 'N/A')} m")

    # Export 3D point cloud and mesh
    ply_path = "outputs/export/sample_run_pointcloud.ply"
    obj_path = "outputs/export/sample_run_mesh.obj"
    export_point_cloud(result, ply_path, step=4)
    export_mesh(result, obj_path, step=4)

    print(f"\n[3D Geospatial Products Generated]")
    print(f" Metric GeoTIFF DSM      : {out_dsm}")
    print(f" Stanford PLY Point Cloud: {ply_path}")
    print(f" Wavefront OBJ Mesh      : {obj_path}")
    print("=" * 70)
    print(" Execution Completed Successfully!")

if __name__ == "__main__":
    main()
