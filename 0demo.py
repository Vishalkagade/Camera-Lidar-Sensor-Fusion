'''import glob as glob
image_files=glob.glob("data/img/*.png")'''
import open3d as o3d
from pathlib import Path

# Get all PCD files from the velodyne directory
point_files = sorted(Path("data/velodyne").glob("*.pcd"))

# Check if files exist
if not point_files:
    print("No .pcd files found in 'data/velodyne/'")
else:

    for file in point_files:
        pcd = o3d.io.read_point_cloud(str(file))
        print(f"Showing: {file.name}")
        o3d.visualization.draw_geometries([pcd])

