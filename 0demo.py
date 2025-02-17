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
    # Load the first PCD file for visualization
    pcd = o3d.io.read_point_cloud(str(point_files[0]))

    # Print basic information about the point cloud
    print(pcd)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
