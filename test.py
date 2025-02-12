import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from early_fusion import *
# Make sure to include LiDAR2Camera class and any other dependencies

def parse_arguments():
    parser = argparse.ArgumentParser(description="Input for fusion")
    parser.add_argument("--model", type=str, default = "yolov8s.pt",required=False, help="Path to model directory")
    parser.add_argument("--img_path", type=str, default = "data/img",required=False, help="Path to image directory")
    parser.add_argument("--pcd_path", type=str, default = "data/velodyne",required=False, help="Path to point cloud directory")
    parser.add_argument("--label_path", type=str,default = "data/label", required=False, help="Path to label directory")
    parser.add_argument("--calib_path", type=str, default = "data/calib",required=False, help="Path to calibration file directory")
    parser.add_argument("--index", type=int, default=0, help="Index to specify file for processing")
    return parser.parse_args()

def main():
    args = parse_arguments()

    image_files = sorted(glob.glob(f"{args.img_path}/*.png"))
    point_files = sorted(glob.glob(f"{args.pcd_path}/*.pcd"))
    label_files = sorted(glob.glob(f"{args.label_path}/*.txt"))
    calib_files = sorted(glob.glob(f"{args.calib_path}/*.txt"))
    
    if args.index >= len(image_files) or args.index >= len(point_files) or args.index >= len(calib_files):
        print("Index out of range.")
        return
    
    model = YOLO(args.model)
    pcd_file = point_files[args.index]
    lidar2cam = LiDAR2Camera(calib_files[args.index])
    cloud = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(cloud.points)

    image = cv2.cvtColor(cv2.imread(image_files[args.index]), cv2.COLOR_BGR2RGB)
    final_result = lidar2cam.pipeline(image.copy(), points,model=model)

    cv2.imwrite("output1.jpg",final_result)

if __name__ == "__main__":
    main()