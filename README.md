# Camera Lidar Sensor Fusion

## Early Fusion

<div align="Left">
    <img src="early_fusion.png" width="1000">
</div>

### Camera detections & point cloud visualization

<div align="Left">
    <img src="combo.png" width="1000">
</div>

## LiDAR points Fused with YOLO detections

<div align="Left">
    <img src="test-ezgif.com-video-to-gif-converter.gif" width="1000" height="400">
</div>

## LiDAR and Image Fusion Process

- **LiDAR Projection:** This file stores LiDAR points projected onto the image using camera calibration data.
- **Object Detection:** Only points within YOLO's bounding boxes for the desired objects are kept.
- **Outlier Filtering:** Filtering removes outliers, which are points that don't truly belong to the objects.
- **Bounding Box Adjustment:** Two methods are explored for outlier removal:
  - Shrinking bounding boxes to ensure only points truly belonging to the objects are considered.
  - Applying the Sigma Rule to keep points within 1 or 2 standard deviations of the average distance, based on point distances.
- **Focus on Specific Objects:** This filtering ensures the data focuses on the specific objects of interest.

