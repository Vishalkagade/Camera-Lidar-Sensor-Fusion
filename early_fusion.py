import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import open3d as o3d
from ultralytics import YOLO
import time
import cv2
import numpy as np
from PIL import Image
from utils import run_obstacle_detection,draw_boxes_cv, filter_outliers, get_best_distance, rectContains


class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))
    
    def project_velo_to_image(self, pts_3d_velo):
        '''
        Input: 3D points in Velodyne Frame [nx3]
        Output: 2D Pixels in Image Frame [nx2]
        '''
        # REVERSE TECHNIQUE
        '''
        homogeneous = self.cart2hom(pts_3d_velo)  # nx4
        dotted_RT = np.dot(homogeneous, np.transpose(self.V2C)) #nx3
        dotted_with_RO = np.transpose(np.dot(self.R0, np.transpose(dotted_RT))) #nx3
        homogeneous_2 = self.cart2hom(dotted_with_RO) #nx4
        pts_2d = np.dot(homogeneous_2, np.transpose(self.P))  # nx3
        '''
        
        # NORMAL TECHNIQUE
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2) #PxR0
        p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) #PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))#PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)
        
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]
    
    def get_lidar_in_image_fov(self,pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.project_velo_to_image(pc_velo)
        fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # We don't want things that are closer to the clip distance (2m)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo
    
    def show_lidar_on_image(self, pc_velo, img, debug="False"):
        """ Project LiDAR points to image """
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True
        )
        if (debug==True):
            print("3D PC Velo "+ str(imgfov_pc_velo)) # The 3D point Cloud Coordinates
            print("2D PIXEL: " + str(pts_2d)) # The 2D Pixels
            print("FOV : "+str(fov_inds)) # Whether the Pixel is in the image or not
        self.imgfov_pts_2d = pts_2d[fov_inds, :]
        '''
        #homogeneous = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        homogeneous = self.cart2hom(imgfov_pc_velo)
        transposed_RT = np.dot(homogeneous, np.transpose(self.V2C))
        dotted_RO = np.transpose(np.dot(self.R0, np.transpose(transposed_RT)))
        self.imgfov_pc_rect = dotted_RO
        
        if debug==True:
            print("FOV PC Rect "+ str(self.imgfov_pc_rect))
        '''
        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        self.imgfov_pc_velo = imgfov_pc_velo
        
        for i in range(self.imgfov_pts_2d.shape[0]):
            #depth = self.imgfov_pc_rect[i,2]
            #print(depth)
            depth = imgfov_pc_velo[i,0]
            #print(depth)
            color = cmap[int(510.0 / depth), :]
            cv2.circle(
                img,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,
                color=tuple(color),
                thickness=-1,
            )

        return img

    
    def lidar_camera_fusion(self, pred_bboxes, image):
        img_bis = image.copy()

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            for i in range(self.imgfov_pts_2d.shape[0]):
                #depth = self.imgfov_pc_rect[i, 2]
                depth = self.imgfov_pc_velo[i,0]
                if (rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0], shrink_factor=0.1)==True):
                    distances.append(depth)

                    color = cmap[int(510.0 / depth), :]
                    cv2.circle(img_bis,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,color=tuple(color),thickness=-1,)
            h, w, _ = img_bis.shape
            if len(distances) >2:
                distances = filter_outliers(distances)
                best_distance = get_best_distance(distances, technique="average")
                cv2.putText(img_bis, f"{round(best_distance,2)}", (int(box[0]+((box[2]-box[0])/2)),(int(box[1]+((box[3]-box[1])/2)))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)    
            distances_to_keep = []
        
        return img_bis, distances
    def pipeline (self, image, point_cloud,model):
        "For a pair of 2 Calibrated Images"
        img = image.copy()
        # Show LidAR on Image
        lidar_img = self.show_lidar_on_image(point_cloud[:,:3], image)
        # Run obstacle detection in 2D
        result, pred_bboxes = run_obstacle_detection(img,model)
        # Fuse Point Clouds & Bounding Boxes
        img_final, _ = self.lidar_camera_fusion(pred_bboxes, result)
        return img_final
