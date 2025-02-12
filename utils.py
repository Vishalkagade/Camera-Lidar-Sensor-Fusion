import cv2
import numpy as np
from PIL import Image
import statistics
import random
import time

def run_obstacle_detection(img,model):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = model(img ,conf = 0.5,classes = [1,2,7])
    for r in predictions:
        pred_bboxes = (r.boxes.data).detach().cpu().numpy()
        result = r.plot()
    return result,pred_bboxes

def draw_boxes_cv(img, detections):
    # Open the image only if it's a path
    if isinstance(img, str):
        img = cv2.imread(img)
    # Convert PIL Image to numpy array if necessary
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    # Assuming class 2 is 'car'
    class_labels = {2: 'car'}
    
    # Loop through detections and draw each as a rectangle
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        # Draw rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Prepare text for class label and confidence
        label = f"{class_labels.get(class_id, str(class_id))}: {confidence:.2f}"
        # Calculate text width & height to draw the transparent boxes as background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1) - 20 + h), (0, 255, 0), -1)
        # Put text above rectangle
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
    # Convert back to PIL Image before returning, if needed
    img_pil = Image.fromarray(img)
    return img_pil
def rectContains(rect, pt,w, h, shrink_factor=0):
    # Calculate the new corners of the rectangle after applying the shrink factor
    x1 = rect[0] + (rect[2] - rect[0]) * shrink_factor
    y1 = rect[1] + (rect[3] - rect[1]) * shrink_factor
    x2 = rect[2] - (rect[2] - rect[0]) * shrink_factor
    y2 = rect[3] - (rect[3] - rect[1]) * shrink_factor

    # Check if the point is within the shrunk rectangle
    return x1 < pt[0] < x2 and y1 < pt[1] < y2

def filter_outliers(distances):
    inliers = []
    mu  = statistics.mean(distances)
    std = statistics.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            # This is an INLIER
            inliers.append(x)
    return inliers

def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))