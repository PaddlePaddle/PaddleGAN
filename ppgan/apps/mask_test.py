import cv2 
import numpy as np

def IOU(sample_a, sample_b):
        ax1, ay1, ax2, ay2, area_a = sample_a
        bx1, by1, bx2, by2, area_b = sample_b
        xA = max(ax1, bx1)
        yA = max(ay1, by1)
        xB = min(ax2, bx2)
        yB = min(ay2, by2)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        iou = interArea / float(area_a + area_b - interArea)

        return iou

def slice_point(detection_A, detection_B):
    if IOU(detection_A, detection_B):
        left_a, up_a, right_a, bottom_a, _ = detection_A
        left_b, up_b, right_b, bottom_b, _ = detection_B
        if left_a < right_b < right_a and up_a < up_b < bottom_a and up_a < bottom_b < bottom_a:
            return "left"
        if left_a < left_b < right_a and up_a < up_b < bottom_a and up_a < bottom_b < bottom_a:
            return "right"
        if left_a < left_b < right_a and up_a < up_b < bottom_a and left_a <right_b < right_a:
            return  "bottom"
        if left_a <  left_b< right_a and up_a < bottom_b < bottom_a and left_a < right_b< right_a:
            return "up"
        if left_a < right_b < right_a and up_a < bottom_b < bottom_a:
            return  "up_left"
        elif left_a < right_b < right_a and up_a < up_b < bottom_a:
            return "bottom_left"
        elif left_a < left_b < right_a and up_a < bottom_b < bottom_a:
            return "up_right"
        else:
            return "bottom_right"  
    return None

def slice_detections(detections, shape):
    coords = []
    h, w = shape
    for i, det in enumerate(detections):
        x1_, x2_, y1_, y2_ = [], [], [], [] 
        x1, x2, y1, y2 = 0, w, 0, h 
        for j, det_ in enumerate(detections):
            if i == j: continue
            print("detections", det, det_)
            point = slice_point(det, det_)
            print("point", point)
            if point is not None:
                if point[2] == 2:
                    x2_.append(point[0])
                    y2_.append(point[1])
                else:
                    x1_.append(point[0])
                    y1_.append(point[1])
        x1 = np.max(x1_) if len(x1_) > 0 else x1
        y1 = np.max(y1_) if len(y1_) > 0 else y1
        x2 = np.min(x2_) if len(x2_) > 0 else x2
        y2 = np.min(y2_) if len(y2_) > 0 else y2
        coords.append([x1, y1, x2, y2])
    return coords

        
if __name__ == '__main__':
    det1 = [1, 4, 9, 12, 64]
    det2 = [5, 7, 15, 15, 80]
    det3 = [3, 14, 12, 22, 110]
    print(slice_detections([det1, det2, det3], (100, 100)))