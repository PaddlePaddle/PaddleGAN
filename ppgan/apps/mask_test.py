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
        left_a, top_a, right_a, bottom_a, _ = detection_A
        left_b, top_b, right_b, bottom_b, _ = detection_B
        if left_a < right_b < right_a and top_a < bottom_b < bottom_a and top_b < top_a:
            return  (right_b + left_a) // 2, (bottom_b + top_a) // 2, "top_left"
        if left_a < right_b < right_a and top_a < top_b < bottom_a and bottom_b > bottom_a:
            return (right_b + left_a) // 2, (bottom_a + top_b) // 2, "bottom_left"
        if left_a < left_b < right_a and top_a < bottom_b < bottom_a and top_b < top_a:
            return (right_a + left_b) // 2, (bottom_a + top_b) // 2, "top_right"
        if left_a < left_b < right_a and top_a < top_b < bottom_a and bottom_b > bottom_a:
            return (right_a + left_b) // 2, (top_b + bottom_a) // 2, "bottom_right"  
        if left_a < right_b < right_a:
            return (right_b + left_a) // 2, None, "left"
        if left_a < left_b < right_a:
            return (right_a + left_b) // 2, None, "right"
        if left_a < left_b < right_a:
            return  None, (bottom_a + top_b) // 2, "bottom"
        if left_a <  left_b< right_a:
            return None, (bottom_b + top_a) // 2, "top"
    return None

def slice_detections(detections, shape):
    coords = []
    h, w = shape
    for i, det in enumerate(detections):
        print("detection")
        x1_, x2_, y1_, y2_ = [], [], [], [] 
        x1, x2, y1, y2 = 0, w, 0, h 
        for j, det_ in enumerate(detections):
            if i == j: continue
            print("detections", det, det_)
            point = slice_point(det, det_)
            print("point", point)
            if point is not None:
                if point[2] == "top_left":
                    x1_.append(point[0])
                    y1_.append(point[1])
                elif point[2] == "top_right":
                    x2_.append(point[0])
                    y1_.append(point[1])
                elif point[2] == "bottom_left":
                    x1_.append(point[0])
                    y2_.append(point[1])
                elif point[2] == "bottom_right":
                    x2_.append(point[0])
                    y2_.append(point[1])
                elif point[2] == "top":
                    y1_.append(point[1])
                elif point[2] == "bottom":
                    y2_.append(point[1])
                elif point[2] == "right":
                    x2_.append(point[0])
                elif point[2] == "left":
                    x1_.append(point[0])
        print(x1_, y1_, x2_, y2_)
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
    det4 = [9, 16, 18, 21, 45]
    print(slice_detections([det1, det2, det3, det4], (100, 100)))