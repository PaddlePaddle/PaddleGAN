from sklearn.metrics import pairwise_distances
import numpy as np
import cv2


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


def Union(samples):
    x1 = min(samples[:, 0])
    y1 = min(samples[:, 1])
    x2 = max(samples[:, 2])
    y2 = max(samples[:, 3])
    area = (x2 - x1) * (y2 - y1)
    return [x1, y1, x2, y2, area]



def cluster_ious(bboxes):
        dist_matrix = pairwise_distances(bboxes, bboxes, metric=IOU)
        indices = set()
        clusters = []
        for i, row in enumerate(dist_matrix):
            cluster = []
            if i not in indices:
                for j in range(len(row)):
                    if row[j] > 0.08:
                        cluster.append(j)
                        indices.add(j)
                clusters.append(cluster)
        return clusters

def union_clusters(bboxes, clusters):
        bboxes_ = np.array(bboxes)
        result_boxes = []
        for cluster in clusters:
            cluster_samples = bboxes_[cluster]
            union_sample = Union(cluster_samples)
            result_boxes.append(union_sample)
        return result_boxes 

def union_results(image, predictions):
    faces_boxes = []
    person_num = len(predictions)
    if person_num == 0:
        return np.array([])
    for rect in predictions:
        area = (rect[3] - rect[1]) * (rect[2] - rect[0])
        faces_boxes.append([*rect, area])
    clusters = cluster_ious(faces_boxes)
    result_boxes = union_clusters(faces_boxes, clusters)
    viz_image = image.copy()
    for res in result_boxes:
        cv2.rectangle(viz_image, (res[0], res[1]), (res[2], res[3]), (255, 255, 0), 3)
    cv2.imwrite("./result_boxes.jpg", viz_image)
    h, w, _ = image.shape
    # possible_ratios = find_upscale_ratios(result_boxes, (h, w))

    detections = upscale_detections(result_boxes, (0.85, 0.8, 0.85, 0.8), (0, 0, w, h))
    viz_image = image.copy()
    for res in detections:
        cv2.rectangle(viz_image, (res[0], res[1]), (res[2], res[3]), (255, 255, 0), 3)
    cv2.imwrite("./detections.jpg", viz_image)
    # viz_image = image.copy()
    # for i, det in enumerate(detections):
    #     for j, det_ in enumerate(detections):
    #         if i == j: continue
    #         if IOU(det, det_) > 0:
    #             line = slice_box(det, det_)
    #             print(line)
    #             cv2.line(viz_image, (int(line[0][0]), int(line[0][1])), 
    #                                 (int(line[1][0]), int(line[1][0])), 
    #                                 (255, 255, 0), 3)
    
    # cv2.imwrite("./lines.jpg", viz_image)
    # max_coords = find_intersections(detections, (h, w))

    # results = rescale_detections(detections, max_coords)
    # viz_image = image.copy()
    # for res in results:
    #     print(res)
    #     cv2.rectangle(viz_image, (res[0], res[1]), (res[2], res[3]), (255, 255, 0), 3)
    # cv2.imwrite("./resutls.jpg", viz_image)
    # return results
    return detections

def largest_results(image, predictions):
    h, w, _ = image.shape
    person_num = len(predictions)
    if person_num == 0:
        return np.array([])
    ratios = [1.0, 0.8, 1.0, 0.8]
    results = upscale_detections(predictions, ratios, (h, w))
    sorted(results, key=lambda area: area[4], reverse=True)
    results_box = [results[0]]
    for i in range(1, person_num):
        num = len(results_box)
        add_person = True
        for j in range(num):
            pre_person = results_box[j]
            iou = IOU(pre_person, results[i])
            if iou > 0.1:
                add_person = False
                break
        if add_person:
            results_box.append(results[i]) 
    return results_box



def count_ious(main_sample, samples):
    return list(map(lambda sample: IOU(main_sample, sample), samples))
        
def upscale_detection(detection, ratios, max_coords):
    ratio_y1, ratio_x1, ratio_y2, ratio_x2 = ratios
    min_x1, min_y1, max_x2, max_y2 = max_coords
    bh, bw = detection[3] - detection[1], detection[2] - detection[0]
    cy, cx = detection[1] + int(bh / 2), detection[0] + int(bw / 2)
    y1, x1 = max(min_y1, cy - int(bh * ratio_y1)), max(min_x1, cx - int(bw * ratio_x1))
    y2, x2 = min(max_y2, cy + int(bh * ratio_y2)), min(max_x2, cx + int(bw * ratio_x2))
    area = (y2 - y1) * (x2 - x1)
    return [x1, y1, x2, y2, area]

# def find_upscale_ratios(detections, shape):
#     h, w = shape 
#     max_ratios = (0.9, 0.85, 1.1, 0.85)
#     possible_ratios = []
#     for i, rect in enumerate(detections):
#         ratio_x1, ratio_y1, ratio_x2, ratio_y2 = max_ratios[0], max_ratios[1], max_ratios[2], max_ratios[3]
#         while True:
#             upscaled_det = upscale_detection(rect, [ratio_x1, ratio_y1, ratio_x2, ratio_y2], shape)
#             ious = count_ious(upscaled_det, [detections[j] for j in range(len(detections)) if i != j])
#             if np.all(np.array(ious) < 0.01):
#                 break
#             ratio_x1, ratio_y1 = ratio_x1 - 0.05, ratio_y1 - 0.05 
#             ratio_x2, ratio_y2 = ratio_x2 - 0.05, ratio_y2 - 0.05
#         possible_ratios.append([ratio_y1, ratio_x1, ratio_y2, ratio_x2])
#     print(possible_ratios)
#     return possible_ratios
    
def upscale_detections(detections, upscale_ratios, coords):
    upscaled_detections = []
    for det in detections: 
        upscaled_detections.append(upscale_detection(det, upscale_ratios, coords))
    return upscaled_detections

def find_intersections(detections, shape): 
    h, w = shape 
    x1s = np.array(detections)[:, 0]
    y1s = np.array(detections)[:, 1]
    x2s = np.array(detections)[:, 2]
    y2s = np.array(detections)[:, 3]
    coords = []
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, _ = det
        max_x1, max_y1, min_x2, min_y2 = 0, 0, w, h
        ious = np.array(count_ious(det,  detections))
        indices_x1 = (x2s < x2) & (ious > 0.0)
        indices_x2 = (x1s > x1) & (ious > 0.0)
        # indices_y1 = (y2s < y2) & (ious > 0.0)
        # indices_y2 = (y1s > y1) & (ious > 0.0)

        max_x1 = np.max(x2s[indices_x1]) if len(x2s[indices_x1]) > 0 else max_x1 
        # max_y1 = np.max(y2s[indices_y1]) if len(y2s[indices_y1]) > 0 else max_y1
        min_x2 = np.min(x1s[indices_x2]) if len(x1s[indices_x2]) > 0 else min_x2
        # min_y2 = np.min(y1s[indices_y2]) if len(y1s[indices_y2]) > 0 else min_y2 
        coords.append([max_x1 - abs(max_x1 - x1) // 2, 
                    max_y1, 
                    min_x2 + abs(min_x2 - x2) // 2, 
                    min_y2])
    return coords

def rescale_detections(detections, coords):
    rescaled_detections = []
    for i in range(len(detections)):
        x1, y1, x2, y2, _ = detections[i]
        max_x1, max_y1, min_x2, min_y2 = coords[i]
        x1 = max(x1, max_x1)
        y1 = max(y1, max_y1)
        x2 = min(x2, min_x2)
        y2 = min(y2, min_y2)
        rescaled_detections.append([x1, y1, x2, y2, (x2 - x1) * (y2 - y1)])
    return rescaled_detections

