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
    area = (x2 - x1) * (y2 -y1)
    return [x1, y1, x2, y2, area]



def cluster_ious(bboxes):
        dist_matrix = pairwise_distances(bboxes, bboxes, metric=IOU)
        indices = set()
        clusters = []
        for i, row in enumerate(dist_matrix):
            cluster = []
            if i not in indices:
                for j in range(len(row)):
                    if row[j] > 0.05:
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
            bh = rect[3] - rect[1]
            bw = rect[2] - rect[0]
            area = bh * bw
            faces_boxes.append([*rect, area])
    clusters = cluster_ious(faces_boxes)
    result_boxes = union_clusters(faces_boxes, clusters)
    h, w, _ = image.shape
    result = []
    for rect in result_boxes:
        bh = rect[3] - rect[1]
        bw = rect[2] - rect[0]
        cy = rect[1] + int(bh / 2)
        cx = rect[0] + int(bw / 2)
        y1 = max(0, cy - int(bh * 0.9))
        x1 = max(0, cx - int(0.8 * bw))
        y2 = min(h, cy + int(0.9 * bh))
        x2 = min(w, cx + int(0.8 * bw))
        area = (y2 - y1) * (x2 - x1)
        result.append([x1, y1, x2, y2, area])
    return result

def largest_results(image, predictions):
    h, w, _ = image.shape
    results = []
    person_num = len(predictions)
    if person_num == 0:
        return np.array([])
    for rect in predictions:
            bh = rect[3] - rect[1]
            bw = rect[2] - rect[0]
            cy = rect[1] + int(bh / 2)
            cx = rect[0] + int(bw / 2)
            margin = max(bh, bw)
            y1 = max(0, cy - margin)
            x1 = max(0, cx - int(0.8 * margin))
            y2 = min(h, cy + margin)
            x2 = min(w, cx + int(0.8 * margin))
            area = (y2 - y1) * (x2 - x1)
            results.append([x1, y1, x2, y2, area])
    sorted(results, key=lambda area: area[4], reverse=True)
    print(results)
    results_box = [results[0]]
    for i in range(1, person_num):
        num = len(results_box)
        add_person = True
        for j in range(num):
            pre_person = results_box[j]
            iou = IOU(pre_person, results[i])
            print(iou)
            if iou > 0.1:
                add_person = False
                break
        if add_person:
            results_box.append(results[i]) 
    return results_box
    

def upscale(detections):
    ...