from sklearn.metrics import pairwise_distances
import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon, LineString
import geopandas as gpd

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

def union_area(sample_a, sample_b):
    ax1, ay1, ax2, ay2, area_a = sample_a
    bx1, by1, bx2, by2, area_b = sample_b
    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return float(area_a + area_b - interArea)

def slice_point(detection_A, detection_B):
    if IOU(detection_A, detection_B) > 0.08:
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
        x1_, x2_, y1_, y2_ = [], [], [], [] 
        x1, x2, y1, y2 = 0, w, 0, h 
        for j, det_ in enumerate(detections):
            if i == j: continue
            point = slice_point(det, det_)
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
        x1 = np.max(x1_) if len(x1_) > 0 else x1
        y1 = np.max(y1_) if len(y1_) > 0 else y1
        x2 = np.min(x2_) if len(x2_) > 0 else x2
        y2 = np.min(y2_) if len(y2_) > 0 else y2
        coords.append([x1, y1, x2, y2])
    return coords

def Union(samples):
    x1 = min(samples[:, 0])
    y1 = min(samples[:, 1])
    x2 = max(samples[:, 2])
    y2 = max(samples[:, 3])
    area = (x2 - x1) * (y2 - y1)
    return [x1, y1, x2, y2, area]



def cluster_ious(bboxes, image_area):
        dist_matrix = pairwise_distances(bboxes, bboxes, metric=IOU)
        indices = set()
        clusters = []
        for i, row in enumerate(dist_matrix):
            cluster = []
            if i not in indices:
                for j in range(len(row)):
                    if row[j] > 0:
                        union = union_area(bboxes[i], bboxes[j])
                        print(0.5 * (union/image_area))
                        if row[j] > 0.5 * (union/image_area):
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
    h, w, _ = image.shape
    if len(predictions) == 0:
        return np.array([])
    for rect in predictions:
        area = (rect[3] - rect[1]) * (rect[2] - rect[0])
        faces_boxes.append([*rect, area])
    clusters = cluster_ious(faces_boxes, h * w)
    result_boxes = union_clusters(faces_boxes, clusters)
    viz_image = image.copy()
    for res in result_boxes:
        cv2.rectangle(viz_image, (res[0], res[1]), (res[2], res[3]), (255, 255, 0), 3)
    cv2.imwrite("./result_boxes.jpg", viz_image)

    upscaled_detections = upscale_detections(result_boxes, [0.85, 0.75, 0.85, 0.75], [0,0,h,w], h*w)
    for res in result_boxes:
        cv2.rectangle(viz_image, (res[0], res[1]), (res[2], res[3]), (255, 255, 0), 3)
    cv2.imwrite("./upscaled_boxes.jpg", viz_image)

    gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in upscaled_detections]})
    res_df = slice_all(gdf)
    coords = polygons2coords(res_df['geometry'])
    return upscaled_detections, coords






def count_ious(main_sample, samples):
    return list(map(lambda sample: IOU(main_sample, sample), samples))
        
def upscale_detection(detection, ratios, max_coords, image_area):
    ratio_y1, ratio_x1, ratio_y2, ratio_x2 = ratios
    min_x1, min_y1, max_x2, max_y2 = max_coords
    bh, bw = detection[3] - detection[1], detection[2] - detection[0]
    cy, cx = detection[1] + int(bh / 2), detection[0] + int(bw / 2)
    extra_ratio = detection[4] / image_area
    y1, x1 = max(min_y1, cy - int(bh * (ratio_y1 + extra_ratio))), max(min_x1, cx - int(bw * (ratio_x1+ extra_ratio)))
    y2, x2 = min(max_y2, cy + int(bh * (ratio_y2 + extra_ratio))), min(max_x2, cx + int(bw * (ratio_x2+extra_ratio)))
    area = (y2 - y1) * (x2 - x1)
    return [x1, y1, x2, y2, area]


def upscale_detections(detections, upscale_ratios, coords, image_area):
    upscaled_detections = []
    for det in detections: 
        upscaled_detections.append(upscale_detection(det, upscale_ratios, coords, image_area))
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

def largest_results(image, predictions):
    h, w,  = image.shape
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




def slice_box(box_A:Polygon, box_B:Polygon, margin=-10, line_mult=10):
    "Returns box_A sliced according to the distance to box_B."
    vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
    vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
    vec_AB_norm = np.linalg.norm(vec_AB)
    split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin
    line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
    split_box = shapely.ops.split(box_A, line)
    if len(split_box) == 1: return split_box, None, line
    is_center = [s.contains(box_A.centroid) for s in split_box]
    if sum(is_center) == 0:
        return split_box[0], None, line
    where_is_center = np.argwhere(is_center).reshape(-1)[0]
    where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]
    split_box_center = split_box[where_is_center]
    split_box_out = split_box[where_not_center]
    return split_box_center, split_box_out, line

def box2polygon(bbox):
    x1, y1, x2, y2, _ = bbox
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def intersection_list(polylist):
    r = polylist[0]
    for p in polylist:
        r = r.intersection(p)
    return r
    
def slice_one(gdf, index):
    inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]
    if len(inter) == 1: return inter.geometry.values[0]
    box_A = inter.loc[index].values[0]
    inter = inter.drop(index, axis=0)
    polys = []
    for i in range(len(inter)):
        box_B = inter.iloc[i].values[0]
        polyA, *_ = slice_box(box_A, box_B)
        polys.append(polyA)
    return intersection_list(polys)

def slice_all(gdf):
    polys = []
    for i in range(len(gdf)):
        polys.append(slice_one(gdf, i))
    return gpd.GeoDataFrame({'geometry': polys})

def polygons2coords(polygons):
    return [
        list(map(lambda sample: [int(sample[0]), int(sample[1])], polygon.exterior.coords[:-1]))
        for polygon in polygons
    ]

def polygon2mask(polygon, shape):
    mask = np.zeros(shape, dtype="int32")
    print(np.array([polygon], dtype=np.int32))
    cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), 255)
    return mask.astype('uint8')
