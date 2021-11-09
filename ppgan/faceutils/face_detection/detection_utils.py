from sklearn.metrics import pairwise_distances
import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon, LineString
import geopandas as gpd
from functools import partial

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
def iou_thresholded(sample_a, sample_b, image_area=1):
    ax1, ay1, ax2, ay2, area_a = sample_a
    bx1, by1, bx2, by2, area_b = sample_b
    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    unionArea = float(area_a + area_b - interArea)
    iou = interArea / unionArea

    return iou > max(0.2, unionArea/image_area)

def union_area(sample_a, sample_b):
    ax1, ay1, ax2, ay2, area_a = sample_a
    bx1, by1, bx2, by2, area_b = sample_b
    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return float(area_a + area_b - interArea)

def Union(samples):
    x1 = min(samples[:, 0])
    y1 = min(samples[:, 1])
    x2 = max(samples[:, 2])
    y2 = max(samples[:, 3])
    area = (x2 - x1) * (y2 - y1)
    return [x1, y1, x2, y2, area]

def cluster_ious(bboxes, image_area):
        dist_matrix = pairwise_distances(bboxes, bboxes, metric=partial(iou_thresholded, image_area=image_area))
        clusters = set()
        indices = np.arange(len(dist_matrix[0]))
        for row in dist_matrix:
            clusters.add(tuple(indices[row == 1]))   
        return clusters

def union_clusters(bboxes, clusters):
        bboxes_ = np.array(bboxes)
        result_boxes = []
        for cluster in clusters:
            cluster_samples = bboxes_[list(cluster)]
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
    upscaled_detections = upscale_detections(result_boxes, (0, 0, w, h))
    bounds, coords = escape_intersections(upscaled_detections, w, h)
    return bounds, coords


def count_ious(main_sample, samples):
    return list(map(lambda sample: IOU(main_sample, sample), samples))
        
def upscale_detection(detection, max_coords):
    bh, bw = detection[3] - detection[1], detection[2] - detection[0]
    min_x1, min_y1, max_x2, max_y2 = max_coords
    ratio_y1, ratio_y2 = 0.95 + bh / max_y2 / 2, 0.65 + bh / max_y2 / 2
    ratio_x = 0.6 + bw / max_x2 / 2  
    cy, cx = detection[1] + int(bh / 2), detection[0] + int(bw / 2)
    y1, x1 = max(min_y1, cy - int(bh * ratio_y1)), max(min_x1, cx - int(bw * ratio_x))
    y2, x2 = min(max_y2, cy + int(bh * ratio_y2)), min(max_x2, cx + int(bw * ratio_x))
    area = (y2 - y1) * (x2 - x1)
    return [x1, y1, x2, y2, area]

def upscale_detections(detections, coords):
    upscaled_detections = []
    for det in detections: 
        upscaled_detections.append(upscale_detection(det, coords))
    return upscaled_detections

def escape_intersections(detections, w, h):
    gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in detections]})
    res_df = slice_all(gdf)
    coords = polygons2coords(res_df['geometry'])
    # print(cv2.fitEllipse(np.array([coords[0]], dtype=np.int32)))
    bounds = [coords2bounds(coord, w, h) for coord in coords]
    return bounds, coords

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

def coords2bounds(polygon_coords, w, h):
    polygon_coords = np.array(polygon_coords)
    x1 = max(min(polygon_coords[:, 0]), 0)
    y1 = max(min(polygon_coords[:, 1]), 0)
    x2 = min(max(polygon_coords[:, 0]), w)
    y2 = min(max(polygon_coords[:, 1]), h)
    return [x1, y1, x2, y2, (x2 - x1)*(y2 - y1)]

def polygon2mask(polygon, shape):
    mask = np.zeros(shape, dtype="int32")
    cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), 255)
    return mask.astype('uint8')

def polygon2ellipse(polygon):
    if len(polygon) < 5:
            polygon.append([(polygon[0][0] + polygon[1][0])//2 +10, 
                            (polygon[0][1] + polygon[1][1])//2 +10])
    (x, y), (MA, ma), angle = cv2.fitEllipse(np.array(polygon).astype("int"))
    return (x, y), (MA, ma), angle

def polygon2ellipsemask(polygon, shape):
    mask = np.zeros(shape, dtype="uint8")
    (x, y), (MA, ma), angle = polygon2ellipse(polygon)
    if x is None or y is None or MA is None or ma is None: 
        return polygon2mask(polygon, shape)
    x, y, MA, ma = int(x), int(y), int(MA/2.5), int(ma/2)
    return cv2.ellipse(mask, (x, y), 
                            (MA, ma), 
                            angle, 0, 360, color=255, thickness=-1)
    

