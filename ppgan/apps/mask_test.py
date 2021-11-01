import cv2 
import numpy as np
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

def slice_box(box_A:Polygon, box_B:Polygon, line_mult=10):
    "Returns box_A sliced according to the distance to box_B."
    vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
    vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
    vec_AB_norm = np.linalg.norm(vec_AB)
    split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)
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
        list(map(lambda sample: (int(sample[0]), int(sample[1])), polygon.exterior.coords[:-1]))
        for polygon in polygons
    ]

def polygon2mask(polygon, detection):
    w, h = detection[2] - detection[0], detection[3] - detection[1]
    mask = np.zeros((h, w), dtype="int32")
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype('uint8')

if __name__ == '__main__':
    det1 = [10, 40, 90, 120, 6400]
    det2 = [50, 70, 150, 150, 8000]
    det3 = [30, 140, 120, 220, 1100]
    det4 = [90, 160, 180, 210, 4500]
    dets = [det1, det2, det3, det4]
    canvas = np.ones((300, 300, 3)) * 255
    gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in dets]})
    print(gdf.head())
    res_df = slice_all(gdf)
    print(polygons2coords(res_df['geometry']))
    for det in dets:
        canvas = cv2.rectangle(canvas, (det[0], det[1]), (det[2], det[3]), (0,0,0), thickness=2)
    cv2.imshow("", canvas)
    cv2.waitKey(10000)

    