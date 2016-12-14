import logging as log


def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def area(bounding_box):
	x = bounding_box[2] - bounding_box[0]
	y = bounding_box[3] - bounding_box[1]
	return x * y

def intersection_over_union(bounding_box, ground_truth_bounding_box):
    area_1 = area(bounding_box)
    area_2 = area(ground_truth_bounding_box)
    # no overlapping area
    if(bounding_box[0] > ground_truth_bounding_box[2]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -100
    if(bounding_box[2] < ground_truth_bounding_box[0]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -200
    if(bounding_box[1] > ground_truth_bounding_box[3]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -300
    if(bounding_box[3] < ground_truth_bounding_box[1]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -400
    # Positive overlapping area
    ex_0 = max(bounding_box[0], ground_truth_bounding_box[0])
    ex_1 = max(bounding_box[1], ground_truth_bounding_box[1])
    ex_2 = min(bounding_box[2], ground_truth_bounding_box[2])
    ex_3 = min(bounding_box[3], ground_truth_bounding_box[3])
    effective_union = [ex_0, ex_1, ex_2, ex_3]
    area_c = area(effective_union)
    union = area_1 + area_2 - area_c
    intersection = area_c
    return (intersection/union)
