import logging as log

xmin = "xmin"#0
ymin = "ymin"#1
xmax = "xmax"#2
ymax = "ymax"#3

def union(a,b):
    x = min(a[xmin], b[xmin])
    y = min(a[ymin], b[ymin])
    w = max(a[xmin]+a[xmax], b[xmin]+b[xmax]) - x
    h = max(a[ymin]+a[ymax], b[ymin]+b[ymax]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[xmin], b[xmin])
    y = max(a[ymin], b[ymin])
    w = min(a[xmin]+a[xmax], b[xmin]+b[xmax]) - x
    h = min(a[ymin]+a[ymax], b[ymin]+b[ymax]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def area(bounding_box):
	x = bounding_box[xmax] - bounding_box[xmin]
	y = bounding_box[ymax] - bounding_box[ymin]
	return x * y

def areaInt(bounding_box):
	x = bounding_box[2] - bounding_box[0]
	y = bounding_box[3] - bounding_box[1]
	return x * y

def intersection_over_union(bounding_box, ground_truth_bounding_box):
    area_1 = area(bounding_box)
    area_2 = area(ground_truth_bounding_box)
    # no overlapping area
    if(bounding_box[xmin] > ground_truth_bounding_box[xmax]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -100
    if(bounding_box[xmax] < ground_truth_bounding_box[xmin]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -200
    if(bounding_box[ymin] > ground_truth_bounding_box[ymax]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -300
    if(bounding_box[ymax] < ground_truth_bounding_box[ymin]):
        log.info([bounding_box,ground_truth_bounding_box])
        return -400
    # Positive overlapping area
    ex_0 = max(bounding_box[xmin], ground_truth_bounding_box[xmin])
    ex_1 = max(bounding_box[ymin], ground_truth_bounding_box[ymin])
    ex_2 = min(bounding_box[xmax], ground_truth_bounding_box[xmax])
    ex_3 = min(bounding_box[ymax], ground_truth_bounding_box[ymax])
    effective_union = [ex_0, ex_1, ex_2, ex_3]
    area_c = areaInt(effective_union)
    union = area_1 + area_2 - area_c
    intersection = area_c
    return (intersection/union)


def intersectionOverUnion(boxA, boxB):
    """
    boxA = []
    boxA.append([[min(x[xmin]) for x in boxAraw],[min(x[ymin]) for x in boxAraw]])
    boxA.append([[min(x[xmin]) for x in boxAraw],[max(x[ymin]) for x in boxAraw]])
    boxA.append([[max(x[xmin]) for x in boxAraw],[min(x[ymin]) for x in boxAraw]])
    boxA.append([[max(x[xmin]) for x in boxAraw],[max(x[ymin]) for x in boxAraw]])
    boxB = []
    boxB.append([[min(x[xmin]) for x in boxBraw],[min(x[ymin]) for x in boxBraw]])
    boxB.append([[min(x[xmin]) for x in boxBraw],[max(x[ymin]) for x in boxBraw]])
    boxB.append([[max(x[xmin]) for x in boxBraw],[min(x[ymin]) for x in boxBraw]])
    boxB.append([[max(x[xmin]) for x in boxBraw],[max(x[ymin]) for x in boxBraw]])
    """
    #print(boxA)
    #print(boxB)
    #print("+++++++++++++++++++++++++")
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[xmin], boxB[xmin])
    yA = max(boxA[ymin], boxB[ymin])
    xB = min(boxA[xmax], boxB[xmax])
    yB = min(boxA[ymax], boxB[ymax])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[xmax] - boxA[xmin] + 1) * (boxA[ymax] - boxA[ymin] + 1)
    boxBArea = (boxB[xmax] - boxB[xmin] + 1) * (boxB[ymax] - boxB[ymin] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
