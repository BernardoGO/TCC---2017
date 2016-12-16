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


def intersectionOverUnion(boxA, boxB):
    """
    boxA = []
    boxA.append([[min(x[0]) for x in boxAraw],[min(x[1]) for x in boxAraw]])
    boxA.append([[min(x[0]) for x in boxAraw],[max(x[1]) for x in boxAraw]])
    boxA.append([[max(x[0]) for x in boxAraw],[min(x[1]) for x in boxAraw]])
    boxA.append([[max(x[0]) for x in boxAraw],[max(x[1]) for x in boxAraw]])
    boxB = []
    boxB.append([[min(x[0]) for x in boxBraw],[min(x[1]) for x in boxBraw]])
    boxB.append([[min(x[0]) for x in boxBraw],[max(x[1]) for x in boxBraw]])
    boxB.append([[max(x[0]) for x in boxBraw],[min(x[1]) for x in boxBraw]])
    boxB.append([[max(x[0]) for x in boxBraw],[max(x[1]) for x in boxBraw]])
    """
    #print(boxA)
    #print(boxB)
    #print("+++++++++++++++++++++++++")
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
