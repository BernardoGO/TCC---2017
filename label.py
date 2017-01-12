import tensorflow as tf
import sys
import cv2

from PIL import Image
import tensorflow as tf
import numpy as np
from random import randint
import os
import PIL
import cv2
from PIL import ImageFont, ImageDraw
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Polygon, box
from time import gmtime, strftime
from xml.etree import ElementTree
from shapely.geometry import Polygon, box
import pickle
from os import listdir
from os.path import isfile, join
import csv
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import copy
from random import randint
import xml.etree.ElementTree
import pickle


sizex, sizey = 1936, 1296
idx = 1
classes_in_image = {}

class roi:
    def __init__(self):
        self.poly = None
        self.clas = None

def readROIS(mypath):

    rois = []
    classes["bkg"] = 0
    document = ElementTree.parse( mypath )


    membership = document.getroot()
    filename = membership.find("filename").text
    #imgpath = "collection/Images/" + membership.find("folder").text +"/"+ filename
    #imgpath = imgpath.replace("//", "/")
    for user in membership.findall( 'object' ):
        if user.find("deleted").text == "1":
            continue
        points = []
        for pt in user.find("polygon").findall( 'pt' ):
            points.append((int(pt.find("x").text),int(pt.find("y").text)))
        print(points)
        clss = user.find("name").text
        if clss not in classes:
            classes[clss] = len(classes)
        th = roi()
        if len(points) < 4 :
            continue
        th.poly = Polygon(points)
        th.clas = clss
        rois.append(th)

    return rois


def heatmap(rect_pos1):
    print("drawing heatmap")
    image = Image.new("RGB", (sizex, sizey), "black" )
    global classes_in_image
    #rect_pos = class_scores
    colors = get_spaced_colors(len(classes_in_image))
    dr = ImageDraw.Draw(image , 'RGBA')
    font = ImageFont.truetype("sans-serif.ttf", 18)
    for ie in range(len(rect_pos1)):
        eft = classes_in_image[rect_pos1[ie][4]]
        xxf = eft[6]-1
        #print(len(colors))
        #print(xxf)
        color = colors[xxf]
        #print(color)
        #print(classes_in_image[rect_pos[ie][4]][6])
        #if rect_pos1[ie][2]-rect_pos1[ie][0] == x_adds_max:
        #    continue
        #elif rect_pos1[ie][3]-rect_pos1[ie][1] == y_adds_max:
        #    continue


        dr.rectangle(((rect_pos1[ie][1]+randint(0,10), rect_pos1[ie][0]+randint(0,10)),(rect_pos1[ie][3]+randint(0,10),rect_pos1[ie][2]+randint(0,10))), fill=(color[0], color[1], color[2], int(rect_pos1[ie][-1]*100)), outline = None)
    posx = 20
    posy = 20
    for x in classes_in_image:
        posy += 40
        #color = colors[classes_in_image[x][6]]
        #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
    image.save("heatmap.png", quality=100)


def get_spaced_colors(n):
    print("generating colors:" + str(n))
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


#png_data = tf.placeholder(tf.string, shape=[])
#decoded_png = tf.image.decode_jpeg(png_data, channels=3)

def runDetection(image_data, class_scores, x, y, x_adds, y_adds, limit = 0.60, target = None):
    global idx
    global classes_in_image
    global Session
    global graph_def
    #image_data = image_data
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #from numpy import array
    #images = tf.image.decode_jpeg(array(image_data[0]), channels=3)
    #image = tf.image.convert_image_dtype(images, dtype=tf.float32)

    #print([x.name for x in sess.graph.get_operations() ])
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg:0':image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    #active segmentation
    for node_id in top_k:#deleted
        human_string = label_lines[node_id]
        score = predictions[0][node_id]

        if score > limit and str(human_string) != "bkg" and str(human_string) != "plate":
            #print(score)
            if human_string not in classes_in_image:
                classes_in_image[human_string] = [y,x,y+y_adds,x+x_adds, human_string, score, idx]
                idx += 1
            if classes_in_image[human_string][0] < score:
                classes_in_image[human_string] = [y,x,y+y_adds,x+x_adds, human_string, score, idx]
            if target is None or target == human_string:
                class_scores.append([y,x,y+y_adds,x+x_adds, human_string, score])

                ##UNCOMMENT FOR DEBUGGING PURPOSES
                #cv2.rectangle(image, (y, x), (y+y_adds, x+x_adds), (255,0,0), 2)
                #print(type(image))
                #cv2.putText(image, str(human_string) +": "+str(score),(int(y+5),int(x+20)), font, 0.5,(255,255,0),1)
        break
    #print ( classes)
    #classes_in_image.extend(classes)

def joinROIS(class_scores):
    print("joining rois")
    foid = True
    while foid:
        breako = False
        for ii in class_scores:
            for xx in class_scores:
                if xx == ii: continue
                rectxx = box(xx[3],xx[2], xx[1], xx[0], True)
                rectii = box(ii[3],ii[2], ii[1], ii[0], True)
                #(not (ii[0] < xx[0]) or (ii[1] < xx[1]) or (ii[2] > xx[2]) or (ii[3] > xx[3])) and
                if rectxx.intersects(rectii) and (ii[4] == xx[4]):
                    ii[0] = min(ii[0],  xx[0])
                    ii[1] = min(ii[1],  xx[1])
                    ii[2] = max(ii[2],  xx[2])
                    ii[3] = max(ii[3],  xx[3])
                    class_scores.remove(xx)
                    breako = True
                    break
            if breako: break
        if breako: continue
        break

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
    print(boxA)
    print(boxB)
    print("+++++++++++++++++++++++++")
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

def compareROIs(imageXmlPath, dr, rect_pos):
    global sizex
    global sizey
    xmlRoot = xml.etree.ElementTree.parse(imageXmlPath).getroot()
    imageSizeRows = int(xmlRoot.findall('imagesize')[0].findall('nrows')[0].text)
    imageSizeCols = int(xmlRoot.findall('imagesize')[0].findall('ncols')[0].text)
    _CLASSES_TO_IGNORE_ = ["plate", "rag"]
    #print("Image Cols: " + imageSizeCols)
    #print("Image Rows: " + imageSizeRows)

    objects = xmlRoot.findall('object')
    count = -1


    for annoObject in objects:
        count += 1
        name = annoObject.findall('name')[0].text.replace("_", " ")
        isDeleted = annoObject.findall('deleted')[0].text
        polygon = annoObject.findall('polygon')[0]
        points = polygon.findall('pt')
        if name in _CLASSES_TO_IGNORE_:
            print("Ignored Class")
            continue
        ptsLst = []
        for x in points:
            ix = int(x.findall('x')[0].text)
            iy = int(x.findall('y')[0].text)
            ix *= (sizex/imageSizeCols)
            iy *= (sizey/imageSizeRows)
            ptsLst.extend([ix,iy])
        if len(ptsLst) > 8:
            print("More than 4")
            continue
        #print(ptsLst)
        for predObj in rect_pos:
            if predObj[4] == name:
                print("---------------->>" + name + ": ")
                jaccard = intersectionOverUnion(predObj[0:4], [ptsLst[1],ptsLst[0],ptsLst[5],ptsLst[4]])
                print(")))))))-" + str(jaccard))

        dr.polygon(ptsLst, fill=(0, 0, 0, 50), outline = (255, 255, 255))

classes = {}
font = cv2.FONT_HERSHEY_SIMPLEX

diff = "/home/bernardo/Chicken-and-Waffles-2.jpg"
#tstImg = "/home/bernardo/projects/100ND40X/DSC_1611.JPG"#"/home/bernardo/projects/hd/tf/supervised/Images/10feb04_static_cars_highland/img_0808.jpg"
#tstImg = "google-image(206).jpg"
#tstImg = "/home/bernardo/test/rag1.jpg"

tstImg = "dsc_1703.jpg"
tstImg = "dsc_1570.jpg"
tstImg = "dsc_1703a.jpg"
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("pos/no_rag.txt")]#output_labels#home/bernardo/food
with tf.gfile.FastGFile("pos/no_rag.pb", 'rb') as f: #outputgraph
    import time
    import numpy as np

    start = time.time()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    class_scores = []
    class_scores_lv1 = []

    with tf.Session() as sess:
        img = Image.open(tstImg)
        image = img.resize((sizex, sizey), Image.ANTIALIAS)#
        truth = copy.deepcopy(image)
        #truth_rois = readROIS(tstImg.replace(".jpg", ".xml"))
        #dr1 = ImageDraw.Draw(truth , 'RGBA')
        #for roi in truth_rois:
        #    #rpoints = list(roi.poly.exterior.coords)
        #    print(roi.clas)
        #    #dr1.rectangle(((rpoints[1]+randint(0,10), rpoints[0]+randint(0,10)),(rpoints[2]+randint(0,10),rpoints[3]+randint(0,10))), fill=(255, 255, 255, 50), outline = (255, 255, 255))

        #truth.save("truth.png", quality=100)
        width, height = image.size
        __HYBRID__ = False
        if __HYBRID__ == True:

            print("runnning high level detection")
            x_adds_max = 175
            x_adds_min = 175
            y_adds_max = 175
            y_adds_min = 175
            for x in range(0,width,125):
                for y in range(0,height,125):
                    #different sizes sliding window

                    x_adds = x_adds_max
                    y_adds = y_adds_max
                    crop_img = image.crop((x, y, x+x_adds, y+y_adds))
                    rdata = crop_img.resize((75, 75), Image.ANTIALIAS)

                    #class_scores.append([y,x,y+y_adds,x+x_adds, human_string, score])
                    runDetection(rdata, class_scores_lv1, x, y, x_adds, y_adds, 0.85)
                    if len(class_scores_lv1) > 0 and class_scores_lv1[-1][-2] == "bkg":
                        y += 0

            heatmap(class_scores_lv1)
            joinROIS(class_scores_lv1)

            x_adds2 = 150
            y_adds2 = 150
            print("runnning low level detection")
            for lvl1 in class_scores_lv1:
                for x in range(lvl1[1],lvl1[3],125):
                    for y in range(lvl1[0],lvl1[2],125):
                        crop_img = image.crop((x, y, x+x_adds2, y+y_adds2))
                        rdata = crop_img.resize((75, 75), Image.ANTIALIAS)
                        runDetection(rdata, class_scores, x, y, x_adds2, y_adds2, 0.50, lvl1[4])
            heatmap(class_scores)
            joinROIS(class_scores)
        else:
            print("runnning low level detection")
            x_adds_max = 150
            x_adds_min = 150
            y_adds_max = 150
            y_adds_min = 150
            for x in range(0,width,125):#50
                for y in range(0,height,125):#50
                    #different sizes sliding window

                    x_adds = x_adds_max
                    y_adds = y_adds_max
                    crop_img = image.crop((x, y, x+x_adds, y+y_adds))
                    rdata = crop_img.resize((75, 75), Image.ANTIALIAS)

                    #class_scores.append([y,x,y+y_adds,x+x_adds, human_string, score])
                    runDetection(rdata, class_scores, x, y, x_adds, y_adds, 0.60)

            heatmap(class_scores)
            joinROIS(class_scores)




        print("drawing")

        rect_pos = class_scores
        colors = get_spaced_colors(len(classes_in_image))
        dr = ImageDraw.Draw(image , 'RGBA')
        font = ImageFont.truetype("sans-serif.ttf", 18)
        for ie in range(len(rect_pos)):
            eft = classes_in_image[rect_pos[ie][4]]
            xxf = eft[6]-1
            #print(len(colors))
            #print(xxf)
            color = colors[xxf]
            #print(color)
            #print(classes_in_image[rect_pos[ie][4]][6])
            if rect_pos[ie][2]-rect_pos[ie][0] == x_adds_max:
                continue
            elif rect_pos[ie][3]-rect_pos[ie][1] == y_adds_max:
                continue

            classesinit = str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0])
            print(classesinit)
            dr.rectangle(((rect_pos[ie][1]+randint(0,10), rect_pos[ie][0]+randint(0,10)),(rect_pos[ie][3]+randint(0,10),rect_pos[ie][2]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
            dr.text((int(rect_pos[ie][1]+5),int(rect_pos[ie][0]+(randint(0,80)))),classesinit,(color[0], color[1], color[2]),font=font)
            #cv2.rectangle(image, (rect_pos[ie][0]+randint(0,10), rect_pos[ie][1]+randint(0,10)), (rect_pos[ie][2]+randint(0,10),rect_pos[ie][3]+randint(0,10)), (color[0], color[1], color[2]), 2)
            #cv2.putText(image,str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0]),(int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))), font, 0.5,(color[0], color[1], color[2]),2)

        #cv2.imshow("cropped", image)
        #cv2.waitKey(0)
        #cv2.imwrite("output.jpg", image)

        # run your code
        end = time.time()

        elapsed = end - start
        #compareROIs("dsc_1703.xml", dr, rect_pos)
        print("elapsed time: " + str(elapsed))
        image.save("output.png", quality=100)
