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

classes = {}

#with open('classes.pickle', 'rb') as handle:
#    classes = pickle.load(handle)

class roi:
    def __init__(self):
        self.poly = None
        self.clas = None

#cv2.namedWindow("input")
target_x, target_y = 1936, 1296
mypath = 'collection/Annotations/users/his109/grier_food_images/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
with open('supervis.csv', 'a') as fp:

    writer = csv.writer(fp, delimiter=',')
    for img33 in onlyfiles:
        rois = []
        classes["bkg"] = 0
        document = ElementTree.parse( mypath + img33)


        membership = document.getroot()
        filename = membership.find("filename").text
        imgsize = membership.find("imagesize")
        orig_x = int(imgsize.find("ncols").text)
        orig_y = int(imgsize.find("nrows").text)
        print([orig_x, orig_y])
        imgpath = "collection/Images/" + membership.find("folder").text +"/"+ filename
        imgpath = imgpath.replace("//", "/")
        for user in membership.findall( 'object' ):
            if user.find("deleted").text == "1":
                continue

            points = []
            for pt in user.find("polygon").findall( 'pt' ):
                x_ref = int(pt.find("x").text)
                y_ref = int(pt.find("y").text)
                #target_x    =     orig_x
                #x_ref       =     x

                x_ref = (target_x*x_ref)//orig_x
                y_ref = (target_y * y_ref) // orig_y
                points.append((x_ref,y_ref))
            print(points)
            clss = user.find("name").text
            if clss not in classes:
                classes[clss] = len(classes)
            th = roi()
            if len(points) < 4 :
                continue
            if len(points) >8 :
                continue
            th.poly = Polygon(points)
            th.clas = classes[clss]
            rois.append(th)
            #labels.append(classes[clss])
            #a.writerow(data)

        img = cv2.imread(imgpath)
        print(imgpath)
        #cv2.imshow("cropped", img)
        #cv2.waitKey(0)

        image = cv2.resize(img, (target_x, target_y))
        #image = copy.deepcopy(img)
        #image = cv.resize(img, (250, 250))
        if image is None:
            continue
        y_class = []
        for y in range(0,len(image),23):
            x_class = []
            for x in range(0,len(image[0]),35):
                x_adds_max = 35*6
                x_adds_min = 35*6
                y_adds_max = 23*6
                y_adds_min = 23*6
                x_adds = x_adds_max
                y_adds = y_adds_max
                #print(y)

                #
                rect = box(x+x_adds,y+y_adds, x, y, False)
                #print(rect.bounds)
                #print(rect.bounds[0])
                intersects = []
                added = 0
                for clse in rois:
                    if clse.clas == 1:
                        continue
                    if clse.poly.intersects(rect):
                        #intersects.append(clse)
                        x_class.append(clse.clas)
                        added = 1
                        break
                if added == 0:
                    x_class.append(0)
                #print(x_class[-1])
                crop_img = image[int(rect.bounds[1]):int(rect.bounds[3]), int(rect.bounds[0]):int(rect.bounds[2])] # Crop from x, y, w, h -> 100, 200, 300, 400
                #cv2.imshow('input',crop_img)
                #cv2.waitKey(0)

                #x_class.append(list(clser)[0])
                #classg = str(list(classes.keys())[list(classes.values()).index(list(clser)[0])])
                #if classg == "plate":
                #    continue

                #name = directory + "/face-" + filename + "-" + str(y) + "-" + str(y)  + "-" + str(randint(0,9000)) + ".jpg"
                #cv2.imwrite(name, rdata)
                #print(clser)
            y_class.append(x_class)


        print(classes)
        writer.writerow([filename, y_class])
        print(y_class)
        p1=Polygon([(0,0),(1,1),(1,0),(0,1)])
        p2=Polygon([(0,2),(1,0),(1,1),(1,0), (2,0)])
        print (p1.intersects(p2))
        #input()

with open('classesROI.pickle', 'wb') as handle:
    pickle.dump(classes, handle)
