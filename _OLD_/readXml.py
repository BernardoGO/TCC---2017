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

mypath = 'collection/Annotations/users/his109/grier_food_images/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
with open('supervis.csv', 'a') as fp:

    a = csv.writer(fp, delimiter=',')
    for img33 in onlyfiles:
        rois = []
        classes["bkg"] = 0
        document = ElementTree.parse( mypath + img33)


        membership = document.getroot()
        filename = membership.find("filename").text
        imgpath = "collection/Images/" + membership.find("folder").text +"/"+ filename
        imgpath = imgpath.replace("//", "/")
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

        image = cv2.resize(img, (1936, 1296))
        #image = copy.deepcopy(img)
        #image = cv.resize(img, (250, 250))
        if image is None:
            continue
        for y in range(0,len(image),50):
            for x in range(0,len(image[0]),50):
                x_adds_max = 125
                x_adds_min = 125
                y_adds_max = 125
                y_adds_min = 125
                x_adds = x_adds_max
                y_adds = y_adds_max
                #print(y)

                #
                rect = box(x+x_adds,y+y_adds, x, y, True)
                #print(rect.bounds)
                #print(rect.bounds[0])
                crop_img = image[int(rect.bounds[1]):int(rect.bounds[3]), int(rect.bounds[0]):int(rect.bounds[2])] # Crop from x, y, w, h -> 100, 200, 300, 400
                rdata = cv2.resize(crop_img, (128, 128))
                image = copy.deepcopy(img)
                intersects = []
                for clse in rois:
                    if clse.poly.intersects(rect):
                        intersects.append(clse)
                        break

                clser = set([ue.clas for ue in intersects])
                if len(clser) == 0:
                    #if randint(0,100) != 50:
                    #    continue
                    clser = [classes["bkg"]]
                classg = str(list(classes.keys())[list(classes.values()).index(list(clser)[0])])
                if classg == "plate":
                    continue
                directory = "fd/"+classg
                directory = directory.replace(" ", "_")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                name = directory + "/face-" + filename + "-" + str(y) + "-" + str(y)  + "-" + str(randint(0,9000)) + ".jpg"
                cv2.imwrite(name, rdata)
                data = [name, list(clser)]
                #print (data )
                #a.writerow(data)
                #print([ str(list(classes.keys())[list(classes.values()).index(xw.clas)])  for xw in intersects])
                #cv2.rectangle(image, (int(rect.bounds[0]),int(rect.bounds[1])), (int(rect.bounds[2]), int(rect.bounds[3])), (255,255,0), 2)
                #cv2.imshow("cropped", image)
                #cv2.imshow("cropped2", crop_img)
                #cv2.waitKey(0)


        print(classes)
        p1=Polygon([(0,0),(1,1),(1,0),(0,1)])
        p2=Polygon([(0,2),(1,0),(1,1),(1,0), (2,0)])
        print (p1.intersects(p2))

with open('classesROI.pickle', 'wb') as handle:
    pickle.dump(classes, handle)
