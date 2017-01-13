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
import time
import matplotlib.pyplot as plt
import copy
from random import randint
import xml.etree.ElementTree
import pickle
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
img_width, img_height = 150, 150



model = load_model('model.h5')
sizex, sizey = 1936, 1296

def get_spaced_colors(n):
    print("generating colors:" + str(n))
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
def joinROIS(class_scores):
    print("joining rois")
    foid = True
    while foid:
        breako = False
        for ii in class_scores:
            for xx in class_scores:
                if xx == ii: continue
                #minx, miny, maxx, maxy
                rectxx = box(xx[1],xx[0], xx[3], xx[2], False)
                rectii = box(ii[1],ii[0], ii[3], ii[2], False)
                #(not (ii[0] < xx[0]) or (ii[1] < xx[1]) or (ii[2] > xx[2]) or (ii[3] > xx[3])) and
                if rectxx.intersects(rectii) :#and (ii[4] == xx[4])
                    #print([rectxx.bounds,rectii.bounds])
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

def heatmap(rect_pos1):
    print("drawing heatmap")
    image = Image.new("RGB", (sizex, sizey), "black" )
    #rect_pos = class_scores
    colors = get_spaced_colors(10)
    dr = ImageDraw.Draw(image , 'RGBA')
    font = ImageFont.truetype("sans-serif.ttf", 18)
    for ie in range(len(rect_pos1)):

        color = colors[5]



        dr.rectangle(((rect_pos1[ie][1]+randint(0,10), rect_pos1[ie][0]+randint(0,10)),(rect_pos1[ie][3]+randint(0,10),rect_pos1[ie][2]+randint(0,10))), fill=(color[0], color[1], color[2], int(1)), outline = None)
    #posx = 20
    #posy = 20
    #for x in classes_in_image:
    #    posy += 40
    #    #color = colors[classes_in_image[x][6]]
    #    #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
    image.save("heatmap.png", quality=100)

tstImg = "dsc_1734.jpg"
#tstImg = "dsc_1601.jpg"


print("runnning high level detection")
x_adds_max = 150
x_adds_min = 150
y_adds_max = 150
y_adds_min = 150
x_adds = x_adds_max
y_adds = y_adds_max

img = Image.open(tstImg)
image = img.resize((sizex, sizey), Image.ANTIALIAS)#
truth = copy.deepcopy(image)
width, height = image.size
nexte = 0
start = time.time()
train_x = []
pos = []
for y in range(0,height,150):
    for x in range(0,width,150):

        crop_img = image.crop((x, y, x+x_adds, y+y_adds))
        rdata = crop_img
        pos.append([y, x, y+y_adds, x+x_adds])
        train_x.append(np.array(rdata))

end = time.time()
elapsed = end - start
print("elapsed time(sliding): " + str(elapsed))
start = time.time()
class_scores_lv1 = []
yFit = model.predict(np.array(train_x), batch_size=24, verbose=0)
end = time.time()

elapsed = end - start
#compareROIs("dsc_1703.xml", dr, rect_pos)
print("elapsed time(eval): " + str(elapsed))

for prob in range(len(yFit)):
    if yFit[prob] >= 1:#+((pos[prob][1]+pos[prob][0])*0.0001)
        #print( yFit[0] )
        #nexte = -0.01
        class_scores_lv1.append(pos[prob])

heatmap(class_scores_lv1)
#joinROIS(class_scores_lv1)
print("drawing")

rect_pos = class_scores_lv1
colors = get_spaced_colors(10)
dr = ImageDraw.Draw(image , 'RGBA')
font = ImageFont.truetype("sans-serif.ttf", 18)
print(len(rect_pos))
for ie in range(len(rect_pos)):
    #print(len(colors))
    #print(xxf)
    color = colors[5]
    #print(color)
    #print(classes_in_image[rect_pos[ie][4]][6])


    #classesinit = str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0])
    #print(classesinit)
    dr.rectangle(((rect_pos[ie][1]+randint(0,10), rect_pos[ie][0]+randint(0,10)),(rect_pos[ie][3]+randint(0,10),rect_pos[ie][2]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
    #dr.text((int(rect_pos[ie][1]+5),int(rect_pos[ie][0]+(randint(0,80)))),classesinit,(color[0], color[1], color[2]),font=font)
    #cv2.rectangle(image, (rect_pos[ie][0]+randint(0,10), rect_pos[ie][1]+randint(0,10)), (rect_pos[ie][2]+randint(0,10),rect_pos[ie][3]+randint(0,10)), (color[0], color[1], color[2]), 2)
    #cv2.putText(image,str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0]),(int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))), font, 0.5,(color[0], color[1], color[2]),2)


#compareROIs("dsc_1703.xml", dr, rect_pos)

image.save("output.png", quality=100)
