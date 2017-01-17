
# coding: utf-8

# In[1]:

import csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import save_model, load_model
from PIL import ImageFont, ImageDraw
from PIL import Image
from PIL import ImageDraw
import os
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.callbacks import TensorBoard
# dimensions of our images.
img_width, img_height = 500, 500
sizex, sizey = 1936, 1296#3024,4032
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
import cv2
imagePath = "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"

train = False
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# In[2]:

def get_spaced_colors(n):
    print("generating colors:" + str(n))
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


# In[3]:

def getData():
    #csv = pd.read_csv("is.csv")
    dataX = []
    dataY = []
    with open("supervis.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[0]
            img = Image.open(imagePath+filename)
            image = img.resize((img_width, img_height ), Image.ANTIALIAS)  #
            imge = np.array(image)
            dataX.append(imge)
            #print(len(eval(row[1])))
            dataY.append(np.array(eval(row[1]))[0:55,0:55])
            ie = 9
        #print(len(dataX))
    return [dataX, dataY]


# In[4]:
STRIDES = 1
def initializeModel():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, trainable=False))
    model.add(Activation('relu', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), trainable=False))
    model.add(Activation('relu', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), trainable=True))
    model.add(Activation('relu', trainable=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    model.load_weights("weights.best.hdf5")
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()


    model.add(Conv2D(32, (6, 6),strides=(STRIDES, STRIDES), trainable=True))
    #model.add(Activation('relu', trainable=True))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())



    model.add(Dense(21, activation='softmax'))
    """
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    """

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.summary()
    return model


prefix = "BIN_SOFTMAX_128x512_"
# In[5]:
import copy
print("Initializing Model...")
ldModel = initializeModel()
if train == True:
    tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # In[6]:


    print("Loading Data...")
    trainX, trainY = getData()


    # In[7]:


    ty = copy.deepcopy(trainY)
    ty = np.array(ty)
    ty = np.eye(21, dtype='uint8')[ty]
    #ty = ty.reshape((1,)+ty.shape)
    print("Fit...")
    batch_size = 32
    nb_epoch = 500
    filepath=prefix+"image_500_retr.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    ldModel.fit(np.array(trainX), np.array(ty),
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.2,
              callbacks=[tensorboard,checkpoint]
             )

    #ldModel.save_weights('image_500_retr.h5')
    save_model(ldModel, prefix+"model_500_retr_xxx.h5")
else:
    #ldModel = load_model("model_500_retr.h5")
    ldModel.load_weights(prefix+'image_500_retr.h5')
    # In[ ]:

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

def joinROIS(class_scores):
    print("RoI Count: " + str(len(class_scores)))
    from shapely.geometry import Polygon, box
    print("joining rois")
    foid = True
    coverage = 0.3

    while foid:
        breako = False
        #if coverage <0.90:
        #    coverage += 0.01
        for ii in class_scores:
            for xx in class_scores:
                if xx == ii: continue
                rectxx = box(xx[3],xx[2], xx[1], xx[0], True)
                rectii = box(ii[3],ii[2], ii[1], ii[0], True)
                #iou = intersectionOverUnion(xx,ii)

                #rectxx.intersects(rectii)
                if rectxx.intersects(rectii) and (ii[4] == xx[4]):
                    #print(coverage)
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
    """

    for ii in class_scores:
        overlaps = []
        for xx in class_scores:
            iou = intersectionOverUnion(xx,ii)
            if iou > coverage and (ii[4] == xx[4]):
                overlaps.append(xx)
        oo = [0,0,0,0,None]
        print(overlaps)
        if len(overlaps) == 0:
            continue
        oo[0] = min([x[0] for x in overlaps])
        oo[1] = min([x[1] for x in overlaps])
        oo[2] = max([x[2] for x in overlaps])
        oo[3] = max([x[3] for x in overlaps])
        oo[4] = overlaps[0][4]
        for x in overlaps:
            class_scores.remove(x)
        class_scores.append(oo)
    """
    return class_scores


def testImage():
    import time
    tstImg = "test/000022.jpg"
    #tstImg = "dsc_1734.jpg"
    img = Image.open(tstImg)
    image = img.resize((img_width, img_height), Image.ANTIALIAS)#
    ime = img.resize((sizex, sizey), Image.ANTIALIAS)#
    fonet = ImageFont.truetype("sans-serif.ttf", 18)
    width, height = image.size
    train_x = []
    train_x.append(np.array(image))
    start = time.time()
    rpn_output = ldModel.predict(np.array(train_x), batch_size=64, verbose=0)
    end = time.time()

    elapsed = end - start
    print("Eval Time: " + str(elapsed))

    onlyres = rpn_output[0]
    print("drawing heatmap")
    image = Image.new("RGB", (sizex, sizey), "black" )#img_width, img_height

    #rect_pos = class_scores
    #colors = get_spaced_colors(10)
    dr = ImageDraw.Draw(image , 'RGBA')

    #17     -     150
    #y      -      x
    #x = (150y)//17
    """
    img_width, img_height = 500, 500
    sizex, sizey = 1936, 1296
    x0x = (img_width * 1) // 55
    y0y = (img_height * 1) // 55
    x0x = (x0x*sizex)/img_width
    y0y = (y0y*sizey)/img_height
    """
    import time
    class_scores = []
    colors = get_spaced_colors(25)
    for y_ in range(len(onlyres)):
        last_x = 0
        last_x_ct = 0
        for x_ in range(len(onlyres[y_])):
            #print(onlyres[y_][x_])
            clase = np.argmax(onlyres[y_][x_])
            if onlyres[y_][x_][clase] < 0.35:
                clase = 0
            color = colors[clase]
            #print(onlyres[y_][x_])
            #(150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17
            #img_width, img_height = 500, 500
            #sizex, sizey = 1936, 1296
            #35.2
            #23.56
            #55x55
            """
            x0 = (img_width*x_)//55
            y0 = (img_height*y_)//55
            x1 = (img_width*(x_+5))//55
            y1 = (img_height*(y_+5))//55

            x0 = (x0*sizex)/img_width
            y0 = (y0*sizey)/img_height
            x1 = ((x1*sizex)/img_width)
            y1 = ((y1*sizey)/img_height)
            """
            x0 = 35*x_
            y0 = 23*y_
            x1 = 35*x_ + 35*6
            y1 = 24*y_ + 24*6

            x0 = 35*(x_*STRIDES)
            y0 = 23*(y_*STRIDES)
            x1 = 35*(x_*STRIDES) + 35*6
            y1 = 24*(y_*STRIDES) + 24*6
            if clase != 0:
                last_x_ct += 1
                #if (last_x != clase) or (last_x == clase and last_x_ct > 10):
                class_scores.append([x0,y0, x1,y1,clase])
                last_x = clase
                last_x_ct = 0

            #to print ROIS
                dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(200)), outline = None)
                dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)

            #dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(10)), outline = None)
            #dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)
            #image.save("rcnn2.png", quality=100)
            #input([(x0,y0),(x1,y1)])
            #image.save("rcnn2.png", quality=100)
            #time.sleep(0.5)
            #print(((150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17))

    rect_pos = joinROIS(class_scores)
    #colors = get_spaced_colors(len(classes_in_image))
    from random import randint
    dri = ImageDraw.Draw(ime , 'RGBA')
    font = ImageFont.truetype("sans-serif.ttf", 18)
    for ie in range(len(rect_pos)):
        #eft = classes_in_image[rect_pos[ie][4]]
        #xxf = eft[6]-1
        #print(len(colors))
        #print(xxf)
        color = colors[rect_pos[ie][4]]
        #print(color)
        #print(classes_in_image[rect_pos[ie][4]][6])
        #if rect_pos[ie][2]-rect_pos[ie][0] == 35*6:
        #    continue
        #elif rect_pos[ie][3]-rect_pos[ie][1] == 24*6:
        #    continue

        #classesinit = str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0])
        #print(classesinit)
        dri.rectangle(((rect_pos[ie][0], rect_pos[ie][1]),(rect_pos[ie][2],rect_pos[ie][3])), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
        dri.text((int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))),str(rect_pos[ie][4]),(color[0], color[1], color[2]),font=font)
    #posx = 20
    #posy = 20
    #for x in classes_in_image:
    #    posy += 40
    #    #color = colors[classes_in_image[x][6]]
    #    #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
    ime.save("output.png", quality=100)
    image.save("output2.png", quality=100)
    print("ok")

import time


testImage()




# In[ ]:
