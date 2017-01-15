
# coding: utf-8

# In[2]:

import csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import save_model
from PIL import ImageFont, ImageDraw
from PIL import Image
from PIL import ImageDraw
import numpy as np

# dimensions of our images.
img_width, img_height = 500, 500
sizex, sizey = 1936, 1296#3024,4032


epochs = 60
batch_size = 16
imagePath = "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"




# In[3]:

def get_spaced_colors(n):
    print("generating colors:" + str(n))
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


# In[4]:

def getData():
    #csv = pd.read_csv("supervis.csv")
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
            dataY.append(np.array(eval(row[1]))[0:55,0:55])
            ie = 9
        print(len(dataX))
    return [dataX, dataY]


# In[5]:

def initializeModel():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, trainable=False))
    model.add(Activation('relu', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), trainable=True))
    model.add(Activation('relu', trainable=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), trainable=True))
    model.add(Activation('relu', trainable=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

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


    model.add(Conv2D(1, (6, 6),strides=(1, 1), trainable=True))

    #model.add(Activation('relu', trainable=True))
    #model.add(Activation('relu', trainable=True))
    #model.add(Conv2D(1, (6, 6),strides=(1, 1), trainable=True))
    #model.add(Activation('relu', trainable=True))
    model.add(Dense(21, activation='relu'))
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


# In[6]:

print("Initializing Model...")
ldModel = initializeModel()


# In[7]:


print("Loading Data...")
trainX, trainY = getData()


# In[9]:

import copy
ty = copy.deepcopy(trainY)
ty = np.array(ty)
ty = np.eye(25, dtype='uint8')[ty]
#ty = ty.reshape((1,)+ty.shape)
print("Fit...")
batch_size = 16
nb_epoch = 8000

ldModel.fit(np.array(trainX), np.array(ty),
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2

         )

ldModel.save_weights('image_500_retr.h5')
save_model(ldModel, "model_500_retr.h5")


# In[10]:
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

def testImage():
    tstImg = "dsc_1570.jpg"
    img = Image.open(tstImg)
    image = img.resize((img_width, img_height), Image.ANTIALIAS)#
    fonet = ImageFont.truetype("sans-serif.ttf", 18)
    width, height = image.size
    train_x = []
    train_x.append(np.array(image))
    rpn_output = ldModel.predict(np.array(train_x), batch_size=24, verbose=0)
    onlyres = rpn_output[0]
    print("drawing heatmap")
    image = Image.new("RGB", (sizex, sizey), "yellow" )#img_width, img_height
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
        for x_ in range(len(onlyres[y_])):
            print(onlyres[y_][x_])
            clase = np.argmax(onlyres[y_][x_])
            #color = colors[clase]
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
            class_scores.append([x0,y0,x1,y1,clase])
            #to print ROIS
            #dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(10)), outline = None)
            #dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)


            #image.save("rcnn2.png", quality=100)
            #time.sleep(0.5)
            #print(((150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17))
    joinROIS(class_scores)
    rect_pos = class_scores
    #colors = get_spaced_colors(len(classes_in_image))
    dr = ImageDraw.Draw(img , 'RGBA')
    font = ImageFont.truetype("sans-serif.ttf", 18)
    for ie in range(len(rect_pos)):
        #eft = classes_in_image[rect_pos[ie][4]]
        #xxf = eft[6]-1
        #print(len(colors))
        #print(xxf)
        color = colors[ie[4]]
        #print(color)
        #print(classes_in_image[rect_pos[ie][4]][6])
        #if rect_pos[ie][2]-rect_pos[ie][0] == x_adds_max:
        #    continue
        #elif rect_pos[ie][3]-rect_pos[ie][1] == y_adds_max:
        #    continue

        #classesinit = str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0])
        #print(classesinit)
        dr.rectangle(((rect_pos[ie][0]+randint(0,10), rect_pos[ie][1]+randint(0,10)),(rect_pos[ie][2]+randint(0,10),rect_pos[ie][3]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
        dr.text((int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))),str(ie[4]),(color[0], color[1], color[2]),font=font)
    #posx = 20
    #posy = 20
    #for x in classes_in_image:
    #    posy += 40
    #    #color = colors[classes_in_image[x][6]]
    #    #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
    img.save("output.png", quality=100)
    image.save("rcnn2.png", quality=100)
    print("ok")
testImage()


# In[ ]:
