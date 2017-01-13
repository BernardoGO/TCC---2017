from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
img_width, img_height = 128, 128

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#model = load_weights('first_try.h5')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.load_weights('first_try.h5')
image = []
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1624.jpg-800-800-3329.jpg")))#
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1613.jpg-1050-1050-8019.jpg")))

#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1699.jpg-800-800-6760.jpg")))
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1741.jpg-1100-1100-8953.jpg")))
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1741.jpg-1150-1150-390.jpg")))

#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1656.jpg-100-100-5668.jpg")))
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1660.jpg-1050-1050-2744.jpg")))
#image.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1662.jpg-150-150-8816.jpg")))
from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='/home/bernardo/projects/tcc/keras/data/validation/bkg'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = []
print("Count: " + str(len(onlyfiles)))
for n in range(0, len(onlyfiles)):
    #print("read")
    images.append(np.array(cv2.imread( join(mypath,onlyfiles[n]) )))

train_x = np.array(images)
yFit = model.predict_classes(train_x, batch_size=10, verbose=1)
print()
print(yFit)
