'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
from keras.models import save_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import keras
from skimage import io
import os
import numpy as np
import glob
import numpy as np
from skimage import color, exposure, transform
NUM_CLASSES = 3
def get_class(img_path):
    print(img_path)
    return int(img_path.split('/')[-2])

root_dir = 'data/train/'
imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = io.imread(img_path)
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train/'
validation_data_dir = 'data/validation/'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = cnn_model()
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 2

model.fit(X, Y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('model.h5',save_best_only=True)]
         )

model.save_weights('first_try.h5')
save_model(model, "model2.h5")


"""
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import numpy as np

mypath='/home/bernardo/projects/tcc/keras/data/train/bkg'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = []
print("Count: " + str(len(onlyfiles)))
for n in range(0, len(onlyfiles)):
    #print("read")
    images.append(np.array(cv2.imread( join(mypath,onlyfiles[n]) )))

train_x = np.array(images)
yFit = model.predict_proba(train_x, batch_size=10, verbose=0)
print(sum(yFit))
print(yFit)

"""
