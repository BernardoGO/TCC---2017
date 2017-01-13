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
#model.load_weights('first_try.h5')

from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='/home/bernardo/projects/tcc/keras/data/validation/1'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = []

print("Count: " + str(len(onlyfiles)))
for n in range(0, 20):#len(onlyfiles)
    #print("read")
    img = cv2.imread( join(mypath,onlyfiles[n]) )
    res = cv2.resize(img, (150, 150))
    images.append(np.array(res))

print("initiated")
"""

images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1624.jpg-800-800-3329.jpg")))#
images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1613.jpg-1050-1050-8019.jpg")))

images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1699.jpg-800-800-6760.jpg")))
images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1741.jpg-1100-1100-8953.jpg")))
images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/roi/face-dsc_1741.jpg-1150-1150-390.jpg")))

images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1656.jpg-100-100-5668.jpg")))
images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1660.jpg-1050-1050-2744.jpg")))
images.append(np.array(cv2.imread("/home/bernardo/projects/tcc/keras/data/validation/bkg/face-dsc_1662.jpg-150-150-8816.jpg")))
"""
train_x = np.array(images)
yFit = model.predict_proba(train_x, batch_size=8, verbose=0)
print(sum(yFit))
print(yFit)
