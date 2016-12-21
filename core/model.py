from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import save_model, load_model
from keras.callbacks import TensorBoard
import par_config



def initializeModel():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, par_config.img_width, par_config.img_height)
    else:
        input_shape = (par_config.img_width, par_config.img_height, 3)
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
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    model.load_weights("image_500.h5")
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()
    model.pop()


    model.add(Conv2D(16, (6, 6),strides=(par_config.STRIDES, par_config.STRIDES), trainable=True))
    #model.add(Activation('relu', trainable=True))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='softmax'))
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


"""

import copy
print("Initializing Model...")
ldModel = initializeModel()
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
"""
