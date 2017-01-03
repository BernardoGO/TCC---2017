
import par_config
from PIL import ImageFont, ImageDraw
from PIL import Image
import time
import numpy as np
import core.model
import core.roi_management
import utils.colors
import utils.classes
import data_management.dataset_io
import copy

print("Loading Data...")
trainX, trainY = data_management.dataset_io.getData()

print("Initializing Model...")
ldModel = core.model.initializeModel()

ty = copy.deepcopy(trainY)
ty = np.array(ty)
ty = np.eye(25, dtype='uint8')[ty]
#ty = ty.reshape((1,)+ty.shape)
print("Fit...")
batch_size = 32
nb_epoch = 200



ldModel.fit(np.array(trainX), np.array(ty),
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          callbacks=[tensorboard]
         )

ldModel.save_weights('image_500_retrIII.h5')
save_model(ldModel, "model_500_retrIII.h5")
