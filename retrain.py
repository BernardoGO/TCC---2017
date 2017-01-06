
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
from keras.callbacks import TensorBoard
from keras.models import save_model, load_model

print("Loading Data...")
trainX, trainY = data_management.dataset_io.getData()

print("Initializing Model...")
ldModel = core.model.initializeModel()

ty = copy.deepcopy(trainY)
ty = np.array(ty)
ty = np.eye(par_config.final_class_count, dtype='uint8')[ty]
#ty = ty.reshape((1,)+ty.shape)
print("Fit...")
batch_size = par_config.batch_size
nb_epoch = par_config.retrain_epochs

tensorboard = TensorBoard(log_dir=par_config.tensorboard_file, histogram_freq=0, write_graph=True, write_images=True)

ldModel.fit(np.array(trainX), np.array(ty),
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          callbacks=[tensorboard]
         )

ldModel.save_weights(par_config.final_weights_filename)
save_model(ldModel, par_config.final_model_filename)
