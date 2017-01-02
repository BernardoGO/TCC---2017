import par_config
import csv
from PIL import ImageFont, ImageDraw
from PIL import Image

def getData():
    #csv = pd.read_csv("supervis.csv")
    dataX = []
    dataY = []
    with open(par_config.dataset_csv_path+par_config.dataset_csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[0]
            img = Image.open(par_config.imagePath+filename)
            image = img.resize((img_width, img_height ), Image.ANTIALIAS)  #
            imge = np.array(image)
            dataX.append(imge)
            dataY.append(np.array(eval(row[1]))[0:55,0:55])
            ie = 9
        #print(len(dataX))
    return [dataX, dataY]


"""
print("Loading Data...")
trainX, trainY = getData()


# In[7]:


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

ldModel.save_weights('image_500_retr.h5')
save_model(ldModel, "model_500_retr.h5")
"""

#ldModel.load_weights('image_500_retr.h5')
