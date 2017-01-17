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
sizex, sizey = 1936, 1296
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 150000
nb_validation_samples = 24000
epochs = 60
batch_size = 16

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
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.load_weights("image_500.h5")




tstImg = "data/validation/0/face-dsc_1570.jpg-0-0-556.jpg"
tstImg = "data/validation/1/face-dsc_1603.jpg-950-950-7319.jpg"
tstImg = "dsc_1570.jpg"
img = Image.open(tstImg)
image = img.resize((img_width, img_height), Image.ANTIALIAS)#

width, height = image.size

train_x = []
train_x.append(np.array(image))










get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[8].output])
layer_output = get_3rd_layer_output([np.array(train_x)])[0]
print(sum(sum(sum(layer_output))))
print(layer_output)

onlyres = np.sum(layer_output[0], axis=2)

print(onlyres)

print("drawing heatmap")
image = Image.new("RGB", (sizex, sizey), "yellow" )#img_width, img_height
#rect_pos = class_scores
#colors = get_spaced_colors(10)
dr = ImageDraw.Draw(image , 'RGBA')

#17     -     150
#y      -      x
#x = (150y)//17



for x_ in range(len(onlyres)):
    for y_ in range(len(onlyres[x_])):
        color = onlyres[x_][y_]
        print(onlyres[x_][y_])
        #(150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17
        x0 = (img_width*y_)//60
        y0 = (img_height*x_)//60
        x1 = (img_width*(y_+1))//60
        y1 = (img_height*(x_+1))//60

        x0 = (x0*sizex)/img_width
        y0 = (y0*sizey)/img_height
        x1 = (x1*sizex)/img_width
        y1 = (y1*sizey)/img_height
        dr.rectangle(((x0,y0),(x1,y1)), fill=(color, color, color, int(255)), outline = None)
        #print(((150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17))
#posx = 20
#posy = 20
#for x in classes_in_image:
#    posy += 40
#    #color = colors[classes_in_image[x][6]]
#    #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
image.save("rcnn.png", quality=100)
