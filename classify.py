import par_config

from PIL import ImageFont, ImageDraw
from PIL import Image
import time
import numpy as np
import core.model
import core.roi_management
import utils.colors
import utils.classes
import utils.set_operations


utils.classes.load_classes()



def testImage():

    img_width = par_config.img_width
    img_height = par_config.img_height
    sizex = par_config.sizex
    sizey = par_config.sizey

    IMAGENAME = par_config.test_images_folder+ par_config.test_image_name
    tstImg = IMAGENAME + ".jpg"
    #tstImg = "dsc_1734.jpg"
    img = Image.open(tstImg)
    image = img.resize((img_width, img_height), Image.ANTIALIAS)#
    ime = img.resize((sizex, sizey), Image.ANTIALIAS)#
    fonet = ImageFont.truetype(par_config.font_filename, 18)
    width, height = image.size
    train_x = []
    train_x.append(np.array(image))


    print("Initializing Model...")
    ldModel = core.model.initializeModel()

    ldModel.load_weights(par_config.final_weights_filename)

    start = time.time()
    rpn_output = ldModel.predict(np.array(train_x), batch_size=par_config.classify_batchsize, verbose=0)
    end = time.time()

    elapsed = end - start
    print("Eval Time: " + str(elapsed))

    onlyres = rpn_output[0]
    print("Drawing Heatmap")
    image = Image.new("RGB", (sizex, sizey), "black" )#img_width, img_height

    dr = ImageDraw.Draw(image , 'RGBA')

    class_scores = []
    colors = utils.colors.get_spaced_colors(25)
    for y_ in range(len(onlyres)):
        last_x = 0
        last_x_ct = 0
        for x_ in range(len(onlyres[y_])):

            clase = np.argmax(onlyres[y_][x_])
            if onlyres[y_][x_][clase] < par_config.backgroundThreshold:
                clase = 0
            color = colors[clase]

            x0 = 35*x_
            y0 = 23*y_
            x1 = 35*x_ + 35*6
            y1 = 24*y_ + 24*6

            x0 = 35*(x_*par_config.STRIDES)
            y0 = 23*(y_*par_config.STRIDES)
            x1 = 35*(x_*par_config.STRIDES) + 35*6
            y1 = 24*(y_*par_config.STRIDES) + 24*6
            if clase != 0:
                last_x_ct += 1
                if (last_x != clase) or (last_x == clase and last_x_ct > 10) or par_config.consider_full_conv == True:
                    bbox = {}
                    bbox["class"] = clase
                    bbox["xmin"] = x0
                    bbox["ymin"] = y0
                    bbox["xmax"] = 35*(x_*par_config.STRIDES) + 35*2
                    bbox["ymax"] = 24*(y_*par_config.STRIDES) + 24*2

                    #class_scores.append([bbox["xmin"],bbox["ymin"], bbox["xmax"],bbox["ymax"],clase])
                    class_scores.append(bbox)
                    last_x = clase
                    last_x_ct = 0

            #to print ROIS
            dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(10)), outline = None)
            dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)


    core.roi_management.joinROIS(class_scores)
    rect_pos = []
    for x in class_scores:
        area = abs(utils.set_operations.area(x))
        if area > 200000:
            rect_pos.append(x)


    dri = ImageDraw.Draw(ime , 'RGBA')
    if par_config.compare_RoIs == True:
        core.roi_management.compareROIs(IMAGENAME + ".xml", dri, rect_pos)
    core.roi_management.draw_boundingboxes(rect_pos,dri,colors)

    ime.save("output.png", quality=100)
    image.save("output2.png", quality=100)
    print("ok")



testImage()
