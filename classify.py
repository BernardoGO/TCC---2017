import par_config
from random import randint
from PIL import ImageFont, ImageDraw
from PIL import Image
import time
import numpy as np
import core.model
import core.roi_management
import utils.colors


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
    fonet = ImageFont.truetype("sans-serif.ttf", 18)
    width, height = image.size
    train_x = []
    train_x.append(np.array(image))


    print("Initializing Model...")
    ldModel = core.model.initializeModel()


    #start = time.time()
    rpn_output = ldModel.predict(np.array(train_x), batch_size=64, verbose=0)
    #end = time.time()

    #elapsed = end - start
    #print("Eval Time: " + str(elapsed))

    onlyres = rpn_output[0]
    print("Drawing Heatmap")
    image = Image.new("RGB", (sizex, sizey), "black" )#img_width, img_height

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
    colors = utils.colors.get_spaced_colors(25)
    for y_ in range(len(onlyres)):
        last_x = 0
        last_x_ct = 0
        for x_ in range(len(onlyres[y_])):
            #print(onlyres[y_][x_])
            clase = np.argmax(onlyres[y_][x_])
            if onlyres[y_][x_][clase] < 0.80:
                clase = 0
            color = colors[clase]
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

            x0 = 35*(x_*par_config.STRIDES)
            y0 = 23*(y_*par_config.STRIDES)
            x1 = 35*(x_*par_config.STRIDES) + 35*6
            y1 = 24*(y_*par_config.STRIDES) + 24*6
            if clase != 0:
                last_x_ct += 1
                if (last_x != clase) or (last_x == clase and last_x_ct > 10):
                    class_scores.append([x0,y0, 35*(x_*par_config.STRIDES) + 35*2,24*(y_*par_config.STRIDES) + 24*2,clase])
                    last_x = clase
                    last_x_ct = 0

            #to print ROIS
            dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(10)), outline = None)
            dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)

            #dr.rectangle(((x0,y0),(x1,y1)), fill=(color[0], color[1], color[2], int(10)), outline = None)
            #dr.text((x0+1,y0+1),str(clase),(color[0], color[1], color[2]),font=fonet)
            #image.save("rcnn2.png", quality=100)
            #input([(x0,y0),(x1,y1)])
            #image.save("rcnn2.png", quality=100)
            #time.sleep(0.5)
            #print(((150*y_)//17,(150*x_)//17),((150*(y_+1))//17,(150*(x_+1))//17))
    core.roi_management.joinROIS(class_scores)
    rect_pos = class_scores
    #colors = get_spaced_colors(len(classes_in_image))

    dri = ImageDraw.Draw(ime , 'RGBA')
    font = ImageFont.truetype("sans-serif.ttf", 18)
    core.roi_management.compareROIs(IMAGENAME + ".xml", dri, rect_pos)
    for ie in range(len(rect_pos)):
        #eft = classes_in_image[rect_pos[ie][4]]
        #xxf = eft[6]-1
        #print(len(colors))
        #print(xxf)
        color = colors[rect_pos[ie][4]]
        #print(color)
        #print(classes_in_image[rect_pos[ie][4]][6])
        if rect_pos[ie][2]-rect_pos[ie][0] == 35*6:
            continue
        elif rect_pos[ie][3]-rect_pos[ie][1] == 24*6:
            continue

        #classesinit = str( rect_pos[ie][4]) + ": " + str(rect_pos[ie][5]) + "-" + str(rect_pos[ie][2]-rect_pos[ie][0])
        #print(classesinit)
        dri.rectangle(((rect_pos[ie][0]+randint(0,10), rect_pos[ie][1]+randint(0,10)),(rect_pos[ie][2]+randint(0,10),rect_pos[ie][3]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
        dri.text((int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))),str(rect_pos[ie][4]),(color[0], color[1], color[2]),font=font)
    #posx = 20
    #posy = 20
    #for x in classes_in_image:
    #    posy += 40
    #    #color = colors[classes_in_image[x][6]]
    #    #dr.text((posx,posy),str(x),(color[0], color[1], color[2]),font=font)
    ime.save("output.png", quality=100)
    image.save("output2.png", quality=100)
    print("ok")

import time


testImage()
