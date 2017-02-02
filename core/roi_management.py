import par_config
import logging as log
import utils.set_operations
import xml.etree.ElementTree
import utils.colors
from random import randint
from PIL import ImageFont, ImageDraw

def draw_boundingboxes(rect_pos,dri,colors):
    font = ImageFont.truetype(par_config.font_filename, 18)
    for ie in range(len(rect_pos)):

        color = colors[rect_pos[ie][4]]
        if par_config.ignore_notJoined_boxes == True:
            if rect_pos[ie][2]-rect_pos[ie][0] == 35*6:
                continue
            elif rect_pos[ie][3]-rect_pos[ie][1] == 24*6:
                continue

        dri.rectangle(((rect_pos[ie][0]+randint(0,10), rect_pos[ie][1]+randint(0,10)),(rect_pos[ie][2]+randint(0,10),rect_pos[ie][3]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
        dri.text((int(rect_pos[ie][0]+5),int(rect_pos[ie][1]+(randint(0,80)))),str(rect_pos[ie][4]),(color[0], color[1], color[2]),font=font)

def compareROIs(imageXmlPath, dr, rect_pos):
    sizex = par_config.sizex
    sizey = par_config.sizey
    classes = par_config.classes
    xmlRoot = xml.etree.ElementTree.parse(imageXmlPath).getroot()
    imageSizeRows = int(xmlRoot.findall('imagesize')[0].findall('nrows')[0].text)
    imageSizeCols = int(xmlRoot.findall('imagesize')[0].findall('ncols')[0].text)

    #print("Image Cols: " + imageSizeCols)
    #print("Image Rows: " + imageSizeRows)

    objects = xmlRoot.findall('object')
    count = -1


    for annoObject in objects:
        count += 1
        name = annoObject.findall('name')[0].text.replace("_", " ")
        isDeleted = annoObject.findall('deleted')[0].text
        polygon = annoObject.findall('polygon')[0]
        points = polygon.findall('pt')
        if name in par_config._CLASSES_TO_IGNORE_ or isDeleted == "1":
            log.info("Ignored Class")
            continue
        ptsLst = []
        for x in points:
            ix = int(x.findall('x')[0].text)
            iy = int(x.findall('y')[0].text)
            ix *= (sizex/imageSizeCols)
            iy *= (sizey/imageSizeRows)
            ptsLst.extend([ix,iy])
        if len(ptsLst) > 8:
            log.info("More than 4")
            continue

        #print(ptsLst)

        for predObj in rect_pos:

            if predObj[4] == classes[name.replace(" ", "_")]:
                log.info("---------------->>" + name + ": ")
                log.info(ptsLst)
                jaccard = utils.set_operations.intersection_over_union(predObj[0:4], [ptsLst[0],ptsLst[1],ptsLst[4],ptsLst[5]])
                log.info(")))))))->>" + str(jaccard))

        dr.polygon(ptsLst, fill=(0, 0, 0, 50), outline = (255, 255, 255))


def AP(imageXmlPath, rect_pos):
    sizex = par_config.sizex
    sizey = par_config.sizey
    classes = par_config.classes
    xmlRoot = xml.etree.ElementTree.parse(imageXmlPath).getroot()
    imageSizeRows = int(xmlRoot.findall('imagesize')[0].findall('nrows')[0].text)
    imageSizeCols = int(xmlRoot.findall('imagesize')[0].findall('ncols')[0].text)

    #print("Image Cols: " + imageSizeCols)
    #print("Image Rows: " + imageSizeRows)

    objects = xmlRoot.findall('object')
    count = -1

    FP = 0
    TP = 0
    FN = 0
    detectedObjects = []
    for annoObject in objects:
        count += 1
        name = annoObject.findall('name')[0].text.replace("_", " ")
        isDeleted = annoObject.findall('deleted')[0].text
        polygon = annoObject.findall('polygon')[0]
        points = polygon.findall('pt')
        if name in par_config._CLASSES_TO_IGNORE_ or isDeleted == "1":
            log.info("Ignored Class")
            continue
        ptsLst = []
        for x in points:
            ix = int(x.findall('x')[0].text)
            iy = int(x.findall('y')[0].text)
            ix *= (sizex/imageSizeCols)
            iy *= (sizey/imageSizeRows)
            ptsLst.extend([ix,iy])
        if len(ptsLst) > 8:
            log.info("More than 4")
            continue

        #print(ptsLst)

        for predObj in rect_pos:

            if predObj[4] == classes[name.replace(" ", "_")]:
                log.info("---------------->>" + name + ": ")
                log.info(ptsLst)
                jaccard = utils.set_operations.intersection_over_union(predObj[0:4], [ptsLst[0],ptsLst[1],ptsLst[4],ptsLst[5]])
                log.info(")))))))->>" + str(jaccard))


def joinROIS(class_scores):
    print("RoI Count: " + str(len(class_scores)))
    from shapely.geometry import Polygon, box
    print("joining rois")
    foid = True
    coverage = par_config.joinRoIs_considered_coverage
    while foid:
        breako = False
        #if coverage <0.90:
        #    coverage += 0.01
        for ii in class_scores:
            for xx in class_scores:
                if xx == ii: continue
                #rectxx = box(xx[3],xx[2], xx[1], xx[0], True)
                #rectii = box(ii[3],ii[2], ii[1], ii[0], True)
                iou = utils.set_operations.intersection_over_union(xx,ii)

                #rectxx.intersects(rectii)
                if iou > coverage and (ii[4] == xx[4]):
                    #print(coverage)
                    ii[0] = min(ii[0],  xx[0])
                    ii[1] = min(ii[1],  xx[1])
                    ii[2] = max(ii[2],  xx[2])
                    ii[3] = max(ii[3],  xx[3])
                    class_scores.remove(xx)
                    breako = True
                    break
            if breako: break
        if breako: continue
        break
