import par_config
import logging as log
from random import randint

import utils.set_operations
import xml.etree.ElementTree
import utils.colors

from PIL import ImageFont, ImageDraw

def draw_boundingboxes(rect_pos,dri,colors):
    font = ImageFont.truetype(par_config.font_filename, 18)
    for ie in range(len(rect_pos)):

        color = colors[rect_pos[ie]["class"]]
        if par_config.ignore_notJoined_boxes == True:
            if rect_pos[ie]["xmax"]-rect_pos[ie]["xmin"] == 35*6:
                continue
            elif rect_pos[ie]["ymax"]-rect_pos[ie]["ymin"] == 24*6:
                continue

        dri.rectangle(((rect_pos[ie]["xmin"]+randint(0,10), rect_pos[ie]["ymin"]+randint(0,10)),(rect_pos[ie]["xmax"]+randint(0,10),rect_pos[ie]["ymax"]+randint(0,10))), fill=(color[0], color[1], color[2], 50), outline = (color[0], color[1], color[2]))
        dri.text((int(rect_pos[ie]["xmin"]+5),int(rect_pos[ie]["ymin"]+(randint(0,80)))),str(rect_pos[ie]["class"]),(color[0], color[1], color[2]),font=font)

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

            if predObj["class"] == classes[name.replace(" ", "_")]:
                log.info("---------------->>" + name + ": ")
                log.info(ptsLst)
                gtbox = {}
                gtbox["class"] = name
                gtbox["xmin"] = ptsLst[0]
                gtbox["ymin"] = ptsLst[1]
                gtbox["xmax"] = ptsLst[4]
                gtbox["ymax"] = ptsLst[5]
                jaccard = utils.set_operations.intersection_over_union(predObj, gtbox)#predObj[0:4]   #[ptsLst[0],ptsLst[1],ptsLst[4],ptsLst[5]]
                log.info(")))))))->>" + str(jaccard))

        dr.polygon(ptsLst, fill=(0, 0, 0, 50), outline = (255, 255, 255))


def getBboxXML(imageXmlPath, rect_pos):
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

    objects = {}
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

        objects[name] = [ptsLst[0],ptsLst[1],ptsLst[4],ptsLst[5]]

    return objects



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
                if iou > coverage and (ii["class"] == xx["class"]):
                    #print(coverage)
                    ii["xmin"] = min(ii["xmin"],  xx["xmin"])
                    ii["ymin"] = min(ii["ymin"],  xx["ymin"])
                    ii["xmax"] = max(ii["xmax"],  xx["xmax"])
                    ii["ymax"] = max(ii["ymax"],  xx["ymax"])
                    class_scores.remove(xx)
                    breako = True
                    break
            if breako: break
        if breako: continue
        break
