import par_config
import logging as log
from random import randint

import utils.set_operations
import core.roi_management
import numpy as np

def AP(imageTitles, bboxes):
    numImages = len(imageTitles)
    TP = np.zeroes(numImages)
    FP = np.zeroes(numImages)
    FN = np.zeroes(numImages)

    detectedClasses = []

    for i, bbox in enumerate(bboxes):
        core.roi_management
