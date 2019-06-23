
from __future__ import division
from __future__ import print_function
import utils
import cv2

from DataLoader import Batch
from Model import Model, DecoderType
from scan import DocScanner
from SamplePreprocessor import preprocess
import matplotlib.pyplot as plt

from main import FilePaths
from WordSegmentation import wordSegmentation, prepareImg
decoderType = DecoderType.BestPath
import os
import numpy as np

def preprocessImage(image):

    #Set scanner
    scanner = DocScanner("false")

    # plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    # plt.show()

    # Perform scan and get image
    scanned_img = scanner.scan(image)

    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(scanned_img, kernel, iterations=1)

    # prepare to segmentation
    return prepareImg(img, 300)
