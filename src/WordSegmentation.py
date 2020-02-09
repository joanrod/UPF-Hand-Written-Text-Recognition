from operator import itemgetter

import math
import cv2
import numpy as np
import os


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0, maxArea = 120000):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        height,width = img.shape[:2]
        ratio = height / width

        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            #and ratio < 0.4:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry: entry[0][0])


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create an isotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def startWordSegmentation(image_path):

    print(image_path);
    #Create output temp directory
    out_dir = "../data/sentences/tempScanedPictures/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print("Directory " , out_dir ,  " Created ")
    else:
        print("Directory " , out_dir ,  " already exists")
    print( os.listdir(out_dir))

    #cleanFolder(out_dir)

    #Create input image directory
    image = cv2.imread(image_path)

    dictCoords ={}
    dictImages = {}
    matrix = []
    #Iterate to find all the words
    id = 0
    if image is not None:
        print("Image loaded")

        #Single channel image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #segmentation
        res = wordSegmentation(image, kernelSize=55, sigma=150, theta=11, minArea=5000)

        #run all segments
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            dictCoords[id] = wordBox
            dictImages[id] = wordImg
            [x,y,w,h] = wordBox
            matrix.append([x,y,w,h,id])

            id += 1

        # Set order of the words: Compute depending on bounding box coordinates
        dictOrder = orderBBoxes(matrix)
        print('hello')
        for id in dictOrder.keys():
            saveWord(out_dir, dictImages.get(id), dictOrder.get(id))



    elif image is None:
        print("Error loading image")
        # end this loop iteration and move on to next image
    print(matrix)

def orderBBoxes(matrix):
    #Order y in ascending
    matrix = sorted(matrix, key=itemgetter(1))
    tempMatrix = []


    initBox = matrix[0]
    currentOrder = 0
    currentY1 = 0
    currentY2 = (initBox[1] + initBox[3]) * 1.3
    dictOrd = {}
    maxX = max(matrix, key=itemgetter(0))
    maxX = maxX[0]
    maxY = max(matrix, key=itemgetter(1))
    maxY = maxY[1]


    l2 = len(matrix)
    while len(dictOrd) != len(matrix):
        l1 = len(dictOrd)
        for row in matrix:
            if dictOrd.get(row[4]) == None:
                x = row[0]
                y = row[1]
                #if the box is in the range
                if row[0] <= maxX and row[1]>=currentY1 and row[1]<=currentY2:
                    tempMatrix.append(row)
                elif currentY2 != maxY:
                    tempMatrix = sorted(tempMatrix, key=itemgetter(0))
                    currentY1 = row[1]
                    currentY2 = row[1] + row[3]
                    for row in tempMatrix:
                        dictOrd[row[4]] = currentOrder
                        currentOrder += 1
                    tempMatrix = []
                    break
    return dictOrd




def saveWord(out_dir, wordImg, id):

    name = '/r07-000-00-%02d.png' % id
    cv2.imwrite(out_dir + name, wordImg)  # save word

def cleanFolder(mydir):

    filelist = [f for f in os.listdir(mydir)]
    for f in filelist:
        os.remove(os.path.join(mydir, f))
        print('folder is clean')

def toCoordinates(bbox):
    rect = np.zeros((4, 2), dtype="float32")

    rect[0] = [bbox[0], bbox[1]]
    rect[1] = [bbox[0] + bbox[2], bbox[1]]
    rect[2] = [bbox[0], bbox[1] + bbox[3]]
    rect[3] = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

    return rect