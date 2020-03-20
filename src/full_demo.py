#Test the Neural Network

from __future__ import division
from __future__ import print_function
#Test the Neural Network

from __future__ import division
from __future__ import print_function
import utils
import cv2

from DataLoader import Batch
from Model import Model, DecoderType
from DataLoader import DataLoader
from scan import DocScanner
from SamplePreprocessor import preprocess
import matplotlib.pyplot as plt
import numpy as np
from main import FilePaths
import tensorflow as tf
from Model import Model
from WordSegmentation import wordSegmentation, prepareImg
from main import train, validate
import os
from DataLoader import DataLoader
from Model import DecoderType
from Model import Model
from WordSegmentation import startWordSegmentation
from main import FilePaths
#
decoderType = DecoderType.WordBeamSearch
#
# # load training data, create TF model
loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

# save characters of model for inference mode
open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

# save words contained in dataset into file
open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))



# execute training or validation
tf.reset_default_graph()

#Load the model
model = Model(loader.charList, decoderType)

#Read the image that has all the text
pathScannedSentences = '../data/sentences/test_scanned.png'
outputWords = "../data/sentences/tempScanedPictures"

#Start the segmentation and store the words in the temp folder
startWordSegmentation(pathScannedSentences)

#Run all the files in folder and infer
for im_path in os.listdir(outputWords):
    im = cv2.imread(outputWords + im_path, cv2.IMREAD_GRAYSCALE)
    wordImg = preprocess(im, Model.imgSize)


    batch = Batch(None, [wordImg])
    (recognized, probability) = model.inferBatch(batch, True)

    print(recognized[0], ' ')
