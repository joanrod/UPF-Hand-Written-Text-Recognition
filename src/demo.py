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


decoderType = DecoderType.WordBeamSearch

# load training data, create TF model
loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

# save characters of model for inference mode
open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

# save words contained in dataset into file
open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

# execute training or validation
tf.reset_default_graph()
model = Model(loader.charList, decoderType)

#Image 1 (diamonds)
image = cv2.imread('/Users/joanrodriguezgarcia/PycharmProjects/DL/data/words/p03/p03-158/p03-158-02-00.png', cv2.IMREAD_GRAYSCALE)


wordImg = preprocess(image, Model.imgSize)
batch = Batch(None, [wordImg])
(recognized, probability) = model.inferBatch(batch, True)

print('Recognized:', '"' + recognized[0] + '"')
print('Probability:', probability[0])

#Image 2 (tomorrow)
image = cv2.imread('/Users/joanrodriguezgarcia/PycharmProjects/DL/data/words/a01/a01-000u/a01-000u-03-01.png', cv2.IMREAD_GRAYSCALE)


wordImg = preprocess(image, Model.imgSize)
batch = Batch(None, [wordImg])
(recognized, probability) = model.inferBatch(batch, True)

print('Recognized:', '"' + recognized[0] + '"')
print('Probability:', probability[0])

#Image 3 (balance)
image = cv2.imread('/Users/joanrodriguezgarcia/PycharmProjects/DL/data/words/a01/a01-058x/a01-058x-02-02.png', cv2.IMREAD_GRAYSCALE)


wordImg = preprocess(image, Model.imgSize)
batch = Batch(None, [wordImg])
(recognized, probability) = model.inferBatch(batch, True)

print('Recognized:', '"' + recognized[0] + '"')
print('Probability:', probability[0])
