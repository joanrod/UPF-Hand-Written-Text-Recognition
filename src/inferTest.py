#Test the Neural Network

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
from PreprocessImage import preprocessImage

#Set scanner
scanner = DocScanner("false")

#Load the current model (trained)
model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

# image path and valid extensions
path = "/Users/joanrodriguezgarcia/PycharmProjects/GaussSmartPen/data/words/sub/sub-sub"

image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#Write to file
file = open(path + "/output_text.txt", "w+")
file_im = open(path + "/output_images.txt", "w+")
image_id = 0

for file in os.listdir(path):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(path, file))
id = 0
# loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath)

    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:


            wordImg = preprocess(image, Model.imgSize)
            batch = Batch(None, [wordImg])
            (recognized, probability) = model.inferBatch(batch, True)
            print('Recognized:', '"' + recognized[0] + '"')
            #print('Probability:', probability[0])

            id += 1

    elif image is None:
        print("Error loading: " + imagePath)
        # end this loop iteration and move on to next image
        continue

#file.close()
#file_im.close()
#close any open windows
#cv2.destroyAllWindows()


