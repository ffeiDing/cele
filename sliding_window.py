# USAGE
# python sliding_window.py --image images/adrian_florida.jpg 

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import itertools
from sklearn.decomposition import PCA
import numpy as np
import glob

features = []
for filename in glob.glob('/Users/dingfeifei/Desktop/cele/cele_images/training/*.jpg'):
    try:
        image = cv2.imread(filename)
    except:
        pass
    # load the image and define the window width and height
    # image = cv2.imread(args["image"])
    # image = cv2.resize(image, None, fx=1.5, fy=1.5)
    (winW, winH) = (96, 128)

    # loop over the image pyramid
    for resized in pyramid(image, scale=1.2):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=48, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            winSize = (96, 128)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (8, 8)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            # hog = cv2.HOGDescriptor()
            h = hog.compute(window)  # 11340
            # print(h.size)

            h = list(itertools.chain.from_iterable(h))
            features.append(h)
            # print(h)

            # since we do not have a classifier, we'll just draw the window
            # clone = resized.copy()
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(1)
            # time.sleep(0.025)

# print(features)
features = np.array(features)
print(features)

# Save Numpy array
np.save('features.npy', features)

pca = PCA(n_components=0.95)
pca.fit(features)
print(features.shape)
features_pca = pca.fit_transform(features)
print(features_pca.shape)
