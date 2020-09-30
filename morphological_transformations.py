import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/smarties.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3,3), np.uint8)

dilation = cv2.dilate(mask, kernel,  iterations=3)
erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #first erosion then dilate
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #first dilate, then erosion
gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'tophat']
images = [img, mask, dilation, erosion, opening, closing, gradient, tophat]


for i in range(len(images)):
    plt.subplot(2, np.ceil(len(images)/2), i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()