import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('data/messi5.jpg', 0)

canny = cv2.Canny(img, 100, 200)


titles = ['image', 'canny']
images = [img, canny]

for i in range(len(images)):
    plt.subplot(2, np.ceil(len(images)/2), i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()