import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('data/pic6.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#matplotlib reads in rgb

kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5))
gaussian = cv2.GaussianBlur(img, (5,5), 0)
median = cv2.medianBlur(img, 5)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

titles = ['image', '2d-convolution', 'blur', 'gaussian', 'median', 'bilateral_filter']
images = [img, dst, blur, gaussian, median, bilateralFilter]

for i in range(len(images)):
    plt.subplot(2, np.ceil(len(images)/2), i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()