import cv2
import numpy as np

img = cv2.imread('data/lena.jpg')

layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)

#A level in Laplacian PYramid is formed by the \
# difference between that level in gaussian pyramid\
#  and expanded version of its upper level in Gaussian pyramid

layer = gp[5]
cv2.imshow('upper level Gaussian PYramid', layer)
lp=[layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gp[i])
    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow('img', img)


cv2.waitKey(0)
cv2.destroyAllWindows()