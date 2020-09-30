import cv2
import numpy as np

img = cv2.imread('data/lena.jpg')

#lr1 = cv2.pyrDown(img)
#lr2 = cv2.pyrDown(lr1)

#hr1 = cv2.pyrUp(lr2)
#lr2 = cv2.pyrDown(lr1)

#cv2.imshow('Original image', img)
#cv2.imshow('pyrdown 1', lr1)
#cv2.imshow('pyrdown 2', lr2)
#cv2.imshow('pyrup 1', hr1)


cv2.waitKey(0)
cv2.destroyAllWindows()

layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    cv2.imshow(str(i), layer)

cv2.imshow('img', img)


cv2.waitKey(0)
cv2.destroyAllWindows()