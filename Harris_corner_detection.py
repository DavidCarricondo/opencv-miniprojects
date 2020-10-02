import cv2
import numpy as np

img = cv2.imread('data/chessboard_img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray) #corerharris takes float numbers
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

#To get better results we dilate the corner result
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [255, 0, 0]

cv2.imshow('dst', img)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()