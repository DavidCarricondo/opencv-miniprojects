import numpy as np
import cv2

cap = cv2.VideoCapture('data/vtest.avi')
#for the following method opencv-contrib-python has to be installed with pip3
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#the GMG method
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = fgbg.apply(frame)
    #for the GMG method:
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()