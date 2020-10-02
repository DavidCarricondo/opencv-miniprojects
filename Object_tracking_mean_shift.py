import numpy as np
import cv2

cap = cv2.VideoCapture('data/slow_traffic_small.mp4')

#  take first frame of the video\

ret, frame = cap.read()

#  setup initial location of window \

x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

#  set up the ROI for tracking \

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#  Setup the termination criteria, either 10 iteration or move by at least 1 point
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    #Apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    #Draw it on image
    x, y, w, h = track_window
    final_image = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 3)
    cv2.imshow('final_image', final_image)
    cv2.imshow('backprojected window', dst)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()