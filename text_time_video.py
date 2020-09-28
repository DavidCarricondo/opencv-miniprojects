import cv2
import datetime


cap = cv2.VideoCapture(0)

cap.set(3, 1280) #3 is the shorthand for CAP_PROP_FRAME_WIDTH
cap.set(4, 720) #4 is the shorthand for CAP_PROP_FRAME_HEIGHT


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        text = 'Width: '+ str(cap.get(3)) + ' Height: ' + str(cap.get(4))
        
        datet = str(datetime.datetime.now())
        
        frame = cv2.putText(frame, text, (10, 700), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, datet, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()