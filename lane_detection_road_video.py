import matplotlib.pyplot as plt 
import cv2
import numpy as np

def region_of_interest(img, vertices):
    '''
    Masking the image to the region of interest
    '''
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    '''
    Draw the lines resulted from the hugh transform
    '''
    img_copy = np.copy(img)
    blank_image = np.zeros((img_copy.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2, y2), (255, 0, 0), 4)
    
    img_copy = cv2.addWeighted(img_copy, 0.8, blank_image, 1, 0)
    return img_copy

def process(image):
    '''
    Process the image first applying a canny edge detector,
    then masking a region of interest with the custom function,
    detect lines with the hough transform probabilistic, 
    and draw the lines with our custom function
    '''
    height = image.shape[0]
    width = image.shape[1]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)

    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]

    mask_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(mask_image, rho=2, theta = np.pi/60, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100)

    image_with_lines = draw_lines(image, lines)

    return image_with_lines

cap = cv2.VideoCapture('data/test2.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
