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
    img_copy = np.copy(img)
    blank_image = np.zeros((img_copy.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2, y2), (0, 0, 255), 4)
    
    img_copy = cv2.addWeighted(img_copy, 0.8, blank_image, 1, 0)
    return img_copy



image = cv2.imread('data/road.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)

region_of_interest_vertices = [(400, height), (width/1.6, height/2), (width, height)]

mask_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(mask_image, rho=6, theta = np.pi/60, threshold=60, lines=np.array([]), minLineLength=40, maxLineGap=25)

image_with_lines = draw_lines(image, lines)


plt.imshow(image_with_lines)
plt.show()