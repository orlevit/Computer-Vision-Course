import cv2
import numpy as np

# Uploading the image and convert it to gray scale
img = cv2.imread('lines.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# use canny edge detection to find interest points first and show them
edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)

# Run the hough transform(probablistic is used for time consideration and ease) and point the lines in blue
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength = 25 ,maxLineGap = 10)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

# Present the image with the line that the transform found
cv2.imshow('hough',img)
cv2.waitKey(0)

