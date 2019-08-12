import cv2
import numpy as np

# Uploading the image and convert it to gray scale
img = cv2.imread('owl.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Apply SIFT to the picture
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

# Draw the key pints on the image and save it as a new one
img=cv2.drawKeypoints(gray,kp,outImage=None, color=(0, 0, 255))
cv2.imwrite('owl_with_keypints.jpg',img)