import numpy as np
import cv2
import matplotlib.pyplot as plt

3 # import the two images of the same object(in this case: Ecerest mountain)
img1 = cv2.imread('Fun-Kids-Science-Facts-on-Mount-Everest-Image-of-the-Mount-Everest.jpg',0)      
img2 = cv2.imread('529814623.jpg',0)

# Initiate the ORB detector and find keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create a brute force matcher and Match descriptors.of the two images obtained by ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show() 