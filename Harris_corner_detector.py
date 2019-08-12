import cv2
import numpy as np

# Uploading the image and convert it to gray scale
img = cv2.imread('owl.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Run harris corners detection
dst = cv2.cornerHarris(gray,2,5,0.002)

# Thresholding for, and mark the corner in blue(because the picture is gray-greenish)
img[dst>0.01*dst.max()]=[255,0,0]

# Show the image
cv2.imshow('Image_with_corners',img)
if cv2.waitKey(0)  >30:
    cv2.destroyAllWindows()