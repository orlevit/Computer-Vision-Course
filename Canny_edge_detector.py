import cv2
from matplotlib import pyplot as plt

# Read color image
img = cv2.imread('owl.jpg',1)

# Run Canny edge detector
edges = cv2.Canny(img,100,250)

# Put the original image and the one with the edges, side by side
plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
