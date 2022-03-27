import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from skimage.feature import hog
from scipy import spatial
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing.image import load_img
from skimage.measure import label
 

all_files = glob.glob('./Animals/*/*.jpg')

hog1 = cv2.GaussianBlur(cv2.resize(cv2.imread(all_files[0]), (500, 500)), (5, 5), 0)
hog2 = cv2.GaussianBlur(cv2.resize(cv2.imread(all_files[1]), (500, 500)), (5, 5), 0)
hog3 = cv2.GaussianBlur(cv2.resize(cv2.imread(all_files[2]), (500, 500)), (5, 5), 0)
hog4 = cv2.GaussianBlur(cv2.resize(cv2.imread(all_files[3]), (500, 500)), (5, 5), 0)

fd, hog_image1 = hog(hog1, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
fd, hog_image2 = hog(hog2, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
fd, hog_image3 = hog(hog3, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
fd, hog_image4 = hog(hog4, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)

plot, ax = plt.subplots(1, 4, figsize=(12, 4), sharey=True, sharex=True)
ax[0].imshow(hog_image1, cmap=plt.cm.gray)
ax[1].imshow(hog_image2, cmap=plt.cm.gray)
ax[2].imshow(hog_image3, cmap=plt.cm.gray)
ax[3].imshow(hog_image4, cmap=plt.cm.gray)
plt.show()

#################################### 9.1.2 ####################################################


