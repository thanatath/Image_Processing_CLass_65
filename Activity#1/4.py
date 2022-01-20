import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG = cv2.imread('logo.jpg')
im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY) # convert to RGB


fig, plot = plt.subplots(1,2)
fig.suptitle('RESIZE')

plot[0].imshow(im_rgb,cmap='gray') #RAW
plot[0].set_title('RAW') #RAW
RESIZE_SCALE = (100,100)
Resized = cv2.resize(im_rgb,RESIZE_SCALE, interpolation = cv2.INTER_AREA)
plot[1].imshow(Resized,cmap='gray') #Resize
plot[1].set_title('Resize') #Resize


#3D Surface

gridx,gridy = np.mgrid[0:Resized.shape[0],0:Resized.shape[1]]
fig = plt.figure()
plot3d = fig.add_subplot(111, projection='3d')
plot3d.plot_surface(gridx, gridy, Resized, cmap='gray')
plot3d.set_xlabel('3D SurFace')
plt.show()