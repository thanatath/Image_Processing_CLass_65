import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG = cv2.imread('logo.jpg')

fig, plot = plt.subplots(2,4)
fig.suptitle('BGR VS RGB')



plot[0,0].imshow(IMG) # BGR
plot[0,0].set_title('BGR')
b, g, r = cv2.split(IMG)
plot[0,1].imshow(r,cmap='gray') 
plot[0,1].set_title('R')
plot[0,2].imshow(g,cmap='gray') 
plot[0,2].set_title('G')
plot[0,3].imshow(b,cmap='gray') 
plot[0,3].set_title('B')

im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB) # convert to RGB

plot[1,0].imshow(im_rgb)
plot[1,0].set_title('RGB') 
b, g, r = cv2.split(im_rgb) 
plot[1,1].imshow(r,cmap='gray') 
plot[1,1].set_title('R')
plot[1,2].imshow(g,cmap='gray') 
plot[1,2].set_title('G')
plot[1,3].imshow(b,cmap='gray') 
plot[1,3].set_title('B')


plt.show() # display plot



fig, plot2 = plt2.subplots(2)

plt.show()