import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG = cv2.imread('./img.jpg') #import Image

fig, plot = plt.subplots(4,4)
fig.suptitle('BGR VS RGB')


im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB) # convert to RGB
plot[0,0].imshow(im_rgb)
plot[0,0].set_title('RGB') 
r, g, b = cv2.split(im_rgb) 
plot[0,1].imshow(r, cmap='gray')
plot[0,1].set_title('R')
plot[0,2].imshow(g,cmap='gray')
plot[0,2].set_title('G')
plot[0,3].imshow(b,cmap='gray') 
plot[0,3].set_title('B')

im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV) # convert to HSV
plot[1,0].imshow(im_rgb)
plot[1,0].set_title('HSV') 
h,s,v = cv2.split(im_rgb)
plot[1,1].imshow(h,cmap='gray') 
plot[1,1].set_title('H')
plot[1,2].imshow(s,cmap='gray') 
plot[1,2].set_title('S')
plot[1,3].imshow(v,cmap='gray') 
plot[1,3].set_title('V')

im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2HLS) # convert to HLS
plot[2,0].imshow(im_rgb)
plot[2,0].set_title('HLS') 
h,s,l = cv2.split(im_rgb)
plot[2,1].imshow(h,cmap='gray') 
plot[2,1].set_title('H')
plot[2,2].imshow(s,cmap='gray') 
plot[2,2].set_title('L')
plot[2,3].imshow(l,cmap='gray') 
plot[2,3].set_title('S')

im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2YCrCb) # convert to YCrCb    
plot[3,0].imshow(im_rgb)
plot[3,0].set_title('YCrCb') 
y,cr,cb = cv2.split(im_rgb)
plot[3,1].imshow(y,cmap='gray') 
plot[3,1].set_title('Y')
plot[3,2].imshow(cr,cmap='gray') 
plot[3,2].set_title('Cr')
plot[3,3].imshow(cb,cmap='gray') 
plot[3,3].set_title('Cb')


plt.show() # display plot