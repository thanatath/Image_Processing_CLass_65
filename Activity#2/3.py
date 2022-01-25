import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg
from matplotlib import pyplot as plt

IMG = cv2.imread('./img.jpg') #import Image

ZEROARRAY = np.zeros(IMG.shape, dtype=np.uint8)
 
def fill_255(x,y,w,h):
    ZEROARRAY[y:y+h,x:x+w] = 255
    
fill_255(250,50,200,120) # X Y W H


fig, plot = plt.subplots(1,3)
fig.suptitle('Masked Image')

im_rgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB) # convert to RGB
plot[0].imshow(im_rgb)
plot[0].set_title('ORIGINAL') 
plot[1].imshow(ZEROARRAY) 
plot[1].set_title('Image Mask')
plot[2].imshow(cv2.bitwise_and(ZEROARRAY, im_rgb)) 
plot[2].set_title('Bitwise_and Result')

 
plt.show() # display plot

