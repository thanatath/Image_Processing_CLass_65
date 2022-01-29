import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution
IMG = cv2.imread('img.jpg')
IMG_RGB = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

 
fig,plot = plt.subplots(2,2)
plot[0,0].imshow(IMG_RGB)
plot[0,0].set_title('Original Image')

color = ('r','g','b')
for i,c in enumerate(color):
    hist = cv2.calcHist([IMG_RGB],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[0,1].plot(hist,color = c)
 
plot[0,1].set_title('Hist Original Image')


#EQ IMAGE

def equalizehistchannel(img):
    img_eq = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2BGR)
    for i in range(3):
        img_eq[:,:,i] = cv2.equalizeHist(img_eq[:,:,i])
    return img_eq
 
img_eq = equalizehistchannel(IMG)
 
 

plot[1,0].imshow(img_eq)
plot[1,0].set_title('Equalized Image')

color = ('r','g','b')
for i,c in enumerate(color):
    hist = cv2.calcHist([img_eq],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[1,1].plot(hist,color = c)
 
plot[1,1].set_title('Hist EQ Image')



plt.show()
 
 
 




cv2.waitKey(0)