import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution
IMG1 = cv2.imread('image1.jpg')
IMG2 = cv2.imread('image2.jpg')
IMG3 = cv2.imread('image3.jpg')
IMG_RGB1 = cv2.cvtColor(IMG1, cv2.COLOR_BGR2RGB)
IMG_RGB2 = cv2.cvtColor(IMG2, cv2.COLOR_BGR2RGB)
def cdf(im):
    c,b = cumulative_distribution(im)
    for i in range(b[0]):
        c = np.insert(c,0,0)
    for i in range(b[-1]+1,256):
        c = np.append(c,1)
    return c
    
def hist_matching(c,c_t,im):
    b = np.interp(c,c_t,np.arange(256))
    pix_repl = {i:b[i] for i in range(256)}
    mp = np.arange(0,256)
    for (k,v) in pix_repl.items():
        mp[k] = v
    s = im.shape
    im = np.reshape(mp[im.ravel()],im.shape)
    im = np.reshape(im,s)
    return im.astype(np.uint8) #mod for uint8


fig,plot = plt.subplots(3,2)
plot[0,0].imshow(IMG_RGB1)
plot[0,0].set_title('Image 1')
plot[0,1].set_title('Hist Original Image')
color = ('r','g','b')

for i,c in enumerate(color):
    hist = cv2.calcHist([IMG1],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[0,1].plot(hist,color = c)
    
    
    
    
def equalizeHist(img):
    imgrs = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2BGR)
    imgrs[:,:,2] = cv2.equalizeHist(imgrs[:,:,2])
    return imgrs

img_eq = equalizeHist(IMG_RGB2)

plot[1,0].imshow(img_eq)
plot[1,0].set_title('Image 2')
plot[1,1].set_title('Hist EQ Image')
color = ('r','g','b')
for i,c in enumerate(color):
    hist = cv2.calcHist([img_eq],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[1,1].plot(hist,color = c)
plot[1,1].set_title('Hist EQ Image')
 
 
img_match = hist_matching(cdf(IMG_RGB1),cdf(IMG_RGB2),IMG_RGB1)
plot[2,0].imshow(img_match)
plot[2,0].set_title('Hist Matching')
plot[2,1].set_title('Hist EQ Image')

for i,c in enumerate(color):
    hist = cv2.calcHist([img_match],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[2,1].plot(hist,color = c)

 
 


plt.show()