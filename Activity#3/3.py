# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution
IMG1 = cv2.imread('./image1.jpg')
IMG2 = cv2.imread('./image2.jpg')
#IMG3 = cv2.imread('./image3.jpg')
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
    return im.astype(np.uint8) #mod


fig,plot = plt.subplots(3,2)
plot[0,0].imshow(IMG_RGB1)
plot[0,0].set_title('Image 1')
plot[0,1].set_title('Hist Original Image')
color = ('r','g','b')
for i,c in enumerate(color):
    hist = cv2.calcHist([IMG1],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[0,1].plot(hist,color = c)
    
    
    
    
 
 

plot[1,0].imshow(IMG_RGB2)
plot[1,0].set_title('Image 2')
plot[1,1].set_title('Hist Image')
color = ('r','g','b')
for i,c in enumerate(color):
    hist = cv2.calcHist([IMG2],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[1,1].plot(hist,color = c)
plot[1,1].set_title('Hist Image')
 
 
 
 
 
img_match = np.zeros(IMG_RGB1.shape,dtype=np.uint8)
for i in range(3):
    img_match[:,:,i] = hist_matching(cdf(IMG_RGB1[:,:,i]),cdf(IMG_RGB2[:,:,i]),IMG_RGB1[:,:,i])
#img_match[:,:,0] = hist_matching(cdf(IMG_RGB1[:,:,0]),cdf(IMG_RGB2[:,:,0]),IMG_RGB1[:,:,0])
#img_match[:,:,1] = hist_matching(cdf(IMG_RGB1[:,:,1]),cdf(IMG_RGB2[:,:,1]),IMG_RGB1[:,:,1])
#img_match[:,:,2] = hist_matching(cdf(IMG_RGB1[:,:,2]),cdf(IMG_RGB2[:,:,2]),IMG_RGB1[:,:,2])

plot[2,0].imshow(img_match)
plot[2,0].set_title('Hist Matching')
plot[2,1].set_title('Hist EQ Image')

for i,c in enumerate(color):
    hist = cv2.calcHist([img_match],[i],None,[256],[0,256]) # images, channels, mask, histSize, ranges
    plot[2,1].plot(hist,color = c)

 
 


plt.show()
# %%
