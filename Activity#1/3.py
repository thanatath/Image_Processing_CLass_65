import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reduceimgagebitdepth(img,bitdepth):
    print("RAW:\n",img)
    print("---------")
    img = img.astype(np.float32)
    img = (img/255)*bitdepth
    img = img.astype(np.uint8)
    print("Reduced:\n",img)
    return img
        

IMG = cv2.imread('logo.jpg',cv2.IMREAD_GRAYSCALE)
#IMG_TEST = np.array(((154,232,13,42,53),(442,523,46,227,38),(27,81,95,60,311)))


fig, plot = plt.subplots(1,2)
fig.suptitle('Reduce Bits')
 
plot[0].imshow(IMG,cmap='gray') #RAW
plot[0].set_title('RAW') #RAW
bitdepth = 16
plot[1].imshow(reduceimgagebitdepth(IMG,bitdepth),cmap='gray') #Reduce
plot[1].set_title('Reduce to '+str(bitdepth) + 'Bit') #Reduce
plt.show()

 