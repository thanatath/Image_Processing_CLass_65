import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG = cv2.imread('logo.jpg')
IMG_Trans = np.transpose(IMG.shape) #transpose
IMG_Moveaxis = np.moveaxis(IMG.shape,0,-1) #Moveaxis
print("Before Transed",IMG.shape)
print("Transed",IMG_Trans)
print("Moveaxis",IMG_Moveaxis)






plt.show() # display plot


