#import lib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import cumulative_distribution

IMG = cv2.imread('img.jpg')

def addjustgamma(img,gamma):
    
    temp=img.copy()
    y = 1.0 / gamma
    a=1.0
    b=0.0
    temp = ((a*((img/255)**y)+b)*255).astype(np.uint8)
    return temp

img_Gramma = addjustgamma(IMG,2)
 

def writeVIdeo(img,name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(name,fourcc, 20.0, (img.shape[1],img.shape[0])) #initiation of video writer
    
    gramma_range = np.arange(0.1,5,0.1) # define the range of the gramma frames
    
    #loop for Gramma Frames
    for i in gramma_range:
        img_Gramma = addjustgamma(IMG,i)
        out.write(img_Gramma)
        
    #Release and close the video writer
    out.release()


writeVIdeo(img_Gramma,'./test.mp4')


cap = cv2.VideoCapture('./test.mp4')
if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


