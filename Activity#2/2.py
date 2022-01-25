import cv2 
import numpy as np
import math
import matplotlib.pyplot as mpimg

IMG = cv2.imread('./img.jpg') #import Image
IMG2 = cv2.imread('./img2.jpg') #import Image


#Write Video

def loopWriteVideo(IMG,IMG2):
    sa=1.0
    sb=0.01
    # Define the codec and create VideoWriter object
    writer = cv2.VideoWriter('Addition.avi', cv2.VideoWriter_fourcc(*'MP4V'), 10, (600,400))
    for i in range(0,100):
            image=cv2.addWeighted(IMG, sa, IMG2, sb, 0)
            writer.write(image)
            sa-=0.01
            sb+=0.01
    sa=0.01
    sb=1.0
    for i in range(0,100):
            image=cv2.addWeighted(IMG, sa, IMG2, sb, 0)
            writer.write(image)
            sa+=0.01
            sb-=0.01
    writer.release()


loopWriteVideo(IMG,IMG2)


#READ VIDEO

cap = cv2.VideoCapture('./Addition.avi')
while True:
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Playback',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# When everything done, release the capture
cap.release()
cv2.waitKey(0)

