import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.activations as act
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal
import os


IMG_RAW = cv2.imread('img.jpg')
IMG = cv2.resize(IMG_RAW, (224, 224),3)
IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

img_mean = np.array([123.68, 116.779, 103.939])  # RGB

# convert image to array
img_array = img_to_array(IMG)

#make 4 D array for VGG16 but it's not necessary 
img = expand_dims(img_array, axis=0)
img_index = cv2.resize(img[0], (224, 224),3)



#subtract mean
img_index -= img_mean

#RGB to BGR
img_index = cv2.cvtColor(img_index, cv2.COLOR_RGB2BGR)

plt.imshow(img_index)
plt.show()
print('img_index SHAPE : ',img_index.shape)

#---------------------------------------3------------------------------------------
#define model
model = VGG16()
kernel , bias = model.layers[1].get_weights()

print('kernel SHAPE : ',kernel.shape)

condvlist = expand_dims(img_array, axis=0)

def image_convolution_with_kernel(img_index, kernel):
    tmp_img_result = np.zeros((64,224,224,3)) #dummpy array
    tmp = np.zeros((224,224,3)) #dummpy array
    for i in range(64):
        tmp[:,:,0] = signal.convolve2d(img_index[:, :, 0], kernel[:, :,0, i], mode='same',boundary='fill', fillvalue=0)
        tmp_img_result[i,:,:,0] = tmp[:,:,0]
    for i in range(64):
        tmp[:,:,1] = signal.convolve2d(img_index[:, :, 1], kernel[:, :,1, i], mode='same',boundary='fill', fillvalue=0)
        tmp_img_result[i,:,:,1] = tmp[:,:,1]
    for i in range(64):
        tmp[:,:,2] = signal.convolve2d(img_index[:, :, 2], kernel[:, :,2, i], mode='same',boundary='fill', fillvalue=0)
        tmp_img_result[i,:,:,2] = tmp[:,:,2]
    return tmp_img_result

img_result = np.zeros((224,224,3)) 
img_result = image_convolution_with_kernel(img_index, kernel)


img_sum = np.zeros((224,224,3))
img_sum = img_result[:,:,:,0]+img_result[:,:,:,1]+img_result[:,:,:,2]

print('img_result SHAPE : ',img_sum.shape)


img_sum = act.relu(img_sum) #activation function


#---------------------------------------PLOT------------------------------------------

def display64filter(feature_map):
    #display 64 filter
    print('feature_map SHAPE : ',feature_map.shape)
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(feature_map[i,:, :],cmap='jet')
        plt.axis('on')
    plt.show()
    
display64filter(img_sum)



 
 

 



 

 