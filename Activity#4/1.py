import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy  import signal

IMG = cv2.imread('img.jpg')
IMG = cv2.resize(IMG, (224, 224))

IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)


#load model
model = VGG16()
#model Detail
model.summary()

#retrive kernel weight of convolution layer
kernel , bias = model.layers[1].get_weights()
#view cnn layer architecture
model.layers[1].get_config()

#preprocess image
#convert image to array

img_array = img_to_array(IMG)

# expand dimentions to fit model
#Reshape to 4D array
img = expand_dims(img_array, axis=0)
#prepare the image for the VGG model
img_ready = preprocess_input(img)


#Extract model CNN Layer 1
model = Model(inputs=model.input, outputs=model.layers[1].output)
model.summary()

#CNN Layer 1 ->n_filter = 64
feature_map = model.predict(img_ready)

def display64filter(feature_map):
    #display 64 filter
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(feature_map[0, :, :, i],cmap='jet')
        plt.axis('off')
    plt.show()
    
display64filter(feature_map)