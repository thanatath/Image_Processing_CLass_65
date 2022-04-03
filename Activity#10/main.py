import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from scipy import spatial
 
#load the model

base_model = MobileNet(weights='imagenet', include_top=False)

#add_layer

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

#Assign transfer learning model to new model

model = Model(inputs=base_model.input, outputs=preds)
model.summary() #Before freezing

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

model.summary() #After freezing

