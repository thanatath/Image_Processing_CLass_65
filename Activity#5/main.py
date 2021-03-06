import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,UpSampling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import glob
import tensorflow as tf

filenames = glob.glob("./face_mini/*/*.jpg")

x,y = [],[]
#load IMG & Resize
for i in range(len(filenames)): #len(filenames)
    img = image.load_img(filenames[i],target_size=(90,90),color_mode='rgb',interpolation='nearest')
    img = img_to_array(img)
    img = img/255
    x.append(img)
    y.append(i)

    

test_x=[]
x_val=[]
val_x=[]


#X_train, x_val = train_test_split(x,test_size=0.33,random_state=32)

train_x,test_x = train_test_split(x,test_size=0.33,random_state=32)
train_x,val_x = train_test_split(train_x,test_size=0.33,random_state=32)




#----------------------------------2-------------------------------------

#noise factor
noise_factor = 0.1

#noise parameter
noise_distribution = 'normal'
noise_mean = 0
noise_std = 1

x_train_noisy = []
x_val_noisy = []

x_train_noisy = train_x + (noise_factor * np.random.normal(noise_mean,noise_std,(len(train_x),90,90,3)))
x_val_noisy = val_x + (noise_factor * np.random.normal(noise_mean,noise_std,(len(val_x),90,90,3)))
x_test_noisy = test_x + (noise_factor * np.random.normal(noise_mean,noise_std,(len(test_x),90,90,3)))


def display5plot(imgs):
    fig,axes = plt.subplots(1,len(imgs),figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(imgs,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
display5plot(x_train_noisy[:5])
display5plot(x_val_noisy[:5])
display5plot(x_test_noisy[:5])
 
    
#----------------------------------3-------------------------------------


input_img = Input(shape=(90,90,3))

#encoder
x1 = Conv2D(256,(3,3),activation='relu',padding='same')(input_img)
x2 = Conv2D(128,(3,3),activation='relu',padding='same')(x1)
x2 = MaxPooling2D((2,2))(x2)
encoded = Conv2D(64,(3,3),activation='relu',padding='same')(x2)

#decoder
x3 = Conv2D(64,(3,3),activation='relu',padding='same')(encoded)
x3 = UpSampling2D((2,2))(x3)
x4 = Conv2D(128,(3,3),activation='relu',padding='same')(x3)
x5 = Conv2D(128,(3,3),activation='relu',padding='same')(x4)
decoded = Conv2D(3,(3,3),padding='same')(x5)

#optimizer
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.summary()

#training
x_train_noisy = data_list = tf.stack(x_train_noisy)
train_x = data_list = tf.stack(train_x)
x_val_noisy = data_list = tf.stack(x_val_noisy)
val_x = data_list = tf.stack(val_x)

callback = tf.keras.callbacks.EarlyStopping()

history = autoencoder.fit(np.array(x_train_noisy),np.array(train_x),epochs=2,batch_size=8,
                          validation_data=(np.array(x_val_noisy),np.array(val_x)),shuffle=True,callbacks=[callback])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

prediction = autoencoder.predict(x_val_noisy)
prediction2 = autoencoder.predict(x_test_noisy)

def display5plot(imgs):
    fig,axes = plt.subplots(1,len(imgs),figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(imgs,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
display5plot(x_val_noisy[:5])
display5plot(prediction[:5])
display5plot(x_test_noisy[:5])
display5plot(prediction2[:5])














