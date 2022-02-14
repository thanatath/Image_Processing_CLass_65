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
for i in range(300): #len(filenames)
    img = image.load_img(filenames[i],target_size=(90,90),color_mode='rgb',interpolation='nearest')
    img = img_to_array(img)
    img = img/255
    x.append(img)
    y.append(i)

    

X_train=[]
x_val=[]



X_train, x_val = train_test_split(x,test_size=0.3,random_state=32)




#----------------------------------2-------------------------------------

#noise factor
noise_factor = 0.5

#noise parameter
noise_distribution = 'normal'
noise_mean = 0
noise_std = 1

def noisy(train,noise_factor,noise_distribution,noise_mean,noise_std):
    train_noisy = train + (noise_factor * np.random.normal(noise_mean,noise_std,[len(train),90,90,3]))
    return train_noisy

X_train_noisy = noisy(X_train,noise_factor,noise_distribution,noise_mean,noise_std)
x_val_noisy = noisy(x_val,noise_factor,noise_distribution,noise_mean,noise_std)


def display5plot(imgs):
    fig,axes = plt.subplots(1,len(imgs),figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(imgs,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
display5plot(X_train_noisy[:5])
display5plot(X_train[:5])
 
    
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
x2 = Conv2D(128,(3,3),activation='relu',padding='same')(x3)
x1 = Conv2D(128,(3,3),activation='relu',padding='same')(x2)
decoded = Conv2D(3,(3,3),padding='same')(x1)

#optimizer
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.summary()

#training
X_train_noisy = data_list = tf.stack(X_train_noisy)
X_train = data_list = tf.stack(X_train)
x_val_noisy = data_list = tf.stack(x_val_noisy)
x_val = data_list = tf.stack(x_val)

callback = tf.keras.callbacks.EarlyStopping()
history = autoencoder.fit(X_train_noisy,X_train,epochs=2,batch_size=8,
                          validation_data=(x_val_noisy,x_val),shuffle=True,callbacks=[callback])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

prediction = autoencoder.predict(x_val_noisy)

def display5plot(imgs):
    fig,axes = plt.subplots(1,len(imgs),figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(imgs,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
display5plot(X_train[:5])
display5plot(X_train_noisy[:5])
display5plot(prediction[:5])














