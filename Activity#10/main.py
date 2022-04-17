import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import pandas as pd
from scipy import spatial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
 
#load the model

base_model = MobileNet(weights='imagenet', include_top=False)

#add_layer

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

#Assign transfer learning model to new model

model = Model(inputs=base_model.input, outputs=preds)
model.summary() #Before freezing

#for layer in model.layers[:5]:
#    layer.trainable = True
#for layer in model.layers[8:10]:
#    layer.trainable = False
#for layer in model.layers[20:-1]:
#    layer.trainable = True

model.summary() #After freezing

############################################10.2############################################

seed_value = 42

# Image generator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.5,shear_range=0.15,fill_mode='nearest')

#Generate training and validation data

train_generator = datagen.flow_from_directory( './Cat_Dog_Horse/train/', target_size=(224,224), batch_size=32, class_mode='categorical',color_mode='rgb' , seed=seed_value,shuffle=True)

val_generator = datagen.flow_from_directory( './Cat_Dog_Horse/validate/', target_size=(224,224), batch_size=16, class_mode='categorical',color_mode='rgb' , seed=seed_value,shuffle=True)

def preview_image(datagen):
    x,y = datagen.next()
    plt.figure(figsize=(10,10))
    for j in range(0,8):
        plt.subplot(4,4,j+1)
        plt.imshow(x[j])
        plt.title(str(datagen.directory).split('/')[2])
        plt.axis('off')
    plt.show()

        
preview_image(train_generator) #Preview training images
preview_image(val_generator) #Preview validation images


############################################10.2.2############################################

#create optimizer

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#training parameters

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size

print(step_size_train)
print(step_size_val)


############################################10.2.3############################################

#train the model

EP = 20
callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
history = model.fit_generator(train_generator, steps_per_epoch=step_size_train, epochs=EP, validation_data=val_generator, validation_steps=step_size_val)

def performanc_plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def performanc_plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
performanc_plot_acc(history)
performanc_plot_loss(history)


############################################10.3.1############################################

#create TEST data generator and predict

test_data_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_data_generator.flow_from_directory( './Cat_Dog_Horse/test/', target_size=(224,224), batch_size=1, class_mode='categorical',color_mode='rgb',shuffle=False)

#class id
y_true = test_generator.classes


#predict

test_generator.reset()
pred_prob = []
for i in range(len(y_true)):
    pred = model.predict(test_generator[i][0])
    pred_prob.append(np.array(pred))
    
#prediction result
pred_prob = np.array(pred_prob).reshape(len(y_true),3)
df_pred = pd.DataFrame(pred_prob)
df_class = df_pred.idxmax(axis=1)


y_real = y_true
y_pred = model.predict_generator(test_generator, steps=len(test_generator)).argmax(axis=-1)
target_name = test_generator.class_indices
print(confusion_matrix(y_real, y_pred),'\n')
print('REAL :\t \t',y_real)
print('PREDICT :\t',y_pred)
print(classification_report(y_real, y_pred, target_names=target_name))
