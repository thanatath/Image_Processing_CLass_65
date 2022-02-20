import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,UpSampling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
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

    

test_x=[]
x_val=[]
val_x=[]


#X_train, x_val = train_test_split(x,test_size=0.33,random_state=32)

train_x,test_x = train_test_split(x,test_size=0.3,random_state=32)
train_x,val_x = train_test_split(train_x,test_size=0.3,random_state=32)




#----------------------------------1.2-------------------------------------

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

#display5plot(train_x[:5])
#display5plot(x_train_noisy[:5])
 
    
#----------------------------------2-------------------------------------

def create_model(optimizer='adam'):

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
    autoencoder.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mean_squared_error'])
    #autoencoder.summary()
    
    return autoencoder


model = KerasRegressor(build_fn=create_model,epochs=2,batch_size=16,verbose=0)

#Grid search





def Grid_search(model,x_train,y_train):
    optimizers = ['adam','rmsprop','sgd','adagrad']
    batch_size = [8,16,32]
    epochs = [2,4,6]
    param_grid = dict(optimizer=optimizers,batch_size=batch_size,epochs=epochs)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        verbose=2,
                        cv=2,n_jobs=1)
    grid_result = grid.fit(x_train,y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    return grid_result


grid_result = Grid_search(model,np.array(x_train_noisy),np.array(train_x))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

def plot_result_from_all_combination_of_grid_parameter(mean_test_score,std_test_score,params):   
    plt.figure(figsize=(12,6))
    plt.errorbar(range(len(mean_test_score)),mean_test_score,yerr=std_test_score,fmt='o')
    plt.xticks(range(len(mean_test_score)),params,rotation=90)
    plt.xlabel('parameters')
    plt.ylabel('score')
    plt.figtext(0.5,0.9,'best score: %.4f'%(max(mean_test_score)),fontsize=15)
    plt.show()
 
    
plot_result_from_all_combination_of_grid_parameter(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
    

print("Best parameters:", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)



#----------------------------------3-------------------------------------

random_search = {'optimizer':['adam','rmsprop','sgd','adagrad'],
                 'batch_size':list(np.linspace(8,64,5,dtype=int)),
                 'epochs':list(np.linspace(1,10,4,dtype=int))}


def Grid_search_rand(model,x_train,y_train):
    #optimizers = ['adam','rmsprop','sgd','adagrad']
    #batch_size = [8]
    #epochs = [2]
    #param_grid = dict(optimizer=optimizers,batch_size=batch_size,epochs=epochs)
    grid_rand = RandomizedSearchCV(estimator=model,
                                verbose=2,
                                cv=2,
                                n_iter=10,
                                param_distributions=random_search,
                                n_jobs=1)
    grid_result = grid_rand.fit(x_train,y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    return grid_result



grid_result = Grid_search_rand(model,np.array(x_train_noisy),np.array(train_x))

print("Best parameters:", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

plot_result_from_all_combination_of_grid_parameter(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
    







