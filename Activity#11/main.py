import numpy as np
import cv2
from matplotlib import pyplot as plt
import keras
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import *
import visualkeras
from numpy.random import *
from numpy import *

img_size = (32,32)


################################### Prepare Real Image ###############################

def list_file_in_path(img_size=img_size):
    img_path = './Art_Bangkok/WaterColor'
    import os
    file_list = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            tmp_file = cv2.imread(os.path.join(root, file))
            tmp_file = cv2.resize(tmp_file , img_size, interpolation=cv2.INTER_LINEAR)
            tmp_file = cv2.cvtColor(tmp_file, cv2.COLOR_BGR2RGB)
            file_list.append(tmp_file)
    return file_list

all_image = list_file_in_path()

train_dataset = list_file_in_path()
train_dataset = np.array(train_dataset) / 255.0

def generate_real_samples(dataset, n_samples):
    dataset = np.array(dataset)
    ix = randint(0,  dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

x_real, y_real = generate_real_samples(train_dataset, 100)

plt.imshow(x_real[0])

###################################11.1.2#####################################
############################## Prepare Fake Image ###########################

def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

def test_generate_fake_samples(n_samples,target_size=img_size):
    hwc = (target_size[0],target_size[1],3)
    X = randn(n_samples, *hwc).astype(np.uint8)
    y = zeros((n_samples, 1))
    return X, y

x_fake, y_fake = test_generate_fake_samples(100)

# View Training Images and Validation Images
def show_all_image():
    plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_real[i], cmap=plt.cm.binary)
        plt.xlabel('Real')
    plt.show()
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_fake[i], cmap=plt.cm.binary)
        plt.xlabel('Fake')
    plt.show()

    
show_all_image()



###################################11.2#####################################
################# Create Discriminator Model (D) and Training #################

def define_discriminator(input_shape = (img_size[0],img_size[1],3)):
      # DEFINE MODEL
    model = Sequential()
    #normal
    model.add(Conv2D(64,(3,3),strides=(2,2),padding='same',input_shape= input_shape))
    model.add(Activation(LeakyReLU(0.2)))
    #down
    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(Activation(LeakyReLU(0.2)))
    #down
    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(Activation(LeakyReLU(0.2)))
    #down
    model.add(Conv2D(256,(3,3),strides=(2,2),padding='same'))
    model.add(Activation(LeakyReLU(0.2)))
    #classifier
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    # COMPILE MODEL
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#Create Model and pre-train Discriminator


def train_discriminator(model,x_real,x_fake,y_real,y_fake,epochs=100,batch_size=128):
    # TRAIN MODEL
    d_loss = []
    d_acc = []
    for epoch in range(epochs):
        # TRAIN DISCRIMINATOR
        # REAL DATA
        idx = np.random.randint(0, x_real.shape[0], batch_size)
        real_images = x_real[idx]
        real_labels = y_real[idx]
        d_loss_real, d_acc_real = model.train_on_batch(real_images, real_labels)
        # FAKE DATA
        idx = np.random.randint(0, x_fake.shape[0], batch_size)
        fake_images = x_fake[idx]
        fake_labels = y_fake[idx]
        d_loss_fake, d_acc_fake = model.train_on_batch(fake_images, fake_labels)
        # AVERAGE LOSS AND ACCURACY
        d_loss.append((d_loss_real + d_loss_fake) / 2)
        d_acc.append((d_acc_real + d_acc_fake) / 2)
        print('>%d, d_loss=%.3f, d_acc=%.3f' % (epoch + 1, d_loss[-1], d_acc[-1]))
    return d_loss, d_acc


d_model = define_discriminator()
d_model.summary()
train_discriminator(d_model,x_real,x_fake,y_real,y_fake,batch_size=128)

######################################## Create Generator Model #################################
# define the standalone generator model
def define_generator(latent_dim,disc_output_shape=(256,4,4)):
    model = Sequential()
    # foundation for 4x4 image
    c,w,h = disc_output_shape
    n_nodes = c*w*h
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((w, h, c)))
    # upsample
    model.add(Conv2DTranspose(128, (w, h), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample
    model.add(Conv2DTranspose(128, (w, h), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample
    model.add(Conv2DTranspose(128, (w, h), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model


g_model = define_generator(latent_dim=100)
g_model.summary()

################################ Visual Layer ########################################
visualkeras.layered_view(d_model,draw_shapes=1,legend=True,padding_left=50,padding_vertical=75)
visualkeras.layered_view(g_model,draw_shapes=1,legend=True)
gan_model_view = Sequential()
gan_model_view.add(g_model)
gan_model_view.add(d_model)
visualkeras.layered_view(gan_model_view,draw_shapes=1,legend=True)
     

###################################11.3#####################################
######################## GAN Training and Prediction #######################

def define_gan(g_model, d_model, image_shape=(img_size[0],img_size[1],3)):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model





# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def train(g_model, d_model, gan_model, dataset, latent_dim,n_batch=128,n_epochs=50):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    d_loss1 = 0
    d_loss2 = 0
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
        # get randomly selected 'real' samples

            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator✬s error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this epoch
        print('>%d, discriminator_Loss=%.3f, generator_Loss=%.3f' % (i+1, d_loss1, d_loss2))
        # evaluate the model performance, sometimes
        if (i+1) % 1000 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    print(X_real.shape, y_real.shape)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save the generator model tile file
    filename = './Saved_model/generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)
    
gan_model = define_gan(g_model, d_model)
gan_model.summary()

train(g_model, d_model, gan_model, train_dataset, latent_dim=100,n_epochs=100)

X = g_model.predict(generate_latent_points(100,1))

# plot the result
def plot_all_image():
    plt.figure(figsize=(10,10))
    for i in range(25):
        X = g_model.predict(generate_latent_points(100,1))
        plt.subplot(5,5,i+1)
        plt.imshow(X[0,:,:,:])
        plt.axis('off')
    plt.show()
plot_all_image()

####################################### Load model 

loaded_model = tf.keras.models.load_model('./Saved_model/generator_model_3000.h5')

#test on real image
pick_img = x_real[8]
real_im_test = cv2.resize(pick_img,(6,6))
plt.imshow(pick_img)
plt.show()
plt.imshow(real_im_test)
plt.show()
real_im_test = np.array(real_im_test)
real_im_test = real_im_test.reshape(real_im_test[0].shape[0]*real_im_test[1].shape[0]*3)
real_im_test = real_im_test[0:100]
real_im_test = real_im_test.reshape(1,100)

pred = loaded_model.predict(real_im_test)
plt.imshow(pred[0,:,:,:])
