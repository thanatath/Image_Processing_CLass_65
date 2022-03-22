import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array, array_to_img


IMG = cv2.imread("./Grid_Image.JPG")

reduce_factor = [1,7,15,30]
scale_factor = 1/np.array(reduce_factor)

interpolation_method = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA]
interpolation_name = ["NEAREST","LINEAR","CUBIC","AREA"]

def display_plot_interpolation(image): 
    for i in range(len(interpolation_method)):
        for j in range(len(reduce_factor)):
            interpolation_method[i]
            img_resize = cv2.resize(image,(int(image.shape[1]*scale_factor[j]),int(image.shape[0]*scale_factor[j])),interpolation=interpolation_method[i])
            plt.subplot(len(interpolation_method),len(reduce_factor),i*len(reduce_factor)+j+1),plt.imshow(img_resize,cmap='gray')
            plt.title(str(interpolation_name[i]))
    plt.show()

display_plot_interpolation(IMG)



############################################### 7.2 ###########################################################

fill_method = ['constant','nearest','reflect','wrap']
Npic=20
rotation_range = 90
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True

IMG2 = load_img('./img.jpg') 
IMG2_reshape = img_to_array(IMG2)  # this is a Numpy array with shape (3, , )
IMG2_reshape = IMG2_reshape.reshape((1,) + IMG2_reshape.shape)  # this is a Numpy array with shape (1, 3, , )


 


for fill_methods in fill_method:
        datagen = ImageDataGenerator(rotation_range=90,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode=fill_methods)

#Create our batch of one image
pic = datagen.flow(IMG2_reshape, batch_size = 1)

#Random generate transformed images and write to a video file
for i in range(4):
    batch = pic.next()
    img_result = batch[0].astype('uint8')
    img_ready = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
out.write(img_ready)
    
    
    

    
 
 
    
    
 