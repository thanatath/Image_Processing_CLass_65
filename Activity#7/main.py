import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array, array_to_img


IMG = cv2.imread("./Grid_Image.JPG")

reduce_factor = [2,4,5,7]
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
Npic=60
rotation_range = 40
width_shift_range = 0.5
height_shift_range = 0.2
shear_range = 0.5
zoom_range = [0.1,0.5]
horizontal_flip = True

IMG2 = load_img('./Grid_Image.JPG') 
IMG2_reshape = img_to_array(IMG2)  # this is a Numpy array with shape (3, , )
IMG2_reshape = IMG2_reshape.reshape((1,) + IMG2_reshape.shape)  # this is a Numpy array with shape (1, 3, , )


 



for m in fill_method:
    datagen = ImageDataGenerator(rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    horizontal_flip=horizontal_flip,
                                    fill_mode=m)
    pic = datagen.flow(IMG2_reshape,batch_size=1)
    #Write the image to the Video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output.avi',fourcc, 1, (IMG2_reshape.shape[2],IMG2_reshape.shape[1]))
    for i in range(1,Npic):
        batch = pic.next()
        im_results = batch[0].astype('uint8')
        out.write(im_results)
    out.release()
    
 
    
    
 