import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
from skimage.feature import hog
from scipy import spatial
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing.image import load_img
image = cv2.imread('./main.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 


tile_size = [10,10]

def mean_each_sub_image(image, tile_size):
    h, w, c = image.shape
    h_tiles = int(h/tile_size[0])
    w_tiles = int(w/tile_size[1])
    tiles = np.zeros((h_tiles, w_tiles, c))
    for h in range(h_tiles):
        for w in range(w_tiles):
            tiles[h,w] = np.mean(image[h*tile_size[0]:(h+1)*tile_size[0], w*tile_size[1]:(w+1)*tile_size[1], :], axis=(0,1))
    return tiles.astype(dtype=np.uint8)

image_feature = mean_each_sub_image(image, tile_size)
plt.imshow(image_feature)


#####################################8.1.2####################################

all_files = glob.glob('./Background/*.jpg')

tile = load_img(target_size=tile_size,path=all_files[0],color_mode='rgb')
tiles = []
for i in range(len(all_files)):
    tile = load_img(target_size=tile_size,path=all_files[i],color_mode='rgb')
    tiles.append(np.array(tile))

Tile_features = []
for tile in tiles:
    mean_color = np.array(tile).mean(axis=(0,1))
    Tile_features.append(mean_color)

for i in range(5):
    print(Tile_features[i])

for i in range(5):
    plt.bar(i+1,Tile_features[i][0])
    plt.bar(i+1,Tile_features[i][1])
    plt.bar(i+1,Tile_features[i][2])
    
#####################################8.2####################################

h_tiles = int(image.shape[0]/tile_size[0])
w_tiles = int(image.shape[1]/tile_size[1])


tree = spatial.KDTree(Tile_features)

closet_tiles = np.zeros((h_tiles, w_tiles, 3))
for h in range(h_tiles):
    for w in range(w_tiles):
        closet = tree.query(image_feature[h,w])
        closet_tiles[h,w] = np.array(closet[1])
        
plt.imshow(closet_tiles[:,:,0], cmap='viridis')


#####################################8.3####################################

main_photo = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

for h in range(image_feature.shape[0]):
    for w in range(image_feature.shape[1]):
        h_tile, y_tile = h*tile_size[0], w*tile_size[1]

        index = closet_tiles[h, w, 0]
        main_photo[h_tile:int(h_tile+tile_size[0]), y_tile:int(y_tile+tile_size[1]), :] = tiles[int(index)]
        
plt.imshow(main_photo)

