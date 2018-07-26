import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model

from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize

from extract_function import *

#img_folders = ["OCT_Case38","OCT_Case39","OCT_Case5"]

img_folders = ["OCT_Case38"]
img_size = (256,256)
center_shift = (1024/img_size[0], 1024/img_size[1])


for folder in img_folders:
    img_list = os.listdir(folder)
    n_img = len(img_list)
    img_data = np.zeros((n_img, img_size[0],img_size[1],3),dtype=float)
    
    center_folder = str.split(folder,"_")[1]
    center_x_data = dict()
    center_y_data = dict()
    for i, img_path in enumerate(img_list):
        img = Image.open(folder + '/' + img_path)
        img = img.resize(img_size)
        img = np.array(img)
        img_data[i] = img
        index = (str.split(img_path,'.'))[0]
        center_path = "Final_Detection_Results/" + index + ".txt"
        center = pd.read_table(center_path, header = None, sep= " ")
        center_x_data[i] = list(center[0]/center_shift[0])
        center_y_data[i] = list(center[1]/center_shift[1])
        

        
patch_h = 30
patch_w = 30
img_to_test = image_sliding_window(img_data[0], patch_h, patch_w, strides=1)

new_img_to_test = np.empty((img_to_test.shape[0],48,48,3))
for i in range(len(img_to_test)):
    new_img_to_test[i] = resize(img_to_test[i], (48, 48), mode='reflect')

img_to_test = new_img_to_test

## Model 
base_model = VGG16(weights='imagenet', include_top=False, input_shape = img_to_test.shape[1:])
x = base_model.output

x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights("256_best_weights.h5")
        

print("predicting")
labels = model.predict(img_to_test)
recombine_img = recombine_test(img_data[0],patch_h,patch_w,1, labels)

write_hdf5(recombine_img,"recombine_img")