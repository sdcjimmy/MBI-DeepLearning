import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

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


patch_h = 15
patch_w = 15
N_patches = 3000

patches_data = np.empty((img_data.shape[0], N_patches, patch_h, patch_w,3))
labels_data = np.empty((img_data.shape[0],N_patches))

for i in range(img_data.shape[0]):
    patches, labels = image_cropping(img_data[i],patch_h,patch_w,center_x_data[i],center_y_data[i],N_patches)
    patches_data[i] = patches
    labels_data[i] = labels

    
train_X = patches_data.reshape((-1,patch_h,patch_w,3))
train_y = labels_data.reshape(-1)


# Fit VGG 16 input
new_train_X = np.empty((train_X.shape[0],48,48,3))
for i in range(len(train_X)):
    new_train_X[i] = resize(train_X[i], (48, 48), mode='reflect')

train_X = new_train_X
train_X = preprocess_input(train_X)

base_model = VGG16(weights='imagenet', include_top=False, input_shape = train_X.shape[1:])
x = base_model.output

x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


    
checkpointer = ModelCheckpoint('256_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=10,
          verbose=2,
          validation_split = 0.2,
          callbacks=[checkpointer]
         )
