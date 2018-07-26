from PIL import Image
import numpy as np
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from skimage.color import gray2rgb

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--experiment",
                  action="store",
                  type="string",
                  dest="experiment",
                  help="experiment name")

parser.add_option("-l", "--learningrate",
                  action="store",
                  type="float",
                  dest="learningrate",
                  help="Learning rate")

parser.add_option("-n", "--n_epochs",
                  action="store",
                  type="int",
                  dest="n_epochs",
                  help="Number of epochs")


parser.add_option("-a", "--augmentation",
                  action="store_true",
                  default = False,
                  dest="augmentation",
                  help="augmentation")

parser.add_option("-b", "--base",
                  action="store",
                  type="string",
                  dest="base",
                  help="base model")

parser.add_option("-m", "--model",
                  action="store",
                  type="string",
                  dest="architecture",
                  help="The top layer")


(options, args) = parser.parse_args()

# Load hdf5 function
def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]
    

# Parameters setting
experiment = options.experiment
learningrate = options.learningrate
epochs = options.n_epochs
augmentation = options.augmentation
base = options.base
architecture = options.architecture

# Data Loading ...
train_labels = pd.read_csv('train.csv',  header=None, index_col=0)

image_size = (224,224)
## Read in the training images
train_images = []
train_dir = './train/'
train_files = os.listdir(train_dir)
for f in train_files:
    img = Image.open(train_dir + f)
    img = img.resize(image_size)
    img_arr = np.array(img)
    train_images.append(img_arr)

train_X = np.array(train_images)

train_labels = train_labels.reindex(train_files)

label_transformer = LabelBinarizer()
train_y = label_transformer.fit_transform(train_labels)
train_y = (train_y[:,10] == 1) * 1

# Data Preprocessing ...
train_X = train_X.astype(np.float32)
train_X /= 255.
train_X = gray2rgb(train_X)


# Data Augmentation
if augmentation:
    print("Data Augmentation...")

    for i in range(10):
        new_X = load_hdf5("augment_data/batch"+str(i)+"_aug_X.h5")
        new_y = load_hdf5("augment_data/batch"+str(i)+"_aug_y.h5")

        train_X = np.concatenate((train_X, new_X), axis = 0)
        train_y = np.concatenate((train_y, new_y), axis = 0)

    print("Finish augmentation")
    print("X training Shape: " + str(train_X.shape))
    print("y training Shape: " + str(train_y.shape))

#weight_y = np.where(train_y == 1)
#weights = 5483/np.sum(train_y, axis = 0)
#weights = np.where(np.sum(train_y, axis = 0) > 1000, 1, 10)
#sample_weights = weights[weight_y[1]]



# Fit the model
#base_model = DenseNet121(include_top=False, weights='imagenet')
base_model = None
if base == "VGG16":
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = train_X.shape[1:])
elif base == "VGG19":
    base_model = VGG19(weights='imagenet', include_top=False, input_shape = train_X.shape[1:])
elif base == "Dense":
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape = train_X.shape[1:])

x = base_model.output


if architecture == "ap1":
    x = GlobalAveragePooling2D()(x)
elif architecture == "flat1":
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.5)(x)

x = Dense(128, activation = 'relu')(x)


predictions = Dense(1, activation='sigmoid')(x)

# define the model to be trained
model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())

for layer in base_model.layers:
    layer.trainable = False

#for i in range(len(base_model.layers[:-4])):
#    base_model.layers[i].trainable = False

adam = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint( 'result/' + experiment + '_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

batch_size = 32
model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=500,
          verbose=2,
          validation_split = 0.2,
          callbacks=[checkpointer]
          )