import numpy as np
import h5py

import keras
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from sklearn.model_selection import train_test_split

file = h5py.File('train.h5', 'r')
x_train = file['data'][:]
y_train = file['label'][:]
file.close()

file = h5py.File('test.h5', 'r')
x_test = file['X_test'][:]
y_test = file['y_test'][:]
file.close()

input_tensor = Input(shape=x_train.shape[1:])
model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    
generator = datagen.flow(x_train, y_train, batch_size=32)
bottleneck_features_train = model.predict_generator(generator, x_train.shape[1:])
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

generator = datagen.flow(x_test, y_test, batch_size=32)
bottleneck_features_validation = model.predict_generator(generator, x_test.shape[1:])
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
