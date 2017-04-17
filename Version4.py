import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import pickle
from PIL import Image
import h5py 

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

img_height = 480
img_width = 640
num_classes = 10
batch_size = 32
nb_epoch = 1
np.random.seed(133)

filename = "/home/ubuntu/imgs.zip"

def maybe_extract(filename, force=True):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
    print(root)
    if os.path.isdir('train') and os.path.isdir('test'):
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        zfile = zipfile.ZipFile(filename,'r')
        for filename in zfile.namelist():
            zfile.extract(filename, r'.')
        zfile.close()
    data_folders = [os.path.join('train', d) for d in sorted(os.listdir('train'))
                    if os.path.isdir(os.path.join('train', d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, ten per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def dataset_size():
    
    size = 0
    for i in range(10):
        size += len([x for x in os.listdir(str(folders[i]))])
    print(size)
    return size

def load_image(folder):
    
    imgs = [i for i in os.listdir(folder)]
    num = len(imgs)
    data = np.zeros((num, img_height, img_width, 3), dtype='uint8')
    label = np.zeros((num,), dtype="uint8")
    for i in range(num):
        img = Image.open(folder+"/"+imgs[i])
        data[i,:,:,:] = np.asarray(img, dtype="uint8")
    label += int(folder.split("c")[1])           
    return data, label
    
def merge_folder(folders):
    
    x_train = np.zeros((num_train, img_height, img_width, 3), dtype='uint8')
    y_train = np.zeros((num_train,), dtype="uint8")
    folders_size = 0
    for f in folders:
        print(f)
        data, label = load_image(f)
        folder_size = data.shape[0]
        folders_size += folder_size
        x_train[folders_size-folder_size:folders_size,:,:,:] = data
        y_train[folders_size-folder_size:folders_size] = label
    return x_train, y_train

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def read_train_data():
    cache_path = os.path.join('cache', 'train_w_' + str(img_width) + '_h_' + str(img_height) + '.dat')
    if not os.path.isfile(cache_path):
        x_train, y_train = merge_folder(folders[0:5]) 
        r = np.random.permutation(len(y_train))
        train = x_train[r,:,:,:] 
        target = y_train[r]
        X_train, X_test, y_train, y_test = split_validation_set(train, target, 0.2)
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        cache_data((X_train, X_test, y_train, y_test), cache_path)
    else:
        print('Restore train from cache!')
        (X_train, X_test, y_train, y_test) = restore_data(cache_path)
    print('Train shape:', X_train.shape)
    print('Test shape:', X_test.shape)
    return X_train, X_test, y_train, y_test
    
folders = maybe_extract(filename)
num_train = dataset_size()
X_train, X_test, y_train, y_test = read_train_data()
    
model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu', border_mode='same', input_shape=X_train.shape[1:]))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=1e-3)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
          verbose=1, validation_data=(X_test, y_test))


json_string = model.to_json()  
open('1_model.json','w').write(json_string)  
model.save_weights('1_model_weights.h5')
train_model = open('test.pkl', 'wb')
pickle.dump(X_test, train_model)
pickle.dump(y_test, train_model)
train_model.close()
