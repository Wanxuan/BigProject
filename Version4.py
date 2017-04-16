
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from PIL import Image

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.cross_validation import train_test_split

img_height = 640
img_width = 480
num_classes = 10
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
    data = np.zeros((num, img_width, img_height, 3), dtype='uint8')
    label = np.zeros((num,), dtype="uint8")
    for i in range(num):
        img = Image.open(folder+"/"+imgs[i])
        data[i,:,:,:] = np.asarray(img, dtype="uint8")
    label += int(folder.split("c")[1])           
    return data, label
    
def merge_folder(folders):
    
    x_train = np.zeros((num_train, img_width, img_height, 3), dtype='uint8')
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

folders = maybe_extract(filename)
num_train = dataset_size()
x_train, y_train = merge_folder(folders) 

r = np.random.permutation(len(y_train))
train = x_train[r,:,:,:] 
target = y_train[r]
print(train.shape)
print(len(target))

