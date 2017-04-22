import numpy as np
import os, pickle, cv2, glob
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils

img_rows = 240
img_cols = 320
color_type = 3 

num_classes = 10
batch_size = 32
np.random.seed(133)

def get_im_cv2(path, img_rows, img_cols, color_type=3):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols, color_type=3):

    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
    return X_train, y_train

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def read_train_data():

    train, target = load_train(img_rows, img_cols, color_type)
    X_train, X_test, y_train, y_test = split_validation_set(train, target, 0.2)
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, color_type)
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, color_type)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
        
    print('Train shape:', X_train.shape)
    print('Test shape:', X_test.shape)
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = read_train_data()


dataset = open('dataset.pkl', 'wb')
pickle.dump((X_train, y_train, X_test, y_test), dataset)
train.close()
