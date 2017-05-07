import numpy as np
import os, pickle, cv2, glob, h5py
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

def get_driver_data():
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    file = open(path, 'r')
    line = file.readline()
    while (1):
        line = file.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    file.close()
    return dr

def load_train(img_rows, img_cols, color_type=3):

    X_train = []
    y_train = []
    driver_id = []
    driver_data = get_driver_data()
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
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
#     data = []
#     target = []
#     index = []
#     for i in range(len(driver_id)):
#         if driver_id[i] in driver_list:
#             data.append(train_data[i])
#             target.append(train_target[i])
#             index.append(i)
#     data = np.array(data, dtype=np.float32)
#     target = np.array(target, dtype=np.float32)
#     index = np.array(index, dtype=np.uint32)
#     return data, target, index

def read_and_normalize_train_data(x_train, y_train, x_test, y_test):

    X_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.uint8)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, color_type)
    X_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.uint8)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, color_type)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
#     X_train = X_train.astype('float32')
    X_train /= 255
#     X_test = X_test.astype('float32')
    X_test /= 255

    print('Train shape:', X_train.shape)
    print('Test shape:', X_test.shape)
    return X_train, X_test, y_train, y_test


data, label, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
# yfull_train = dict()
# unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
#                  'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
#                  'p050', 'p051', 'p052', 'p056', 'p061']
# x_train, y_train, train_index = copy_selected_drivers(data, label, driver_id, unique_list_train)
# unique_list_test = ['p064', 'p066', 'p072', 'p075', 'p081']
# x_test, y_test, test_index = copy_selected_drivers(data, label, driver_id, unique_list_test)
x_train, x_test, y_train, y_test = split_validation_set(data, label, 0.3)
X_train, X_test, y_train, y_test = read_and_normalize_train_data(x_train, y_train, x_test, y_test)
    
# train_driver_id = [driver_id[i] for i in train_index]
    
train = h5py.File('train.h5', 'w')
train.create_dataset('data', data=X_train, compression="gzip")
train.create_dataset('label', data=y_train, compression="gzip")
train.close()

test = h5py.File('test.h5', 'w')
test.create_dataset('X_test', data=X_test, compression="gzip")
test.create_dataset('y_test', data=y_test, compression="gzip")
test.close()

# driver = open('driver.pkl', 'wb')
# pickle.dump((train_driver_id, unique_list_train), driver)
# driver.close()

# test = open('test.pkl', 'wb')
# pickle.dump((X_test, y_test), test)
# test.close()
