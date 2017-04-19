import numpy as np
import os, pickle, cv2

img_row = 24
img_col = 32
color_type = 1 

num_classes = 10
batch_size = 32
nb_epoch = 1
np.random.seed(133)

def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols, color_type=1):

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
    cache_path = os.path.join('cache', 'train_w_' + str(img_row) + '_h_' + str(img_cols) + '.dat')
    if not os.path.isfile(cache_path):
        train, target = load_train(img_rows, img_cols, color_type)
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
    
X_train, X_test, y_train, y_test = read_train_data()


train_model = open('dataset.pkl', 'wb')
pickle.dump((X_train, X_test, y_train, y_test), train_model)
train_model.close()
