import numpy as np
import pickle, h5py, time

import keras
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers, regularizers
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV

batch_size = 32
num_classes = 10

driver_pkl = open('driver.pkl', 'rb')
driver_id, unique_drivers = pickle.load(driver_pkl)
file = h5py.File('train.h5', 'r')
data = file['data'][:]
label = file['label'][:]
file.close()

file = h5py.File('test.h5', 'r')
x_test = file['X_test'][:]
y_test = file['y_test'][:]
file.close()
print("Data loaded!")

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
        data = []
        target = []
        index = []
        for i in range(len(driver_id)):
                if driver_id[i] in driver_list:
                        data.append(train_data[i])
                        target.append(train_target[i])
                        index.append(i)
        data = np.array(data, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        index = np.array(index, dtype=np.uint32)
        return data, target, index

yfull_train = dict()
unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                 'p050', 'p051', 'p052', 'p056']
x_train, y_train, train_index = copy_selected_drivers(data, label, driver_id, unique_list_train)
unique_list_valid = ['p061']
# x_val, y_val, val_index = copy_selected_drivers(data, label, driver_id, unique_list_valid)

print('Start Single Run')
print('Train Sample: ', x_train.shape, len(y_train))
# print('Validation Sample: ', x_val.shape, len(y_val))

start = time.clock()
# print('Train drivers: ', unique_list_train)
# print('Test drivers: ', unique_list_valid)

def create_model(learn_rate = 0.01, momentum=0):
        input_tensor = Input(shape=x_train.shape[1:])
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        x = base_model.output
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', W_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.5)(x)
        prediction = Dense(10, activation='softmax')(x)

        model = Model(input=base_model.input, output=prediction)
        opt = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

np.random.seed(7)

model = KerasClassifier(build_fc=create_model, nb_epoch=100, batch_size=batch_size, verbose=0)
learn_rate = [0.00001, 0.0001, 0.001, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_param_))
for params, mean_score, scores in grid_result.grid_scores_:
  print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
