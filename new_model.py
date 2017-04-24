'''
It gets down to 0.44 test logloss in 10 epochs, and down to 0.33 after 20 epochs.
It gets 90.33% accuracy in 20 epochs and does not change too much after 20 epochs.
cifar_10模型的基础上加两层model.add(Conv2D(64, 3, 3))跑25次，loss达到0.31，acc达到91%。
cifar_10模型跑15次就差不多没什么大变化了，15次loss是0.429，acc0.872；20次loss0.401，acc是0.884
'''

import numpy as np
import pickle, h5py

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

batch_size = 32
num_classes = 10
np.random.seed(133)

driver_pkl = open('driver.pkl', 'rb')
driver_id, unique_drivers = pickle.load(driver_pkl)
file = h5py.File('train.h5', 'r')
data = file['data'][:]
label = file['label'][:]
file.close()

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
                 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                 'p075']
x_train, y_train, train_index = copy_selected_drivers(data, label, driver_id, unique_list_train)
unique_list_valid = ['p081']
x_val, y_val, val_index = copy_selected_drivers(data, label, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(x_train), len(y_train))
print('Split valid: ', len(x_val), len(y_val))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)

model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
# validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

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

datagen.fit(x_train)
    
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    samples_per_epoch=X_train.shape[0], 
                    nb_epoch=20, validation_data=(x_val, y_val), 
                    nb_val_samples=X_test.shape[0], callbacks=early_stop)

model.save_weights('new_model.h5')
with open('new_model.json', 'w') as f:
    f.write(model.to_json())
    
predictions_valid = model.predict(x_val, batch_size=128, verbose=1)
score = log_loss(y_val, predictions_valid)
print('Score log_loss: ', score)
