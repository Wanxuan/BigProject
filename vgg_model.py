'''
It gets down to 0.44 test logloss in 10 epochs, and down to 0.33 after 20 epochs.
It gets 90.33% accuracy in 20 epochs and does not change too much after 20 epochs.
cifar_10模型的基础上加两层model.add(Conv2D(64, 3, 3))跑25次，loss达到0.31，acc达到91%。
cifar_10模型跑15次就差不多没什么大变化了，15次loss是0.429，acc0.872；20次loss0.401，acc是0.884
'''

import numpy as np
import pickle, h5py

import keras
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from sklearn.model_selection import train_test_split

batch_size = 32
num_classes = 10

driver_pkl = open('driver.pkl', 'rb')
driver_id, unique_drivers = pickle.load(driver_pkl)
file = h5py.File('train.h5', 'r')
x_train = file['data'][:]
y_train = file['label'][:]
file.close()

file = h5py.File('test.h5', 'r')
x_test = file['X_test'][:]
y_test = file['y_test'][:]
file.close()

# def split_validation_set(train, target, test_size):
# #     random_state = 51
#     X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size)
#     return X_train, X_test, y_train, y_test

# x_train, x_test, y_train, y_test = split_validation_set(data, label, 0.2)
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

# yfull_train = dict()
# unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
#                  'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
#                  'p050', 'p051', 'p052', 'p056']
# x_train, y_train, train_index = copy_selected_drivers(data, label, driver_id, unique_list_train)
# unique_list_valid = ['p061']
# x_val, y_val, val_index = copy_selected_drivers(data, label, driver_id, unique_list_valid)

print('Start Single Run')
print('Train Sample: ', len(x_train), len(y_train))
print('Test Sample: ', len(x_test), len(y_test))
# print('Train drivers: ', unique_list_train)
# print('Test drivers: ', unique_list_valid)

#-----------------------------------Cifar10----------------------------------#
# model = Sequential()

# model.add(Conv2D(32, 3, 3, border_mode='same',
#                  input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#-----------------------------------Cifar10 End----------------------------------#

model = VGG16(include_top=True, weights='imagenet',
                                input_tensor=None, input_shape=None,
                                pooling=None,
                                classes=10)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

# add the model on top of the convolutional base
model.add(top_model)

for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

#-----------------------------VGG---------------------------------#

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
    
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    samples_per_epoch=x_train.shape[0], 
                    nb_epoch=50, validation_data=(x_test, y_test), 
                    nb_val_samples=x_test.shape[0])

# model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=10, verbose=1, 
#           validation_split=0.1, validation_data=(x_test, y_test), shuffle=True)


score = model.evaluate(x_test, y_test, verbose=1) # 评估测试集loss损失和精度acc
print('Test score(val_loss): %.4f' % score[0])  # loss损失
print('Test accuracy: %.4f' % score[1]) # 精度acc

model.save_weights('vgg_model.h5')
with open('vgg_model.json', 'w') as f:
    f.write(model.to_json())
          
