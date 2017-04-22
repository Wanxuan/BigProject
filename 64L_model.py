'''
It gets down to 0.44 test logloss in 10 epochs, and down to 0.33 after 20 epochs.
It gets 90.33% accuracy in 20 epochs and does not change too much after 20 epochs.
cifar_10模型的基础上加两层model.add(Conv2D(64, 3, 3))跑25次，loss达到0.31，acc达到91%。
cifar_10模型跑15次就差不多没什么大变化了，15次loss是0.429，acc0.872；20次loss0.401，acc是0.884
'''

import numpy as np
import h5py, pickle

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 32
num_classes = 10
np.random.seed(133)

pkl_file = open('dataset.pkl', 'rb')
X_train, y_train, X_test, y_test = pickle.load(pkl_file)
    
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


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)


model.fit_generator(train_generator, samples_per_epoch=X_train.shape[0], 
                    nb_epoch=20, validation_data=validation_generator, 
                    nb_val_samples=X_test.shape[0])
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
print(score)

model.save_weights('64L_model.h5')
with open('64L_model.json', 'w') as f:
    f.write(model.to_json())
