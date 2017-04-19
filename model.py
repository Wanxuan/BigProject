import numpy as np
import h5py, pickle

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

img_rows = 24
img_cols = 32
num_classes = 10
batch_size = 32
nb_epoch = 1
np.random.seed(133)

pkl_file = open('dataset.pkl', 'rb')
X_train, X_test, y_train, y_test = pickle.load(pkl_file)
    
model = Sequential()

nb_filters = 8
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 2
model = Sequential()
model.add(Conv2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
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
