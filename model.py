import numpy as np
import h5py, pickle

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

batch_size = 32
num_classes = 10
np.random.seed(133)

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
    
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 4, 4))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(             
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=True, 
        vertical_flip=False)

datagen.fit(x_train)
    
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    samples_per_epoch=x_train.shape[0], 
                    nb_epoch=10, validation_data=(x_test, y_test), 
                    nb_val_samples=x_test.shape[0])

score = model.evaluate(x_test, y_test, verbose=1) # 评估测试集loss损失和精度acc
print('Test score(val_loss): %.4f' % score[0])  # loss损失
print('Test accuracy: %.4f' % score[1]) # 精度acc

# json_string = model.to_json()  
# open('gen_model.json','w').write(json_string)  
# model.save_weights('gen_model_weights.h5')
model.save_weights('cifar10_model.h5')
with open('cifar10_model.json', 'w') as f:
    f.write(model.to_json())
