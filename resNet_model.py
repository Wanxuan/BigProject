'''
It gets down to 0.44 test logloss in 10 epochs, and down to 0.33 after 20 epochs.
It gets 90.33% accuracy in 20 epochs and does not change too much after 20 epochs.
cifar_10模型的基础上加两层model.add(Conv2D(64, 3, 3))跑25次，loss达到0.31，acc达到91%。
cifar_10模型跑15次就差不多没什么大变化了，15次loss是0.429，acc0.872；20次loss0.401，acc是0.884
'''

import numpy as np
import pickle, h5py, time

import keras
from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import train_test_split

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
# def split_validation_set(train, target, test_size):
# #     random_state = 51
#         X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size)
#         return X_train, X_test, y_train, y_test

# x_train, x_val, y_train, y_val = split_validation_set(data, label, 0.1)
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
x_val, y_val, val_index = copy_selected_drivers(data, label, driver_id, unique_list_valid)

print('Start Single Run')
print('Train Sample: ', x_train.shape, len(y_train))
print('Validation Sample: ', x_val.shape, len(y_val))

start = time.clock()
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)

input_tensor = Input(shape=x_train.shape[1:])
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu', W_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.5)(x)
prediction = Dense(10, activation='softmax')(x)

model = Model(input=base_model.input, output=prediction)



datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
# opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0)
filepath='weights_best.h5'
checkPoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    samples_per_epoch=x_train.shape[0], callbacks=[earlyStop, checkPoint],
                    nb_epoch=30, validation_data=(x_val, y_val), 
                    nb_val_samples=x_val.shape[0])

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)

end = time.clock()
print('Running time: %s Seconds'%(end-start))
score = model.evaluate(x_test, y_test, verbose=1) # 评估测试集loss损失和精度acc
print('Validation score(val_loss): %.4f' % score[0])  # loss损失
print('Validation accuracy: %.4f' % score[1]) # 精度acc

model.save_weights('resNet_model.h5')
with open('resNet_model.json', 'w') as f:
        f.write(model.to_json())
          
