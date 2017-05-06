from keras.layers import merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import keras.backend as K
import numpy as np
import pickle, h5py, time

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

def identity_block(x,nb_filter,kernel_size=3):
        """
        identity_block is the block that has no conv layer at shortcut
        """
        k1,k2,k3 = nb_filter
        out = Convolution2D(k1,1,1)(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k3,1,1)(out)
        out = BatchNormalization()(out)

        out = merge([out,x],mode='sum')
        out = Activation('relu')(out)
        return out

def conv_block(x,nb_filter,kernel_size=3, strides=(2, 2)):
        """
        conv_block is the block that has a conv layer at shortcut
        """
        k1,k2,k3 = nb_filter
        out = Convolution2D(k1,1,1,subsample=strides)(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Convolution2D(k3,1,1)(out)
        out = BatchNormalization()(out)
        
        x = Convolution2D(k3,1,1)(x)
        x = BatchNormalization()(x)

        out = merge([out,x],mode='sum')
        out = Activation('relu')(out)
        return out


inp = Input(shape=x_train.shape[1:])
out = ZeroPadding2D((3,3))(inp)
out = Convolution2D(64,7,7,subsample=(2,2))(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((3,3),strides=(2,2))(out)

out = conv_block(out,[64,64,256], strides=(1, 1))
out = identity_block(out,[64,64,256])
out = identity_block(out,[64,64,256])

out = conv_block(out,[128,128,512])
out = identity_block(out,[128,128,512])
out = identity_block(out,[128,128,512])
out = identity_block(out,[128,128,512])

out = conv_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])
out = identity_block(out,[256,256,1024])

out = conv_block(out,[512,512,2048])
out = identity_block(out,[512,512,2048])
out = identity_block(out,[512,512,2048])

out = AveragePooling2D((7,7))(out)
out = Flatten()(out)
out = Dense(10,activation='softmax')(out)

model = Model(inp,out)

# x = base_model.output
# x = Flatten()(x)
# x = Dense(512, activation='relu', W_regularizer=regularizers.l2(0.0001))(x)
# x = Dropout(0.5)(x)
# prediction = Dense(10, activation='softmax')(x)

# model = Model(input=base_model.input, output=prediction)


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0)
filepath='weights_best.h5'
checkPoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    samples_per_epoch=x_train.shape[0], callbacks=[earlyStop, checkPoint],
                    nb_epoch=30, validation_data=(x_val, y_val), 
                    nb_val_samples=x_val.shape[0])

end = time.clock()
print('Running time: %s Seconds'%(end-start))
score = model.evaluate(x_test, y_test, verbose=1) # 评估测试集loss损失和精度acc
print('Validation score(val_loss): %.4f' % score[0])  # loss损失
print('Validation accuracy: %.4f' % score[1]) # 精度acc

model.save_weights('resNet_model.h5')
with open('resNet_model.json', 'w') as f:
        f.write(model.to_json())
          