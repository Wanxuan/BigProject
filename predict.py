import h5py, random
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

img_rows = 240
img_cols = 320

file = h5py.File('test.h5', 'r')
X_test = file['X_test'][:]
y_test = file['y_test'][:]
file.close()

#读取model  
model = model_from_json(open('e20_model.json').read())  
model.load_weights('e20_model.h5')

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

score = model.evaluate(X_test, y_test, verbose=1) # 评估测试集loss损失和精度acc
print('Test score(val_loss): %.4f' % score[0])  # loss损失
print('Test accuracy: %.4f' % score[1]) # 精度acc


# def get_result(result):
#     # 将 one_hot 编码解码
#     resultstr = str(np.argmax(result))
#     return resultstr

# n_test = X_test.shape[0]
# index = random.randint(0, n_test-1)
# y_pred = model.predict(X_test[index].reshape(1, img_rows, img_cols, 1))
# print(X_test[index])
# print(get_result(y_pred))

# plt.title('real: %s\npred:%s'%(get_result(y_test[index]), get_result(y_pred)))
# plt.imshow(X_test[index,:,:,0])
# plt.axis('off')
