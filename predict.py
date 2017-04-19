import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

pkl_file = open('dataset.pkl', 'rb')
X_train, X_test, y_train, y_test = pickle.load(pkl_file)

#读取model  
model = model_from_json(open('2_model.json').read())  
model.load_weights('2_model_weights.h5')

def get_result(result):
    # 将 one_hot 编码解码
    resultstr = str(np.argmax(result))
    return resultstr

n_test = X_test.shape[0]
index = random.randint(0, n_test-1)
y_pred = model.predict(X_test[index])
print(X_test[index])
print(get_result(y_pred))

plt.title('real: %s\npred:%s'%(get_result(y_test[index]), get_result(y_pred)))
plt.imshow(X_test[index,:,:,0])
plt.axis('off')
