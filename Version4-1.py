import pickle
import random
import matplotlib.pyplot as plt
from keras.models import model_from_json

pkl_file = open('test.pkl', 'rb')
X_test = pickle.load(pkl_file)
y_test = pickle.load(pkl_file)

img_width = 640
img_height = 480
#读取model  
model = model_from_json(open('my_model.json').read())  
model.load_weights('my_model_weights.h5')

def get_result(result):
    # 将 one_hot 编码解码
    resultstr = str(np.argmax(result[i]))
    return resultstr

n_test = X_test.shape[0]
index = random.randint(0, n_test-1)
y_pred = model.predict(X_test[index].reshape(1, img_height, img_width, 3))

plt.title('real: %s\npred:%s'%(get_result(y_test[index]), get_result(y_pred)))
plt.imshow(X_test[index,:,:,0])
plt.axis('off')
