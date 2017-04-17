import pickle

pkl_file = open('train_model.pkl', 'rb')
model = pickle.load(pkl_file)
X_test = pickle.load(pkl_file)
y_test = pickle.load(pkl_file)

def get_result(result):
    # 将 one_hot 编码解码
    resultstr = str(np.argmax(result[i]))
    return resultstr

n_test = X_test.shape[0]
index = random.randint(0, n_test-1)
y_pred = model.predict(X_test[index].reshape(1, img_width, img_height, 3))

plt.title('real: %s\npred:%s'%(get_result(y_test[index]), get_result(y_pred)))
plt.imshow(X_test[index,:,:,0])
plt.axis('off')
