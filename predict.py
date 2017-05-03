import h5py, random
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.metrics import log_loss

img_rows = 240
img_cols = 320
num_epochs = 50

file = h5py.File('final_test.h5', 'r')
x_test = file['x_test'][:]
file.close()

#读取model  
model = model_from_json(open('new3_model.json').read())  
model.load_weights('new3_model.h5')

def create_submission(predictions, test_id, info):
  result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
  result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
  now = datetime.datetime.now()
  if not os.path.isdir('submit'):
    os.mkdir('submit')
  suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
  sub_file = os.path.join('submit', 'submission_' + suffix + '.csv')
  result.to_csv(sub_file, index=False)

score = log_loss(y_test, y_pred)
print('Score log_loss:', score)
y_pred = model.predict(x_test, batch_size=128, verbose=1)
print(y_pred.shape)
test_pred = []
test_pred.append(y_pred)
info_string = 'loss_' + str(score) + '_ep_' + str(num_epochs)
create_submission(test_pred, test_id, info_string)
