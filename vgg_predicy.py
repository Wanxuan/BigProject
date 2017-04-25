from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

input_tensor = Input(shape=(1, 480, 640, 3))
model = VGG16(weights='imagenet', input_tensor=input_tensor)

img_path = 'test/img_1.jpg'
img = image.load_img(img_path, target_size=(480, 640))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
