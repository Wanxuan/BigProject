import numpy as np
import os, time, cv2, glob, h5py, math
import keras

img_rows = 240
img_cols = 320
color_type = 3 

num_classes = 10
batch_size = 32

def get_im_cv2(path, img_rows, img_cols, color_type=3):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols, color_type=3):

    print('Read test images')
    start_time = time.time()
    test = []
    test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)    
    for fl in files:
        flbase = ps.path.basename(fl)
        if getDocSize(fl) < 1.0:
            new_img = np.zeros((img_rows, img_cols, color_type), dtype='uint8')
        else:
            img = cv2.imread(fl)
            new_img = cv2.resize(img, (img_rows, img_cols))
        test.append(new_img)
        test_id.append(flbase)
        total += 1
        if total%thr == 0:
          print('Read {} images from {}'.format(total, len(files)))
          
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test, test_id

def read_and_normalize_train_data():
  
    test, test_id = load_train(img_rows, img_cols, color_type)
    test = np.array(test, dtype=np.float32)
    test = test.reshape(test.shape[0], img_rows, img_cols, color_type)
    test /= 255

    print('Test shape:', test.shape)
    return test, test_id

test, test_id = read_and_normalize_data()

test = h5py.File('final_test.h5', 'w')
test.create_dataset('test', data=test, compression="gzip")
test.create_dataset('test_id', data=test_id,  compression="gzip")
test.close()
