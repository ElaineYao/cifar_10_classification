import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from keras.utils import np_utils
import scipy.misc

base_dir = './cifar-10-python/cifar-10-batches-py/'


def unpickle(file):
    with open(file, 'rb')as f:
        dict = pickle.load(f, encoding='latin1')
    return dict

# Transform the data into 10000*5  size: (32,32,3) training data
def transform():
    tem_train_X = []
    tem_train_Y =[]
    tem_test_X = []
    tem_test_Y = []
    # For train_data

    # Transform the training data
    for i in range(1,6):
        train_path = os.path.join(base_dir, 'data_batch_')+ str(i)
        train_dict = unpickle(train_path)
    # There are 4 key-value pairs in train_dict:
    # 'batch_lable': 'training batch 1 of 5'
    # 'labels': [6,9,9...]
    # 'data' : [154,126,105,...,139,142,144]  -> pixel values of the picture
    #          The shape of data is 10000*3072
    # 'filenames' : ['leptodactylus_pentadactylus_s_000004.png',...]
        train_data = train_dict['data']
        train_label = train_dict['labels']
    # The first 1024 entries contain the red channel values
    # The next 1024 the green, and the final 1024 the blue.
    # The image is stored in row-major order
    # so that the first 32 entries of the array are
    # the red channel values of the first row of the image.
    # Transform the data into 10000*32*32*3 shape
        train_data = train_data.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float32')
        #print(train_data.shape)
        train_label = np.array(train_label)
        tem_train_X.append(train_data)
        tem_train_Y.append(train_label)
        train_X = np.concatenate(tem_train_X)
        train_Y = np.concatenate(tem_train_Y)

    # for i in range(50000):
    #     train_X[i] = scipy.misc.toimage(train_X[i], cmin=0.0, cmax=255)

    test_path = os.path.join(base_dir,'test_batch')
    test_dict = unpickle(test_path)
    test_data = test_dict['data']
    test_label = test_dict['labels']
    test_data = test_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    test_label = np.array(test_label)
    tem_test_X.append(test_data)
    tem_test_Y.append(test_label)
    test_X = np.concatenate(tem_test_X)
    test_Y = np.concatenate(tem_test_Y)

    return train_X, train_Y, test_X, test_Y

# return label_names, which is a dict, label_names[0]="airplane"
def get_label_name():
    label_path = os.path.join(base_dir, 'batches.meta')
    label_dict = unpickle(label_path)
    label_names = label_dict['label_names']
    return label_names

# plot the training/test data and its label
def show_pictures(X, Y, label_names):
    # plot the data
    n = 6
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(320 + 1 + i)  # 321 means 3*2 grid, the first figure
        plt.imshow(np.uint8(X[i]))
    plt.show()
    print('The {}th label is {}'.format(i, label_names[Y[i]]))

def preprocess():
    (train_X, train_Y, test_X, test_Y) = transform()
    # Normalize the data
    train_X = train_X/255.0
    test_X = test_X/255.0
    # Perform the one-hot encoding
    train_Y = np_utils.to_categorical(train_Y)
    test_Y = np_utils.to_categorical(test_Y)
    return train_X, train_Y, test_X, test_Y

