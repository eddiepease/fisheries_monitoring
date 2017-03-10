#read in the data and normalize

import os
import glob
import time
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer
import zipfile
from PIL import Image



def get_im(path):
    img = imread(path)
    resized = resize(img,(32,32,3))
    #imsave('test.png',resized) # to test resize working
    return resized

def one_hot(X):
    one_hot_label = LabelBinarizer().fit_transform(X)
    #one_hot_label = lb.transform(X)
    return one_hot_label

# #unit test
# test = np.array(['ALF','UGA','TER','ALF','TUE','UGA'])
# test_output = one_hot(test)
# print(test_output)


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('data', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('data', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def read_and_normalize_train_data(save_as_npy=False):
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    # print('Reshape...')
    # train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255.
    train_target = one_hot(train_target)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    if save_as_npy:
        print('Saving training data as .npy files')
        np.save(file='data/train_data.npy', arr=train_data)
        np.save(file='data/train_target.npy', arr=train_target)
        np.save(file='data/train_id.npy', arr=train_id)

    return train_data, train_target, train_id


def read_and_normalize_test_data(save_as_npy=False):
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    # test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255.

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    if save_as_npy:
        print('Saving test data as .npy files')
        np.save(file='data/test_data.npy', arr=test_data)
        np.save(file='data/test_id.npy', arr=test_id)

    return test_data, test_id


def load_saved_normalised_train_data(saved):

    if saved == True:

        train_data, train_target, train_id = np.load('data/train_data.npy'), \
                                             np.load('data/train_target.npy'), \
                                             np.load('data/train_id.npy')

    else:
        train_data, train_target, train_id = read_and_normalize_train_data()

    return train_data, train_target, train_id


def load_saved_normalised_test_data(saved):
    if saved == True:

        test_data, test_id = np.load('data/test_data.npy'), \
                             np.load('data/test_id.npy')

    else:
        test_data, test_id = read_and_normalize_test_data()

    return test_data, test_id


if __name__ == "__main__":
    read_and_normalize_train_data()
    read_and_normalize_test_data()

