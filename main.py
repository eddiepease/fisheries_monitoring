##########
# code to solve fish problem
##########

import os
import glob
import datetime
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
import warnings

from read_data import load_saved_normalised_train_data, load_saved_normalised_test_data
from helper_code import create_submission, dict_to_list, merge_several_folds_mean
from model import create_model

def run_cv_convnet_whole_image(nfolds=10):
    print('Running the conv net that applies to the whole image....')
    # input image dimensions
    batch_size = 16
    nb_epoch = 30
    random_state = 51

    train_data, train_target, train_id = load_saved_normalised_train_data(saved=True)

    yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf.split(train_data):
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cv_conv_bb_image(nfolds=10):
    print('Using CV to assess the accuracy of the bb algorithm')

    # run a cross-validation loop as above to
    # a) assess the accuracy of the bb classifier on its own

    pass

#TODO: rethink this according to thoughts on paper

def run_full_convnet_bb_image():
    print('Running the convnet that applies to the bounding box images...')

    #refer to saved location of the bb image

    # then run a model on the whole dataset

    #output this fitted model


def run_cross_validation_process_test(info_string, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = load_saved_normalised_test_data(saved=True)
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cv_convnet_whole_image(num_folds)

    run_cross_validation_process_test(info_string, models)
