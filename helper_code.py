#code to help define the model
#this code is used for the PART A questions

import functools
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

np.random.seed(2016)


def create_confusion_matrix(prediction,y_label,name):

    test_true = pd.Series(np.argmax(y_label, axis=1), name="Actual")
    test_prediction = pd.Series(prediction, name="Pred")
    con_mat = pd.crosstab(test_true, test_prediction)

    #plot heatmap
    ax = sns.heatmap(con_mat, annot=True, fmt='.4g')
    plt.savefig('figures/' + name + '.png')

    return con_mat


def create_submission(df, test_id, info='bb_test'):
    df.loc[:, 'image'] = pd.Series(test_id, index=df.index)
    now = datetime.datetime.now()
    sub_file = 'results/submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    df.to_csv(sub_file, index=False)


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

#this code finds the average of the results from all the folds
#outputs a list of a list - list of prob of every category inside list of every test value
def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


# #unit test
# X = np.random.rand(10,3)
# y = np.random.rand(10,1)
#
# X_train,y_train,X_valid,y_valid = create_validation_set(X,y,0.7)
#
# print(X_train.shape)
# print(y_train.shape)
# print(X_valid.shape)
# print(y_valid.shape)
