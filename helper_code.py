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

#function to define the variable scope
def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__, str(__name__)):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


#TODO: function which creates a confusion matrix
def create_confusion_matrix(prediction,y_label,name):

    test_true = pd.Series(np.argmax(y_label, axis=1), name="Actual")
    test_prediction = pd.Series(prediction, name="Pred")
    con_mat = pd.crosstab(test_true, test_prediction)

    #plot heatmap
    ax = sns.heatmap(con_mat, annot=True, fmt='.4g')
    plt.savefig('figures/' + name + '.png')

    return con_mat



#function to generate random batches of the data
def create_random_batch(x,y,batch_size):
    indices = np.random.choice(x.shape[0], batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    return x_batch,y_batch#,indices

# #unit test
# x = np.random.rand(10,5)
# y = np.random.rand(10,1)
# _,_,ind = create_random_batch(x,y,7)
# print(ind)


# save the model params to the hard drive
def save_model(session,model_folder,model_name):
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    saver = tf.train.Saver()
    saver.save(session, model_folder + str(model_name))

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    #result1 = pd.DataFrame(predictions, columns=['BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'results/submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

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
