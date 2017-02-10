#code to help define the model
#this code is used for the PART A questions

import functools
import tensorflow as tf
import numpy as np

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



#function to generate random batches of the data
def create_random_batch(x,y,batch_size):
    indices = np.random.choice(x.shape[0], batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    return x_batch,y_batch