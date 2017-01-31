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

#function to generically define a fully connected layer
def fc_layer(input_tensor, input_size, output_size, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        W = weights([input_size, output_size])
        b = bias(output_size)
        z = tf.matmul(input_tensor, W) + b
        activations = act(z)
        return activations

#function to generically define a convolutional layer
def conv_layer(input_tensor,patch_size,input_size,output_size,stride_shape,padding,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        W = weights([patch_size,patch_size,input_size,output_size])
        b = bias(output_size)
        z = tf.nn.conv2d(input=input_tensor,filter=W,strides=stride_shape,
                         padding=padding) + b
        activations = act(z)
        return activations

#function to generically define a max pooling layer
def max_pooling(input_tensor,pool_size,padding,layer_name):
    with tf.name_scope(layer_name):
        result = tf.nn.max_pool(input_tensor,ksize=[1,pool_size,pool_size,1],
                                strides=[1,pool_size,pool_size,1],padding=padding)
        return result

#defining weights
def weights(shape):
    W = tf.truncated_normal(shape)
    return tf.Variable(W)

#defining bias
def bias(output_size):
    b = tf.constant(0.1, dtype=tf.float32, shape=[output_size])
    return tf.Variable(b)

#function to flatten a convolutional layer
def flatten(input_tensor):
    shape = input_tensor.get_shape().as_list()  # a list: [None, dim_1, dim_2, dim_3 etc]
    dim = np.prod(shape[1:])  # dim = dim_1 * dim_2 * dim_3 etc
    result = tf.reshape(input_tensor, [-1, dim])
    return result,dim

#function to generate random batches of the data
def create_random_batch(x,y,batch_size):
    indices = np.random.choice(x.shape[0], batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    return x_batch,y_batch