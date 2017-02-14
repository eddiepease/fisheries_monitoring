##############
#specify the model that is used
##############

import tensorflow as tf
import numpy as np

np.random.seed(2016)

#TODO: add Rob Collyer variable initialisation technique
#W = tf.Variable(tf.truncated_normal([convo_patch, convo_patch, num_channels, convo_depth1],stddev= 1.0 / np.sqrt(float(convo_neurons1))),name='weights')

#creating the main model
def create_model(input_array,retained_pc):

    #conv layer 1
    model = conv_layer(input_array,patch_size=3,input_size=3,
                       output_size=4,stride_shape=[1,1,1,1],padding='SAME',
                       layer_name='conv_1_1') # output: 32 x 32 x 4
    model = conv_layer(model,patch_size=3,input_size=4,
                       output_size=4,stride_shape=[1,1,1,1],padding='SAME',
                       layer_name='conv_1_2') # output: 32 x 32 x 4
    model = max_pooling(model,pool_size=2,padding='SAME',layer_name='conv_1_3') #output: 16 x 16 x 4

    #conv layer 2
    model = conv_layer(model,patch_size=3,input_size=4,
                       output_size=8,stride_shape=[1,1,1,1],padding='SAME',
                       layer_name='conv_2_1') #output: 16 x 16 x 8
    model = conv_layer(model,patch_size=3,input_size=8,
                       output_size=8,stride_shape=[1,1,1,1],padding='SAME',
                       layer_name='conv_2_2') #output: 16 x 16 x 8
    model = max_pooling(model,pool_size=2,padding='SAME',layer_name='conv_2_3') #output: 8 x 8 x 8

    #fc layer
    model,dim = flatten(model) # output: None x 2048
    model = fc_layer(model, input_size=dim, output_size=32, layer_name='fc_1') #output: None x 32
    model = tf.nn.dropout(model, retained_pc)
    model = fc_layer(model, input_size=32, output_size=32,layer_name='fc_2') #output: None x 32
    model = tf.nn.dropout(model, retained_pc)
    model = fc_layer(model, input_size=32, output_size=8,layer_name='fc_3', act=tf.identity) #output: None x 8

    #output
    return model


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