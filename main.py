###################
# code to solve qu_1_a
##################

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from helper_code import define_scope,fc_layer,conv_layer,max_pooling,flatten,create_random_batch
import random


class Model:
    def __init__(self, x, y_true):
        self.x = x
        self.y_true = y_true
        self.initial_input_size = int(self.x.get_shape()[1])
        self.target_size = int(self.y_true.get_shape()[1])
        self.build_model
        self.optimize
        self.accuracy
        self.loss

    @define_scope
    def build_model(self):
        layer_1 = fc_layer(self.x,self.initial_input_size,128,
                           layer_name='layer_1')
        layer_2 = fc_layer(layer_1,128,self.target_size,
                           layer_name='second_layer',act=tf.identity)
        return layer_2

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.build_model, self.y_true))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.build_model), 1),tf.argmax(self.y_true, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @define_scope
    def loss(self):
        mistakes = tf.not_equal(tf.argmax(tf.nn.softmax(self.build_model), 1),tf.argmax(self.y_true, 1))
        return tf.reduce_mean(tf.cast(mistakes,tf.float32))

    # @define_scope
    # def summaries(self):
    #     tf.summary.scalar('accuracy', self.accuracy)
    #     tf.summary.scalar('cross_entropy',self.loss)
    #     merged = tf.summary.merge_all()
    #     return merged



if __name__ == "__main__":

    # TODO: work out how to use tensorboard to get a good visualisation of the training of model

    # import dataset with one-hot class encoding
    mnist = input_data.read_data_sets('data/', one_hot=True)

    #define placeholders and model
    x = tf.placeholder(tf.float32, shape=[None,784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    nn_model = Model(x, y_true)
    epochs = 25
    batch_size = 64

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print('----- Epoch', epoch, '-----')
            n_iter = mnist.train.images.shape[0] // batch_size
            total_loss = 0
            for i in range(n_iter):
                x_batch,y_batch = create_random_batch(mnist.train.images,mnist.train.labels,batch_size)
                nn_model.optimize.run(feed_dict={x:x_batch,y_true:y_batch})
                current_loss = nn_model.loss.eval(feed_dict={x:x_batch,y_true:y_batch})
                total_loss += current_loss

            print(' Train loss:', total_loss / n_iter)

            #calc train accuracy
            train_accuracy = nn_model.accuracy.eval(feed_dict={x:mnist.train.images,y_true:mnist.train.labels})
            print(' Train accuracy:', train_accuracy)

            #calc test accuracy
            test_accuracy = nn_model.accuracy.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
            print(' Test accuracy:', test_accuracy)

            # #calc summary
            # if epoch % 5 == 0:
            #     train_writer = tf.summary.FileWriter('/tmp/mnist_logs' + '/train', sess.graph)
            #     #test_writer = tf.summary.FileWriter('/tmp/mnist_logs' + '/test')
            #     summary = nn_model.summaries.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
            #     train_writer.add_summary(summary, i)