##########
#code to solve fish problem
##########

import numpy as np
import tensorflow as tf

from read_data import load_saved_normalised_train_data,load_saved_normalised_test_data
from model import create_model
from helper_code import create_random_batch,create_validation_set,save_model

def train(model_folder):

    X_train,y_train,index_train = load_saved_normalised_train_data(saved=True)
    #X_train,y_train = create_random_batch(X_train,y_train,5000)
    X_test, index_test = load_saved_normalised_test_data(saved=True)

    #create valid set
    X_train, y_train, X_valid, y_valid = create_validation_set(X_train,y_train,valid_pc=0.5)

    # define placeholders for tf model
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 8])
    retained_pc = tf.placeholder(tf.float32)

    scores = create_model(x, retained_pc=0.5)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
    train_step = tf.train.MomentumOptimizer(learning_rate=1e-2,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
    #SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    true_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    #calculate logloss
    logloss = tf.reduce_mean(cross_entropy)


    batch_size = 32
    epochs = 10

    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print('----- Epoch', epoch, '-----')
            n_iter = X_train.shape[0] // batch_size
            total_accuracy = 0
            for i in range(n_iter):
                X_batch, y_batch = create_random_batch(X_train, y_train, batch_size)
                train_step.run(feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})
                current_accuracy = accuracy.eval(feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})
                total_accuracy += current_accuracy

            print(' Train Accuracy:', total_accuracy / n_iter)

            # calc train loss
            train_logloss = logloss.eval(feed_dict={x: X_train, y_true: y_train, retained_pc: 1.0})
            print(' Train Logloss:', train_logloss)

            # calc valid loss
            test_logloss = logloss.eval(feed_dict={x: X_valid, y_true: y_valid, retained_pc:1.0})
            print(' Test Logloss:', test_logloss)

        #save the model with checkpoint
        save_model(sess, model_folder)

        #evaluate the model
        





#testing the model
def evaluate_model(model_folder):

    X_test, id_test = load_saved_normalised_test_data()

    with tf.Session() as sess:
        #load model
        saver = tf.train.Saver()
        saver.restore(sess, model_folder + 'model.checkpoint')

        #run test set evaluation
        print('Running Test evaluation')

        # calc test prediction
        test_scores = scores.eval()
        test_prediction = tf.nn.softmax(scores)
        test_prediction = tf.argmax(scores, 1).eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})



if __name__ == "__main__":

    model_folder = 'saved_model/'

    train(model_folder)

    #TODO: work out how to apply to test set + output result
    #TODO: set up visualation via tensorboard
    #TODO: introduce seed


