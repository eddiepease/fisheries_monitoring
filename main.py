##########
#code to solve fish problem
##########

import numpy as np
import tensorflow as tf

from read_data import load_saved_normalised_train_data,load_saved_normalised_test_data
from model import create_model
from helper_code import create_random_batch,create_validation_set,save_model

def train(model_folder):

    X_train,y_train,_ = load_saved_normalised_train_data(saved=True)
    #X_train,y_train = create_random_batch(X_train,y_train,5000)
    # X_test,_ = read_and_normalize_test_data()

    #create valid set
    X_train, y_train, X_valid, y_valid = create_validation_set(X_train,y_train,valid_pc=0.3)

    # define placeholders for tf model
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 8])
    retained_pc = tf.placeholder(tf.float32)

    scores = create_model(x, retained_pc=0.5)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    true_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    #calculate logloss
    y_pred = tf.nn.softmax(scores)
    logloss = tf.reduce_mean(-tf.reduce_sum(tf.log(y_pred)))


    batch_size = 64
    epochs = 30

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

            # calc valid loss
            train_logloss = logloss.eval(feed_dict={x: X_train, y_true: y_train, retained_pc: 0.5})
            print(' Train Logloss:', train_logloss)

            # calc valid loss
            test_logloss = logloss.eval(feed_dict={x: X_valid, y_true: y_valid, retained_pc:0.5})
            print(' Test Logloss:', test_logloss)

        #save the model with checkpoint
        save_model(sess, model_folder)


# #testing the model
# def evaluate_model(model_folder):
#     with tf.Session() as sess:
#         #load model
#         saver = tf.train.Saver()
#         saver.restore(sess, model_folder + 'model.checkpoint')
#
#         #run test set evaluation
#         print('Running Test evaluation')
#
#         # calc test prediction
#         test_prediction = tf.argmax(scores, 1).eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
#
#         test_prediction = test_prediction.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
#         print(test_loss)








if __name__ == "__main__":

    model_folder = 'saved_model/'

    train(model_folder)

    #TODO: work out the discrepancy in loss
    #TODO: work out how to apply to test set
    #TODO: set up visualation via tensorboard


