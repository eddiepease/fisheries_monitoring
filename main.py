##########
#code to solve fish problem
##########

import numpy as np
import tensorflow as tf

from read_data import load_saved_normalised_train_data,load_saved_normalised_test_data
from model import create_model
from helper_code import create_random_batch,create_validation_set,save_model,create_submission


def train(model_folder):

    X_train,y_train,id_train = load_saved_normalised_train_data(saved=True)
    #X_train,y_train = create_random_batch(X_train,y_train,5000)

    #create valid set
    X_train, y_train, X_valid, y_valid = create_validation_set(X_train,y_train,valid_pc=0.3)

    # define placeholders for tf model
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 8])
    retained_pc = tf.placeholder(tf.float32)

    scores,layer_weights,layer_bias = create_model(x, retained_pc)
    probs = tf.nn.softmax(scores)

    # #sorting out the weights
    # for i,weight in enumerate(layer_weights):
    #     tf.summary.tensor_summary(str(i),weight)
    #     # tf.summary.image([['%s_w%d%d' % (weight.name, i, j) for i in range(len(layer_weights))] for j in range(32)],
    #     #                  weight)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(learning_rate=5e-3,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(learning_rate=).minimize(cross_entropy)

    true_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    #calculate logloss
    logloss = tf.reduce_mean(cross_entropy)

    batch_size = 32
    epochs = 20

    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # #logging
        # summary_writer = tf.summary.FileWriter('tmp/logs', sess.graph)
        # merged = tf.summary.merge_all()

        for epoch in range(epochs):
            print('----- Epoch', epoch, '-----')
            n_iter = X_train.shape[0] // batch_size
            total_accuracy = 0

            for i in range(n_iter):
                X_batch, y_batch = create_random_batch(X_train, y_train, batch_size)

                _, current_accuracy = sess.run([train_step,accuracy],
                                               feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})

                # _, current_accuracy, summary_str, printed_weights = sess.run(
                #     [train_step, accuracy, merged, layer_weight_1],
                #     feed_dict={x: X_batch, y_true: y_batch, retained_pc: 0.5})

                total_accuracy += current_accuracy
                # summary_writer.add_summary(summary_str, i)


            print(' Train Accuracy:', total_accuracy / n_iter)

            # calc train loss
            train_logloss = logloss.eval(feed_dict={x: X_train, y_true: y_train, retained_pc: 1.0})
            print(' Train Logloss:', train_logloss)

            # calc valid loss
            test_logloss = logloss.eval(feed_dict={x: X_valid, y_true: y_valid, retained_pc:1.0})
            print(' Test Logloss:', test_logloss)


        #save the model with checkpoint
        save_model(sess, model_folder)

        #evaluate model
        evaluate_model(x,retained_pc,probs)


#TODO: work out how to have this as a standalone function
def evaluate_model(x,retained_pc,probs):
    X_test, id_test = load_saved_normalised_test_data(saved=True)
    test_probs = probs.eval(feed_dict={x: X_test, retained_pc: 1.0})
    create_submission(test_probs,id_test,'corrected_resize')



if __name__ == "__main__":

    model_folder = 'saved_model/'

    train(model_folder)

    #TODO: understand tensorboard better


