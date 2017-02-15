##########
#code to solve fish problem
##########

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from read_data import load_saved_normalised_train_data,load_saved_normalised_test_data
from model import create_model
from helper_code import create_random_batch,save_model,create_submission,merge_several_folds_mean

#TODO: make a big object
# class Model():
#     def __init__(self):
#


def run_cross_validation_create_models(model_folder,nfolds=10):
    # input image dimensions
    batch_size = 32
    nb_epoch = 8
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = load_saved_normalised_train_data(saved=True)

    yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_logloss = 0
    for train_index, test_index in kf.split(train_data):

        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        ####################################################
        #train tensorflow model
        ###################################################

        # define placeholders
        # define placeholders for tf model
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y_true = tf.placeholder(tf.float32, shape=[None, 8])
        retained_pc = tf.placeholder(tf.float32)

        # create model
        logits = create_model(x, retained_pc)
        probs = tf.nn.softmax(logits)

        #add collections for saved model
        tf.add_to_collection('x', x)
        tf.add_to_collection('retained_pc', retained_pc)
        tf.add_to_collection('probs', probs)

        # optimiser
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_true)
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        true_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
        # accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))
        logloss = tf.reduce_mean(cross_entropy)

        # init session
        # train the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # #logging
            # summary_writer = tf.summary.FileWriter('tmp/logs', sess.graph)
            # merged = tf.summary.merge_all()

            for epoch in range(nb_epoch):
                print('----- Epoch', epoch, '-----')
                n_iter = X_train.shape[0] // batch_size
                total_accuracy = 0

                for i in range(n_iter):
                    X_batch, Y_batch = create_random_batch(X_train, Y_train, batch_size)

                    sess.run([train_step], feed_dict={x: X_batch, y_true: Y_batch, retained_pc: 0.5})

                    # _, current_accuracy, summary_str, printed_weights = sess.run(
                    #     [train_step, accuracy, merged, layer_weight_1],
                    #     feed_dict={x: X_batch, y_true: y_batch, retained_pc: 0.5})

                    # summary_writer.add_summary(summary_str, i)

                # calc train loss
                train_logloss = logloss.eval(feed_dict={x: X_train, y_true: Y_train, retained_pc: 1.0})
                print(' Train Logloss:', train_logloss)

                # calc valid loss
                test_logloss = logloss.eval(feed_dict={x: X_valid, y_true: Y_valid, retained_pc: 1.0})
                print(' Test Logloss:', test_logloss)

                # valid_predictions
                test_probs = probs.eval(feed_dict={x: X_valid, retained_pc: 1.0})

            # save outputs
            save_model(sess, model_folder, num_fold)
            sum_logloss += test_logloss * len(test_index)

            for i in range(len(test_index)):
                yfull_train[test_index[i]] = test_probs[i]

        #############################################################

    score = sum_logloss/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = '_' + str(np.round(score,3)) + '_flds_' + str(nfolds) + '_eps_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string

#TODO: add info string back in
def run_cross_validation_process_test(model_folder,num_models):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = num_models

    for i in range(nfolds):
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = load_saved_normalised_test_data(saved=True)

        ##########
        #tensorflow section
        ##########
        with tf.Session() as sess:
            # load model
            model_name = model_folder + str(num_fold)
            new_saver = tf.train.import_meta_graph(model_name + '.meta')
            new_saver.restore(sess, model_name)
            x = tf.get_collection('x')[0]
            retained_pc = tf.get_collection('retained_pc')[0]
            probs = tf.get_collection('probs')

            # run test set evaluation
            print('Running Test evaluation')
            test_probs = probs.eval(feed_dict={x: test_data, retained_pc: 1.0})
            print(test_probs.shape)

        yfull_test.append(test_probs)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    # info_string = 'loss_' + info_string \
    #             + '_folds_' + str(nfolds)
    create_submission(test_res, test_id,info='full')



# def train(model_folder):
#
#     X_train,y_train,id_train = load_saved_normalised_train_data(saved=True)
#     #X_train,y_train = create_random_batch(X_train,y_train,5000)
#
#     #create valid set #TODO: improve this
#     X_train, y_train, X_valid, y_valid = create_validation_set(X_train,y_train,valid_pc=0.5)
#
#     # define placeholders for tf model
#     x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
#     y_true = tf.placeholder(tf.float32, shape=[None, 8])
#     retained_pc = tf.placeholder(tf.float32)
#
#     scores,layer_weights,layer_bias = create_model(x, retained_pc)
#     probs = tf.nn.softmax(scores)
#
#     # #sorting out the weights
#     # for i,weight in enumerate(layer_weights):
#     #     tf.summary.tensor_summary(str(i),weight)
#     #     # tf.summary.image([['%s_w%d%d' % (weight.name, i, j) for i in range(len(layer_weights))] for j in range(32)],
#     #     #                  weight)
#
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
#     train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
#     #train_step = tf.train.MomentumOptimizer(learning_rate=5e-3,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
#     #train_step = tf.train.AdagradOptimizer(learning_rate=).minimize(cross_entropy)
#
#     true_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
#     accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))
#
#     #calculate logloss
#     logloss = tf.reduce_mean(cross_entropy)
#
#     batch_size = 64
#     epochs = 10
#
#     # train the model
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         # #logging
#         # summary_writer = tf.summary.FileWriter('tmp/logs', sess.graph)
#         # merged = tf.summary.merge_all()
#
#         for epoch in range(epochs):
#             print('----- Epoch', epoch, '-----')
#             n_iter = X_train.shape[0] // batch_size
#             total_accuracy = 0
#
#             for i in range(n_iter):
#                 X_batch, y_batch = create_random_batch(X_train, y_train, batch_size)
#
#                 _, current_accuracy = sess.run([train_step,accuracy],
#                                                feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})
#
#                 # _, current_accuracy, summary_str, printed_weights = sess.run(
#                 #     [train_step, accuracy, merged, layer_weight_1],
#                 #     feed_dict={x: X_batch, y_true: y_batch, retained_pc: 0.5})
#
#                 total_accuracy += current_accuracy
#                 # summary_writer.add_summary(summary_str, i)
#
#
#             print(' Train Accuracy:', total_accuracy / n_iter)
#
#             # calc train loss
#             train_logloss = logloss.eval(feed_dict={x: X_train, y_true: y_train, retained_pc: 1.0})
#             print(' Train Logloss:', train_logloss)
#
#             # calc valid loss
#             test_logloss = logloss.eval(feed_dict={x: X_valid, y_true: y_valid, retained_pc:1.0})
#             print(' Test Logloss:', test_logloss)
#
#
#         #save the model with checkpoint
#         save_model(sess, model_folder)
#
#         #evaluate model
#         evaluate_model(x,retained_pc,probs)
#
#
# #TODO: work out how to have this as a standalone function
# def evaluate_model(x,retained_pc,probs):
#     X_test, id_test = load_saved_normalised_test_data(saved=True)
#     test_probs = probs.eval(feed_dict={x: X_test, retained_pc: 1.0})
#     create_submission(test_probs,id_test,'corrected_resize')



if __name__ == "__main__":

    model_folder = 'saved_model/'
    n_folds = 3
    #run_cross_validation_create_models(model_folder,n_folds)
    run_cross_validation_process_test(model_folder,num_models=n_folds)

    #TODO: understand tensorboard better


