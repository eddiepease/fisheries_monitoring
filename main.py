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
    y_true = tf.placeholder(tf.float32, shape=[None, 6]) #TODO: change back to 8 afterwards
    retained_pc = tf.placeholder(tf.float32)

    scores,filter_summary = create_model(x, retained_pc)
    probs = tf.nn.softmax(scores)
    tf.add_to_collection('probs', probs)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
    train_step = tf.train.AdamOptimizer(learning_rate=2e-2).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(learning_rate=5e-3,momentum=0.9,use_nesterov=True).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(learning_rate=).minimize(cross_entropy)

    true_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))

    #calculate logloss
    logloss = tf.reduce_mean(cross_entropy)

    batch_size = 32
    epochs = 2

    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('/tmp/logs', sess.graph)
        #merged = tf.summary.merge_all()
        for epoch in range(epochs):
            print('----- Epoch', epoch, '-----')
            n_iter = X_train.shape[0] // batch_size
            total_accuracy = 0

            for i in range(n_iter):
                X_batch, y_batch = create_random_batch(X_train, y_train, batch_size)
                _, current_accuracy,summary_str = sess.run([train_step,accuracy,filter_summary],
                                                           feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})

                for weight in layer_weights:
                    tf.scalar_summary(
                        [['%s_w%d%d' % (weight.name, i, j) for i in xrange(len(layer_weights))] for j in xrange(5)],
                        weight)
  #               current_accuracy = accuracy.eval(feed_dict={x: X_batch, y_true: y_batch, retained_pc:0.5})
                total_accuracy += current_accuracy
                summary_writer.add_summary(summary_str, i)

            print(' Train Accuracy:', total_accuracy / n_iter)

            ###    "        for epoch in range(args.epochs):\n",
    # "\n",
    # "            print('----- Epoch', epoch, '-----')\n",
    # "            total_loss = 0\n",
    # "            for i in range(n // BATCH_SIZE):\n",
    # "\n",
    # "                inst_story = train_stories[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]\n",
    # "                inst_order = train_orders[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]\n",
    # "                feed_dict = {datum: inst_story, order: inst_order}\n",
    # "                _, current_loss = sess.run([optim_op, loss], feed_dict=feed_dict)\n",
    # "                total_loss += current_loss\n",
    # "\n",
    # "                if i % 20 == 0:\n",
    # "                    print('Batch: ' + str(i))\n",
    # "\n",
    # "            print(' Train loss:', total_loss / n)\n",
    # "\n",
    # "            train_feed_dict = {datum: train_stories, order: train_orders}\n",
    # "            train_predicted = sess.run(predict, feed_dict=train_feed_dict)\n",
    # "            train_accuracy = nn.calculate_accuracy(train_orders, train_predicted)\n",
    # "            print(' Train accuracy:', train_accuracy)\n",





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

    #TODO: set up visualation via tensorboard
    #TODO: introduce seed


