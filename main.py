##########
#code to solve fish problem
##########

import numpy as np
import tensorflow as tf

from read_data import read_and_normalize_train_data,read_and_normalize_test_data
from model import create_model

def train(scores,y_true):
    learning_rate = 0.0005
    xent = tf.nn.softmax_cross_entropy_with_logits(scores, y_true)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    false_prediction = tf.not_equal(tf.argmax(scores, 1), tf.argmax(y_true, 1))
    loss = tf.reduce_mean(tf.cast(false_prediction, tf.float32))

    batch_size = 64
    n_iter = 50001  # 20000
    print_freq = 2000  # print out every 1000
    # df_index = [i * print_freq for i in range(int(n_iter / print_freq))]
    # results = pd.DataFrame(0.0, index=df_index, columns=['train_loss', 'test_loss'])
    # model_folder = 'models/4_a/'
    # model_name = 'model.checkpoint'

    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iter in range(n_iter):
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={x: batch[0], y_true: batch[1], retained_pc: 0.5})

            if iter % print_freq == 0:  # print out every 1000
                print('----- Iteration', iter, '-----')

                # calc train loss
                train_loss = loss.eval(feed_dict={x: mnist.train.images, y_true: mnist.train.labels, retained_pc: 1.0})
                results.loc[iter, 'train_loss'] = round(train_loss, 3)
                print(' Train loss:', train_loss)

                # calc train loss
                test_loss = loss.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels, retained_pc: 1.0})
                results.loc[iter, 'test_loss'] = round(test_loss, 3)
                print(' Test loss:', test_loss)

                # calc test prediction
                test_prediction = tf.argmax(scores, 1).eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})


if __name__ == "__main__":
    #X_train,y_train,_ = read_and_normalize_train_data()
    #X_test,_ = read_and_normalize_test_data()

    #define placeholders for tf model
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
    y_true = tf.placeholder(tf.float32, shape=[None, 8])
    retained_pc = tf.placeholder(tf.float32)

    scores = create_model(x,retained_pc=0.5)



