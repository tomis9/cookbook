#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris

data = load_iris()
X_train, y_train = data.data, data.target

np.set_printoptions(suppress=True)

n_epochs = 10
learning_rate = 0.01
batch_size = 100

n, k = X_train.shape

y = tf.placeholder(tf.float32, shape=(1, 1), name="y")
X = tf.placeholder(tf.float32, shape=(None, n), name="X")

W0 = tf.Variable(tf.random_uniform([3, 4], -1, 1, dtype=tf.float64), name="W0")
h = tf.matmul(X_train, W0)


# h = tf.nn.sigmoid()

# theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")

# 5
# with tf.name_scope("loss") as scope:
#     error = y_pred - y
#     mse = tf.reduce_mean(tf.square(error), name="mse")
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
#
# # 6
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# # 7
# mse_summary = tf.summary.scalar('MSE', mse)
#
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = "tf_logs"
# logdir = "{}/run-{}/".format(root_logdir, now)
# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# 8
# logger.info("starting execution")
# with tf.Session() as sess:
#     # 6
#     # saver.restore(sess, "/tmp/my_model_final.ckpt")
#
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             # 9
#             X_batch, y_batch = X_split[batch_index], y_split[batch_index]
#
#             if batch_index % 10 == 0:
#                 summary_str = mse_summary.eval(feed_dict={X: X_batch,
#                                                           y: y_batch})
#                 step = epoch * n_batches + batch_index
#
#             # 10
#             file_writer.add_summary(summary_str, step)
#
#             # 11
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#
#         logger.info("Epoch {}, MSE = {}"
#                     .format(epoch, mse.eval(feed_dict={X: X_batch,
#                                                        y: y_batch})))
#
#     # 12
#     best_theta = theta.eval()

#     # 13
#     save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
#     file_writer.close()
#
# logger.info("execution ended")

# 14
# pd_comp = functions.compare_scores(X_np, y_np, best_theta.ravel())
# logger.info(pd_comp)
