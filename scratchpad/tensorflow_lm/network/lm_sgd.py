import tensorflow as tf
import numpy as np
import logging.config
import functions
import json

np.set_printoptions(suppress=True)

with open('./logging_config.json') as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger('base')

n_epochs = 50
learning_rate = 0.01
batch_size = 100

X_np, y_np = functions.get_data()
m, n = X_np.shape

X_split, y_split, n_batches = functions.split_data(X_np, y_np, batch_size, m)

X = tf.placeholder(tf.float32, shape=(None, n), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")


theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # saver.restore(sess, "/tmp/my_model_final.ckpt")
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = X_split[batch_index], y_split[batch_index]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        print("Epoch", epoch, "MSE =", mse.eval(feed_dict={X: X_batch,
                                                           y: y_batch}))

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

print(best_theta.ravel())

functions.lm_sklearn(X_np, y_np)
