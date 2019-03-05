import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import scale
import sklearn.linear_model


np.set_printoptions(suppress=True)
# preparing data
housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data = scale(housing.data, axis=0, with_mean=True,
                            with_std=True, copy=True)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 50
learning_rate = 0.01

# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    ran = range(batch_index*batch_size,(batch_index+1)*batch_size)
    try:
        X_batch = scaled_housing_data_plus_bias[ran]
        y_batch = housing.target.reshape(-1, 1)[ran]
    except IndexError:
        X_batch = scaled_housing_data_plus_bias[batch_index*batch_size:]
        y_batch = housing.target.reshape(-1, 1)[batch_index*batch_size:]
    return X_batch, y_batch


theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)  # to dosyć długo trwa
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        print("Epoch", epoch, "MSE =", mse.eval(feed_dict={X: X_batch, y: y_batch}))

    best_theta = theta.eval()


print(best_theta.ravel())

ls = sklearn.linear_model.LinearRegression()
ls.fit(scaled_housing_data_plus_bias, housing.target)
print(ls.intercept_)
print(ls.coef_)
