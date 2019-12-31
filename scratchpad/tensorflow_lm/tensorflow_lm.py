import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
import sklearn.linear_model

# preparing data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# tensorflow way
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

np.set_printoptions(suppress=True)
print(theta_value.ravel())

# scikit-learn way
ls = sklearn.linear_model.LinearRegression()
ls.fit(housing_data_plus_bias, housing.target)
print(ls.intercept_)
print(ls.coef_)
