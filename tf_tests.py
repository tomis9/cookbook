# the best way to gain intuiton to a any new thing you learn is to start from a
# very beginning and play with it (*let's see what happens if I do this*). The
# power of reinforcement learning ;)

# a trivial tensorflow example and what we are doing

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

a = tf.Variable(10, name='a')  # a variable
b = tf.Variable(12, name='b')  # another variable

s = a + b  # a tensor

sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
sess.run(s)

# lesson #1: you cannot initialize a tensor

sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
# print(s.eval())  # does not work - you eval() is not connected to the session
# anyhow
sess.run(s)

# lesson #2: eval does not recognize session by itself

with tf.Session() as sess:
    sess.run(a.initializer)
    sess.run(b.initializer)
    print(s.eval())  # does work - eval recognizes a default session sess

# lesson #2: eval works in `with`

with tf.Session() as sess:
    a.initializer.run()
    b.initializer.run()
    print(s.eval())  # does work - eval recognizes a default session sess

# lesson #3: so far there are 2 ways to initialize a variable in a session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(s.eval())

# lesson #4: the third and most compact way to initialize variables

s1 = tf.add(a, b)  # a tensor, not variable
with tf.Session() as sess:
    init.run()
    print(s1.eval())

# lesson #5: tensorflow has it's own mathematical functions

c = tf.Variable(np.array([[1, 2], [3, 4]]), name='c')
d = tf.Variable(np.array([[5, 6], [7, 8]]), name='d')

m = tf.matmul(c, d)  # a tensor again

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(m.eval())

# lesson #6: tensorflow can interpret numpy arrays as matrices
# lesson #7: you can multiply matrices!

iris = load_iris()
data = iris.data
y = tf.Variable(data[:, 0].reshape(150, 1), name='y')
x0 = np.ones(150).reshape(150, 1)
x0_X = np.concatenate((x0, data[:, 1:]), axis=1)
X = tf.Variable(x0_X, name='X')

cov = tf.matmul(tf.transpose(X), X, name='cov')
inv_cov = tf.matrix_inverse(cov, name='inv_cov')
xy = tf.matmul(tf.transpose(X), y, name='xy')
beta = tf.matmul(inv_cov, xy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())


lr = LinearRegression()
lr.fit(data[:, 1:], data[:, 0])
np.concatenate((np.array([lr.intercept_]), lr.coef_))

# lesson #8: when creating a tensorflow vector, ou have to provide information
# if it's horizontal or mathematical, just like in mathematics
# lesson #9: tensorflow gives the same results as sklearn (linear regression)

# the code above looks quite like a mess, let's clear it up


def get_data():
    iris = load_iris()
    data = iris.data
    y = tf.constant(data[:, 0].reshape(150, 1), name='y')
    x0 = np.ones(150).reshape(150, 1)
    x0_X = np.concatenate((x0, data[:, 1:]), axis=1)
    X = tf.constant(x0_X, name='X')  # constant is a tensor
    return X, y


def construct_beta_graph(X, y):
    cov = tf.matmul(tf.transpose(X), X, name='cov')
    inv_cov = tf.matrix_inverse(cov, name='inv_cov')
    xy = tf.matmul(tf.transpose(X), y, name='xy')
    beta = tf.matmul(inv_cov, xy, name='beta')
    return beta


X, y = get_data()
beta = construct_beta_graph(X, y)
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())
    print(mse.eval())


# lesson #10: you can easily divide your code into modules to make it easier to
# read
# lesson #11: when dealing with input data, you can use tf.constant instead of
# tf.Variable, as the data never changes; constant is a tensor

X, y = get_data()
learning_rate = 0.0001

beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)
new_beta = beta - learning_rate * gradient

mse_old = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))
mse_new = tf.reduce_mean(tf.square(y - tf.matmul(X, new_beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())
    print(new_beta.eval())
    print(mse_old.eval())
    print(mse_new.eval())

# lesson #12: you can calculate the gradient of mse pretty simply on a piece of
# paper

X, y = get_data()
learning_rate = 0.01

beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)

_training = tf.assign(beta, beta - learning_rate * gradient)

mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(100):
        _training.eval()
        print(mse.eval())
    print(beta.eval())

# lesson #13: tf.assign - assign one value to another, _training tensor is
# only technical, so we could point that the assignment should be made in every
# iteration

# let's clear the code a little bit

learning_rate = 0.01
n_iter = 1000

X, y = get_data()
beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)

_training = tf.assign(beta, beta - learning_rate * gradient)

mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.eval()
        if not i % 100:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())


# we can make this even better if we use different starting values in each run
learning_rate = 0.01
n_iter = 10000

X, y = get_data()
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

# gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)
gradient = tf.gradients(mse, [beta])[0]

_training = tf.assign(beta, beta - learning_rate * gradient)  # a tensor

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.eval()
        if not i % 1000:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())

# lesson #14: it is better to use tensorflow starting values, as they change in
# every run
# lesson #15: you can calculate gradient manually by yourself, but you can use
# numerical algorithms implemented in tf.gradients

learning_rate = 0.01
n_iter = 10000

X, y = get_data()
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(mse)  # an operation - a new class of objects

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.run()  # operations are being run, not evaluated
        if not i % 1000:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())

# lesson #16: an optimizer knows, that it can change variables, not constants
# lesson #17: operations (like optimizer) are run, not evaluated


