import tensorflow as tf
# import numpy as np

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


# def neuron_layer(X, n_neurons, name, activation=None):
#     with tf.name_scope(name):
#         n_inputs = int(X.get_shape()[1])
#         stddev = 2 / np.sqrt(n_inputs + n_neurons)
#         init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
#         W = tf.Variable(init, name="kernel")
#         b = tf.Variable(tf.zeros([n_neurons]), name="bias")
#         Z = tf.matmul(X, W) + b
#         if activation is not None:
#             return activation(Z)
#         else:
#             return Z
#
#
# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
#                            activation=tf.nn.relu)
#     hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden1",
#                            activation=tf.nn.relu)
#     logits = neuron_layer(hidden2, n_outputs, name="outputs")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden1",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
