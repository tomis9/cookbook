---
title: "tensorflow"
date: 2019-01-01T18:17:38+01:00
draft: false
categories: ["Machine learning", "Python"]
tags: ["tensorflow"]
---

## 1. What is tensorflow and why would you use it?

- tensorflow is a machine learning framework 

- which has APIs to Python, C++ and R

- and let's you evaluate any machine learning algorithm, especially deep learning:

    - quickly (all the computations are performed in C++)

    - easily - you can view your results in a GUI - Tensorboard

    - on a huge amount of data, as tensorflow scales easily to many machines and can even make use of GPU.


## 2. A simple example - linear regression

There are two additional files used in this example: [functions.py](functions.py) and [logging_config.json](logging_config.json). I will not go through any of them, as they do not use tensorflow at all.

We will be working on a well-known housing data available in sklearn.

```{python}
import tensorflow as tf
import numpy as np
import logging.config
import functions
import json
from datetime import datetime

np.set_printoptions(suppress=True)

# 1 - logging
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

with open('./logging_config.json') as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger('base')

# 2
logger.info("setting metaparameters")
n_epochs = 10
learning_rate = 0.01
batch_size = 100
logger.info("n_epochs: {}, learning_rate: {}, batch_size: {}"
            .format(n_epochs, learning_rate, batch_size))

# 3
logger.info("data preparation")
X_np, y_np = functions.get_data()
m, n = X_np.shape
X_split, y_split, n_batches = functions.split_data(X_np, y_np, batch_size, m)


# 4
logger.info("starting construction phase")
X = tf.placeholder(tf.float32, shape=(None, n), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

# 5
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# 6
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 7
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
logger.info("ending construction phase")

# 8
logger.info("starting execution")
with tf.Session() as sess:
    # 6
    # saver.restore(sess, "/tmp/my_model_final.ckpt")

    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # 9
            X_batch, y_batch = X_split[batch_index], y_split[batch_index]

            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch,
                                                          y: y_batch})
                step = epoch * n_batches + batch_index
                # 10
                file_writer.add_summary(summary_str, step)

            # 11
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        logger.info("Epoch {}, MSE = {}"
                    .format(epoch, mse.eval(feed_dict={X: X_batch,
                                                       y: y_batch})))

    # 12
    best_theta = theta.eval()

    # 13
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
    file_writer.close()

logger.info("execution ended")

# 14
pd_comp = functions.compare_scores(X_np, y_np, best_theta.ravel())
logger.info(pd_comp)
```

What happened here?

1. tensorflow logging is usefull if you want to see the steps of your calculations in Tensorboard. During every execution of your code, you should provide a different logging directory. Using timestamps is a comfortable way to achieve this.

2. Metaparameters for the algorithm. 10 epochs is enough for SGD to reach the minimum of Least Squares.

3. Preparing the dataset. I split it into batches right here, which is not the nicest solution, but sufficient for the purposes of this tutorial. (TODO)

4. This is where graph creation beging:

    - creating placeholders, which are empty slots for the data. In general, in tensorflow your create variables for parameter values and placeholders for the data. 

    - as you can see, creating computations, like matrix multipication, is rather straightforward, but remember to give your new variable a `name` in the tendorflow graph.

5. You can encapsulate several nodes into one scope, which will make the final graph easier to read. Here we also create an optimiser, which is Gradient Descent in this case, and minimisation target: mse.

6. This is where the graph definition ends. Now we prepare for the execution, first by initialising the variables (or, strictly speaking, creating an initialiser). We also create a saver, which saves the parameters of the model to a file. 

7. Here we declare what we want to export to our log file. Clearly we don't have to export anything nor even use a log file.

8. Execution phase starts with creating a session. All the computations are run within a session, so it is convenient to use the `with` clause, which will eventually close the session for us, even if an exception occurs.

9. In every epoch, for every batch we choose a subset of our dataset.

10. Every tenth batch produces a tensorflow log message, which we will be able to see plotted in Tensorboard.

11. This is where the cumputation is run, e.g. the gradient is calculated and the parameter values are updated. Notice that we have to `feed_dict` the values of the placeholders with the data.

12. Let's save the resulting parameters to a variable. So far this is our goal - to compare the results of tensorflow optimisation to sklearn regression.

13. We save our model. We don't have to do that, but it may come up useful in the future.

14. We compare the results with sklearn.linear_model (more on that [here](tomis9.githu.io/sklearn_regerssion)). They are almost identical.


## 3. Tensorboard

I mentioned several times that Tensorboard is a nice and usefull tool for diagnosing our algorithm.

Simply start it with

```
tensorboard --logdir tf_logs/  
```

I believe the interface is simple enough so it needs no tutorial :)

## 3. Useful links

- [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do) - definitely the best book on machine learning I've ever read, the second best is [Data mining and analysis](http://www.dataminingbook.info/pmwiki.php)

- [Tensorflow on Spark](https://github.com/yahoo/TensorFlowOnSpark) - an interesting combo

- [tensorflow for R](https://tensorflow.rstudio.com/)

- [example of tensorflow for R](https://www.r-bloggers.com/step-by-step-tutorial-deep-learning-with-tensorflow-in-r/)

- [good examples of tensorflow](https://github.com/aymericdamien/TensorFlow-Examples)

- [LSTM using tensorflow](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)

- [word2vec using tensorflow](https://www.tensorflow.org/tutorials/representation/word2vec)

*tensorflow 2.0 has just been released! https://www.youtube.com/watch?v=k5c-vg4rjBw*

