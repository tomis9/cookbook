---
title: "learning tensorflow"
date: 2019-08-05T18:45:18+02:00
draft: false
categories: ["Python", "Machine learning"]
---






The best way to gain intuiton to any new thing you learn is to start from a very beginning and play with it (*let's see what happens if I do this*). That's the power of reinforcement learning ;)

These packages will be useful in the nearest future:


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # ignore warnings
```

A trivial example of tensorflow:


```python
import tensorflow as tf
a = tf.Variable(10, name='a')  # a variable
b = tf.Variable(12, name='b')  # another variable
s = a + b  # a tensor
sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
print(sess.run(s))
```

```
## 22
```

As you can see, one does not simply add two numbers in tensorflow.

* lesson #1: you cannot initialize a tensor


```python
sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
# print(s.eval())  # does not work - you eval() is not connected to the session
# anyhow
print(sess.run(s))
```

```
## 22
```

* lesson #2: eval does not recognize session by itself


```python
with tf.Session() as sess:
    sess.run(a.initializer)
    sess.run(b.initializer)
    print(s.eval())  # does work - eval recognizes a default session
```

```
## 22
```

* lesson #2: eval works in `with` clause

* lesson #3: so far there are 2 ways to initialize a variable in a session


```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(s.eval())
```

```
## 22
```

* lesson #4: the third and most compact way to initialize variables - all the variables in one statement


```python
s1 = tf.add(a, b)  # a tensor, not variable
with tf.Session() as sess:
    init.run()
    print(s1.eval())
```

```
## 22
```

* lesson #5: tensorflow has it's own mathematical functions


```python
c = tf.Variable(np.array([[1, 2], [3, 4]]), name='c')
d = tf.Variable(np.array([[5, 6], [7, 8]]), name='d')
m = tf.matmul(c, d)  # a tensor again
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(m.eval())
```

```
## [[19 22]
##  [43 50]]
```

* lesson #6: tensorflow can interpret numpy arrays as matrices
* lesson #7: you can multiply matrices!


```python
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
```

```
## [[ 1.8450608 ]
##  [ 0.65486424]
##  [ 0.71106291]
##  [-0.56256786]]
```

```python
lr = LinearRegression()
lr.fit(data[:, 1:], data[:, 0])
print(np.concatenate((np.array([lr.intercept_]), lr.coef_)))
```

```
## [ 1.8450608   0.65486424  0.71106291 -0.56256786]
```

* lesson #8: when creating a tensorflow vector, ou have to provide information if it's horizontal or mathematical, just like in mathematics lesson #9: tensorflow gives the same results as sklearn (linear regression)

the code above looks quite like a mess, let's clear it up


```python
def get_data(tensorflow=True):
    iris = load_iris()
    data = iris.data
    y = data[:, 0].reshape(150, 1)
    x0 = np.ones(150).reshape(150, 1)
    X = np.concatenate((x0, data[:, 1:]), axis=1)
    if tensorflow:
        y = tf.constant(y, name='y')
        X = tf.constant(X, name='X')  # constant is a tensor
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
```

```
## [[ 1.8450608 ]
##  [ 0.65486424]
##  [ 0.71106291]
##  [-0.56256786]]
## 0.09589065804790765
```

* lesson #10: you can easily divide your code into modules to make it easier to read lesson #11: when dealing with input data, you can use tf.constant instead of tf.Variable, as the data never changes; constant is a tensor


```python
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
```

```
## [[0.90284599]
##  [0.52665587]
##  [0.29606479]
##  [0.09843398]]
## [[0.93436522]
##  [0.62244743]
##  [0.42040273]
##  [0.13826967]]
## 4.579071403794317
## 1.7091145550832831
```

* lesson #12: you can calculate the gradient of mse pretty simply on a piece of paper


```python
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
```

```
## 1.419054716306455
## 1.0035156867331174
## 0.8832634721505079
## 0.823266122045173
## 0.7775400386445852
## 0.7369497455263202
## 0.6995516293561233
## 0.6648147384562303
## 0.6324939485390587
## 0.602409670498605
## 0.5744042438125687
## 0.5483328036457722
## 0.5240609091747175
## 0.5014635304425142
## 0.48042433820197533
## 0.46083508842035475
## 0.4425950583791847
## 0.42561052367460916
## 0.40979427189114154
## 0.3950651501308448
## 0.3813476439930582
## 0.36857148580923116
## 0.35667129009820786
## 0.34558621435029435
## 0.3352596433801756
## 0.32563889561109777
## 0.3166749497665225
## 0.30832219055133336
## 0.3005381720031944
## 0.2932833972863343
## 0.28652111378532796
## 0.2802171224358309
## 0.27433960030307364
## 0.2688589354876624
## 0.2637475735021766
## 0.25897987432157055
## 0.2545319793657601
## 0.25038168772430186
## 0.2465083409810217
## 0.24289271604106594
## 0.23951692540436384
## 0.23636432436812072
## 0.23341942467691457
## 0.23066781417240934
## 0.22809608202583354
## 0.22569174916533058
## 0.22344320353723873
## 0.221339639865438
## 0.21937100359623798
## 0.21752793873799295
## 0.2158017393248366
## 0.21418430425273258
## 0.21266809525353092
## 0.2112460977890007
## 0.20991178466195762
## 0.2086590821557024
## 0.20748233852610176
## 0.2063762946828499
## 0.20533605690780501
## 0.2043570714688656
## 0.20343510099768114
## 0.2025662025086488
## 0.20174670694515576
## 0.20097320014695638
## 0.20024250513994138
## 0.19955166565641982
## 0.19889793080041684
## 0.19827874077843227
## 0.19769171362162932
## 0.19713463283057114
## 0.196605435878404
## 0.1961022035128438
## 0.19562314980146353
## 0.19516661286863696
## 0.194731046276083
## 0.19431501100229254
## 0.19391716797922648
## 0.1935362711475676
## 0.19317116099449455
## 0.19282075854045314
## 0.19248405974373076
## 0.1921601302937995
## 0.19184810076642214
## 0.19154716211538156
## 0.19125656147744613
## 0.19097559826880803
## 0.19070362055274126
## 0.19044002165963747
## 0.19018423704188048
## 0.1899357413472474
## 0.1896940456956486
## 0.1894586951450824
## 0.18922926633365483
## 0.18900536528543257
## 0.18878662536874632
## 0.1885727053963521
## 0.18836328785759293
## 0.18815807727339062
## 0.18795679866553455
## 0.18775919613232256
## [[1.16278366]
##  [0.94787487]
##  [0.25485803]
##  [0.67099061]]
```

* lesson #13: tf.assign - assign one value to another, training tensor is only technical, so we could point that the assignment should be made in every iteration

let's clear the code a little bit



```python
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
```

```
## 0.34121015958816003
## 0.19938206278933268
## 0.1841637663910814
## 0.1712793271459909
## 0.16035963722500987
## 0.15110182437076303
## 0.143249831882852
## 0.13658717201591622
## 0.1309308070240725
## 0.12612597660366254
## 0.12207947792685436
## [[ 1.00911013]
##  [ 0.9308095 ]
##  [ 0.55361139]
##  [-0.08539095]]
```

we can make this even better if we use different starting values in each run


```python
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
```

```
## 0.9580391110620229
## 0.14556241356270813
## 0.12951894057596267
## 0.11894525969951293
## 0.11175082038452416
## 0.106811700951172
## 0.10341261176805527
## 0.10107181773742359
## 0.09945953318434024
## 0.0983489751813528
## 0.0975846328442456
## [[ 1.44547459]
##  [ 0.7545403 ]
##  [ 0.76593601]
##  [-0.65833953]]
```

* lesson #14: it is better to use tensorflow starting values, as they change in every run
* lesson #15: you can calculate gradient manually by yourself, but you can use numerical algorithms implemented in tf.gradients


```python
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
```

```
## 5.481628842400672
## 0.1343093854477265
## 0.12015697763073525
## 0.11219644817611368
## 0.10704617792185021
## 0.10356062058250871
## 0.10117125356721461
## 0.09952755808889892
## 0.09839574495334209
## 0.09761620131211479
## 0.0970796911517879
## [[ 1.5102637 ]
##  [ 0.73838778]
##  [ 0.75700249]
##  [-0.64272054]]
```

* lesson #16: an optimizer knows, that it can change variables, not constants
* lesson #17: operations (like optimizer) are run, not evaluated


you should always use get_variable() insetad of Variable (interesting) [link to stackoverflow discussion](https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow)


```python
learning_rate = 0.01
n_iter = 1000
X_train, y_train = get_data(tensorflow=False)
X = tf.placeholder("float64", shape=(None, 4))  # placeholder -
y = tf.placeholder("float64", shape=(None, 1))
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(mse)
batch_indexes = np.arange(150).reshape(5, 30)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        for batch_index in batch_indexes:
            _training.run(feed_dict={X: X_train[batch_index],
                                     y: y_train[batch_index]})
        if not i % 100:
            print(mse.eval(feed_dict={X: X_train, y: y_train}))
    print(mse.eval(feed_dict={X: X_train, y: y_train}), "- final score")
    print(beta.eval())
```

```
## 2.8740067856065066
## 0.12748201332419298
## 0.11517319774791858
## 0.10894834299993382
## 0.10548239203953817
## 0.10333717632930743
## 0.1018753263894252
## 0.10080306141061039
## 0.09997667856161177
## 0.09931994344899633
## 0.0987931347982303 - final score
## [[ 1.32240838]
##  [ 0.78822447]
##  [ 0.76283509]
##  [-0.63327425]]
```

* lesson #18: in mini-batch processing it is comfortable to use placeholders


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
x = tf.placeholder(tf.float64, shape=(4, None))
y = tf.placeholder(tf.float64, shape=(3, None))
W = tf.Variable(tf.random_uniform([3, 4], -1, 1, dtype="float64"))
b = tf.Variable(tf.random_uniform([3, 1], -1, 1, dtype="float64"))
mult = tf.matmul(W, x) + b  # broadcasting just like in numpy
y_hat = tf.nn.softmax(mult, axis=0)
error = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(error)
init = tf.global_variables_initializer()
batches = np.arange(150).reshape(5, 30)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch].transpose(),
                                     y: y_all[batch].transpose()})
        if not i % 100:
            print(error.eval(feed_dict={x: X_np.transpose(),
                                        y: y_all.transpose()}))
    preds = y_hat \
        .eval(feed_dict={x: X_np.transpose()}) \
        .transpose()
```

```
## 0.412990124666561
## 0.12633307216808745
## 0.10555463098243682
## 0.09415511726165647
## 0.0858551346838089
## 0.07920889139325946
## 0.07366969861802858
## 0.06895702149814582
## 0.06489536175236055
## 0.061361279604265025
```

```python
def calculate_accuracy(preds):
    preds_max = np.amax(preds, axis=1)
    max_indexes = []
    for pred, pred_max in zip(preds, preds_max):
        prediction = np.where(pred == pred_max)[0][0]
        max_indexes.append(prediction)
    preds_cat = np.array(max_indexes)
    return(accuracy_score(y_np, preds_cat))
calculate_accuracy(preds)  # maybe overfitting?
```

* lesson #19: tf.reshape is NOT the same as tf.transpose
* lesson #20: tf.nn.softmax works on rows, not columns. Oh, that's nice. You can provide "axis" parameter in this function
* lesson #21: in tensorflow you will find broadcasting, just like in numpy

* name_scope


```python
with tf.name_scope("constants"):
    a = tf.constant(10, name='a')
    b = tf.constant(12, name='b')
```

* and variable scope


```python
with tf.variable_scope("variables"):
    c = tf.constant(20, name='c')
    d = tf.constant(22, name='d')
```


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')
W0 = tf.Variable(tf.random_uniform([4, 3], -1, 1, dtype=tf.float64), name='W0')
b0 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64), name='b0')  # will broadcast
h = tf.nn.softmax(tf.matmul(x, W0) + b0)
W1 = tf.Variable(tf.random_uniform([3, 3], -1, 1, dtype=tf.float64), name='W1')
b1 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64), name='b1')
y_hat = tf.nn.softmax(tf.matmul(h, W1) + b1)
error = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(error)
init = tf.global_variables_initializer()
batches = np.arange(150).reshape(5, 30)
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})
```

```
## 0.19240954276556932
## 0.1164575111876838
## 0.09940291036882618
## 0.08438836443511233
## 0.07264154150247036
## 0.06371230109907892
## 0.05678858264752773
## 0.0513196494744813
## 0.04693915133761486
## 0.043385366237396736
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.9666666666666667
```

* lesson #21: deeper neaural networks converge much more slowly


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')
with tf.variable_scope('layer1'):
    W0 = tf.Variable(tf.random_uniform([4, 3], -1, 1, dtype=tf.float64))
    b0 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64))
    h = tf.nn.softmax(tf.matmul(x, W0) + b0)
with tf.variable_scope('layer2'):
    W1 = tf.Variable(tf.random_uniform([3, 3], -1, 1, dtype=tf.float64))
    b1 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64))
    y_hat = tf.nn.softmax(tf.matmul(h, W1) + b1)
with tf.variable_scope('training'):
    error = tf.reduce_mean(tf.square(y - y_hat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    _training = optimizer.minimize(error)
init = tf.global_variables_initializer()
batches = np.arange(150).reshape(5, 30)
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})
```

```
## 0.2872064311644101
## 0.2220759655777477
## 0.22150162574416218
## 0.14628311587003895
## 0.11901825369489172
## 0.11384320273667424
## 0.10538323113021801
## 0.09066078537163652
## 0.0776566290507637
## 0.06775736527191677
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.9466666666666667
```


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')
def neural_layer(scope_name, x, input_size, output_size, func):
    with tf.variable_scope(scope_name):
        W_shape = [input_size, output_size]
        b_shape = [1, output_size]
        W = tf.Variable(tf.random_uniform(W_shape, -1, 1, dtype=tf.float64))
        b = tf.Variable(tf.random_uniform(b_shape, -1, 1, dtype=tf.float64))
        z = func(tf.matmul(x, W) + b)
    return z
h = neural_layer('layer1', x, 4, 3, tf.nn.relu)
y_hat = neural_layer('layer1', h, 3, 3, tf.nn.softmax)
with tf.variable_scope('training'):
    error = tf.reduce_mean(tf.square(y - y_hat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    _training = optimizer.minimize(error)
init = tf.global_variables_initializer()
batches = np.arange(150).reshape(5, 30)
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})
```

```
## 0.36149517248519397
## 0.21386852992818475
## 0.2093263489888735
## 0.03761171624792657
## 0.020039821155515922
## 0.0157687574089451
## 0.013835930870713381
## 0.012755086965365797
## 0.012073798291109142
## 0.011606316662469581
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.98
```
