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
## [[0.67110767]
##  [0.90575761]
##  [0.96308186]
##  [0.82049505]]
## [[0.63814725]
##  [0.80803302]
##  [0.80199153]
##  [0.76455731]]
## 7.015069415185945
## 2.823271583728836
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
## 0.9114453873114652
## 0.8179542310812994
## 0.7675596139495855
## 0.727786607237632
## 0.6921531165990021
## 0.6592523404926055
## 0.6286763606119184
## 0.6002211825496451
## 0.5737311965376358
## 0.5490682704198361
## 0.5261051682681498
## 0.5047237475731099
## 0.48481412367458404
## 0.46627405630315477
## 0.4490084107099929
## 0.4329286624658326
## 0.4179524378802172
## 0.40400308658900974
## 0.3910092838847362
## 0.3789046606861218
## 0.36762745921921247
## 0.357120212622225
## 0.34732944681162303
## 0.33820540306264757
## 0.32970177986504323
## 0.32177549271473643
## 0.3143864505952621
## 0.3074973479893356
## 0.30107347134152784
## 0.29508251896797627
## 0.2894944334788269
## 0.2842812458440206
## 0.2794169302934391
## 0.2748772692986342
## 0.27063972793567276
## 0.2666833369772896
## 0.26298858410783665
## 0.25953731269665375
## 0.25631262760469514
## 0.2532988075357467
## 0.2504812234775038
## 0.24784626280939434
## 0.24538125868341126
## 0.2430744243115899
## 0.24091479181921077
## 0.23889215534650338
## 0.23699701810366108
## 0.2352205431044906
## 0.23355450732310382
## 0.2319912590358168
## 0.23052367812694868
## 0.22914513915258553
## 0.22784947697068522
## 0.2266309547592149
## 0.22548423425639824
## 0.2244043480686796
## 0.22338667390273925
## 0.2224269105878781
## 0.22152105576437278
## 0.22066538512205247
## 0.2198564330813854
## 0.21909097481684991
## 0.21836600952932708
## 0.21767874488073322
## 0.21702658251013873
## 0.21640710455623208
## 0.21581806111620683
## 0.2152573585760094
## 0.21472304875140452
## 0.2142133187835237
## 0.2137264817364735
## 0.21326096784822482
## 0.21281531638939388
## 0.212388168087677
## 0.21197825807863815
## 0.21158440934627823
## 0.21120552661935527
## 0.21084059069179062
## 0.2104886531376944
## 0.21014883139359347
## 0.20982030418234673
## 0.20950230725500932
## 0.20919412942855237
## 0.20889510889888335
## 0.20860462981003963
## 0.2083221190617544
## 0.20804704333883464
## 0.20777890634693869
## 0.20751724624041215
## 0.20726163322883961
## 0.20701166734989243
## 0.20676697639692104
## 0.20652721399053783
## 0.20629205778418755
## 0.20606120779439452
## 0.20583438484702538
## 0.20561132913150554
## 0.20539179885548894
## 0.2051755689930026
## 0.20496243011956833
## [[0.78201863]
##  [1.05422933]
##  [0.26637932]
##  [0.67845271]]
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
## 2.0849093943812873
## 0.1606489243483661
## 0.15001444426854282
## 0.14181947778576845
## 0.13488617261401223
## 0.1290190229414255
## 0.12405335057183287
## 0.11984994078667299
## 0.11629110367474489
## 0.11327734510393396
## 0.11074804195552614
## [[ 1.39199981]
##  [ 0.82197163]
##  [ 0.55484665]
##  [-0.12754992]]
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
## 1.3944808864321094
## 0.12508442317132273
## 0.10943779390686362
## 0.1039996087967157
## 0.10124849567365374
## 0.09953881708983342
## 0.09839568509413402
## 0.09761470397353998
## 0.0970779452955403
## 0.09670843714950987
## 0.09645416323805676
## [[ 1.61454178]
##  [ 0.71238951]
##  [ 0.74262716]
##  [-0.6175897 ]]
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
## 6.564837133110268
## 0.12081736013500162
## 0.10115799362923618
## 0.09730128053664185
## 0.09644915164234558
## 0.09619838109199859
## 0.09608828224650859
## 0.09602411388139055
## 0.09598208757626543
## 0.09595354401206715
## 0.0959339741894007
## [[ 1.78108326]
##  [ 0.67085898]
##  [ 0.71970463]
##  [-0.57754347]]
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
## 2.535222351106631
## 0.12846362123982338
## 0.1112875929692529
## 0.10374404573962583
## 0.10032108430819742
## 0.09868819993937093
## 0.09785220169712251
## 0.09738453084679985
## 0.09709665511561415
## 0.09690315796465317
## 0.09676484496155102 - final score
## [[ 1.58220849]
##  [ 0.7233463 ]
##  [ 0.72544679]
##  [-0.56314812]]
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
## 0.45392196786456535
## 0.23028836638825181
## 0.22345761454866808
## 0.22163356967305903
## 0.22076498447036816
## 0.22016801572134176
## 0.2196033234624687
## 0.21876368038712377
## 0.213483372634145
## 0.13625407891794145
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
## 0.2964802369201862
## 0.2220347731897072
## 0.22156484306327323
## 0.1344317901799443
## 0.11731894216400554
## 0.11032577176706294
## 0.09617048894193216
## 0.08239093439339572
## 0.07191907065020767
## 0.0636935050673083
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.94
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
## 0.2859951410059936
## 0.22257812560965745
## 0.2223745396506137
## 0.2223189347618648
## 0.22229302140089696
## 0.2222780467605852
## 0.22226829724604202
## 0.2222614466411118
## 0.22225637052110314
## 0.22225245914119146
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.3333333333333333
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
## 0.21003867797082468
## 0.027672732590951175
## 0.01835151415705064
## 0.015576771711975928
## 0.014169411926178896
## 0.013289171738431785
## 0.01267072726690756
## 0.012202911878898983
## 0.011831112236878907
## 0.011525282991871144
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.98
```
