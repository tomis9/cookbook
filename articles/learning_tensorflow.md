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
## [[0.83349441]
##  [0.37190224]
##  [0.35786372]
##  [0.69669721]]
## [[0.85890218]
##  [0.45081084]
##  [0.44406612]
##  [0.72254982]]
## 3.157298072233297
## 1.5726878912265576
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
## 0.6840303198185235
## 0.578636364775043
## 0.5391448903835383
## 0.5138014030188889
## 0.49243102189606514
## 0.4729667260809907
## 0.4549271361411519
## 0.43814528920141244
## 0.42252056842591385
## 0.4079700146722525
## 0.3944185111654365
## 0.381796565194568
## 0.37003959129610353
## 0.35908750676681217
## 0.3488844064321336
## 0.3393782699783111
## 0.3305206915611174
## 0.32226662857982497
## 0.3145741679801695
## 0.3074043088090216
## 0.300720759880031
## 0.29448975149807227
## 0.2886798602652423
## 0.2832618460594168
## 0.27820850033959243
## 0.2734945049910229
## 0.26909630097784254
## 0.2649919661217476
## 0.2611611013726538
## 0.25758472498130675
## 0.25424517402481234
## 0.2511260127742059
## 0.24821194742867247
## 0.2454887467740611
## 0.24294316835407032
## 0.24056288977108475
## 0.23833644476025165
## 0.23625316370515023
## 0.23430311828645253
## 0.23247706997641363
## 0.2307664221119798
## 0.22916317529787403
## 0.22765988590828853
## 0.22624962747189453
## 0.22492595473983187
## 0.22368287025026803
## 0.22251479321606182
## 0.22141653057412236
## 0.2203832500462674
## 0.21941045507182266
## 0.21849396148191114
## 0.2176298757944214
## 0.21681457501704846
## 0.2160446878536247
## 0.21531707721624183
## 0.21462882395243602
## 0.21397721170301356
## 0.21335971281195956
## 0.21277397521533023
## 0.21221781024110994
## 0.21168918125673689
## 0.2111861931054041
## 0.21070708227632778
## 0.21025020775798944
## 0.20981404252689753
## 0.2093971656277123
## 0.20899825480364703
## 0.20861607963890957
## 0.20824949517761093
## 0.20789743598603305
## 0.20755891062745252
## 0.20723299652085497
## 0.20691883515686793
## 0.20661562764609145
## 0.2063226305767331
## 0.2060391521600549
## 0.20576454864363586
## 0.20549822097384296
## 0.2052396116901932
## 0.20498820203549786
## 0.20474350926679358
## 0.20450508415311097
## 0.2042725086471004
## 0.20404539371843125
## 0.20382337733772965
## 0.20360612260059138
## 0.20339331598193958
## 0.2031846657116699
## 0.20297990026315624
## 0.20277876694677538
## 0.20258103060115437
## 0.20238647237535
## 0.20219488859564327
## 0.2020060897110679
## 0.20181989931220598
## 0.20163615321815562
## 0.20145469862693813
## 0.201275393324934
## 0.20109810495124794
## 0.20092271031318606
## [[0.23682669]
##  [1.18158289]
##  [0.38753845]
##  [0.42654925]]
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
## 1.6292667293165621
## 0.16946482216741962
## 0.15744304185502916
## 0.14806321524735594
## 0.14012961883621847
## 0.13341816089336023
## 0.12773999050163484
## 0.12293546079205972
## 0.11886961165103749
## 0.11542835156205636
## 0.11254202582426671
## [[ 1.41748784]
##  [ 0.81945282]
##  [ 0.53584347]
##  [-0.08285912]]
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
## 4.565551669158054
## 0.11987725887888741
## 0.1032870725044554
## 0.09928516310491146
## 0.0979120791836044
## 0.09722403628273586
## 0.09679812243495771
## 0.09651369117428574
## 0.09631943559938695
## 0.09618593857866592
## 0.09609411652416278
## [[ 1.70650952]
##  [ 0.68945514]
##  [ 0.72997021]
##  [-0.59547869]]
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
## 2.2523536361361494
## 0.12409843434500042
## 0.11514114386032116
## 0.10911735273524585
## 0.10499526710287195
## 0.10216095623143268
## 0.10020956443499626
## 0.09886557788901951
## 0.09793984244916866
## 0.09730218110364383
## 0.09686330922841886
## [[ 1.54227894]
##  [ 0.73039095]
##  [ 0.75264915]
##  [-0.6351546 ]]
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
## 0.4734918262326216
## 0.1252768998630438
## 0.11793149490751005
## 0.1131915915827518
## 0.10984605999131562
## 0.10732851789978255
## 0.10535525574801488
## 0.10377063006396646
## 0.10248006893196229
## 0.10142033921628946
## 0.10055379117729907 - final score
## [[ 1.17851064]
##  [ 0.82384774]
##  [ 0.78472946]
##  [-0.67490088]]
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
## 0.4354274214255428
## 0.21912611416046948
## 0.21788887121481754
## 0.21721579607504315
## 0.21671594738860955
## 0.21628559784736992
## 0.21589051114604357
## 0.21551820007493
## 0.21516397311132826
## 0.21482606894967682
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
## 0.2507956878250209
## 0.12293241924853508
## 0.10005380246274236
## 0.0841783861923155
## 0.07196967061423154
## 0.06285626874111537
## 0.05588797307256066
## 0.05043710496292351
## 0.046100839067329015
## 0.04259976671650775
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
## 0.2566376349318794
## 0.2192288481105225
## 0.12653926070708935
## 0.11089130147234345
## 0.09536170530110043
## 0.08145877481131746
## 0.07086276164712739
## 0.06259573470144443
## 0.055963906105674824
## 0.05057102683385096
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.96
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
## 0.24784709449154785
## 0.11258385704639647
## 0.0805156227847725
## 0.05860355282841195
## 0.04555542083350024
## 0.037775565076625835
## 0.0326953352550764
## 0.029109345626393387
## 0.026433819813278456
## 0.02435867156150821
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.98
```
