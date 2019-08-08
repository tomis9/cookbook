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
## [[ 1.85599749]
##  [ 0.65083716]
##  [ 0.70913196]
##  [-0.55648266]]
```

```python
lr = LinearRegression()
lr.fit(data[:, 1:], data[:, 0])
print(np.concatenate((np.array([lr.intercept_]), lr.coef_)))
```

```
## [ 1.85599749  0.65083716  0.70913196 -0.55648266]
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
## [[ 1.85599749]
##  [ 0.65083716]
##  [ 0.70913196]
##  [-0.55648266]]
## 0.09630269942460723
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
## [[0.22183254]
##  [0.16161104]
##  [0.17320714]
##  [0.17617964]]
## [[0.28581041]
##  [0.35729306]
##  [0.42197357]
##  [0.2560269 ]]
## 18.436448399425043
## 6.767540775525738
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
## 3.268241880748393
## 1.9006667503220156
## 1.557243515210368
## 1.4185891059977678
## 1.3248103719437423
## 1.2444291049466343
## 1.1709912467390364
## 1.1029371971478714
## 1.039681399529468
## 0.9808476375721104
## 0.9261187853234132
## 0.8752062745328324
## 0.8278429580575826
## 0.78378061137267
## 0.7427884145799357
## 0.7046516974350612
## 0.6691708026700833
## 0.6361600347455492
## 0.6054466832705824
## 0.5768701149374688
## 0.5502809289982957
## 0.5255401718050177
## 0.502518606275951
## 0.4810960324465256
## 0.461160655531868
## 0.44260849817881315
## 0.42534285381730297
## 0.40927377823721967
## 0.39431761671766885
## 0.3803965642226368
## 0.36743825635079685
## 0.35537538888892106
## 0.3441453639687361
## 0.33368996096692954
## 0.32395503041809204
## 0.31489020933137407
## 0.30644865641416347
## 0.298586805810744
## 0.2912641380612447
## 0.2844429670767149
## 0.2780882420103697
## 0.27216736298336
## 0.26665000969626856
## 0.26150798202526704
## 0.2567150517648893
## 0.25224682473796955
## 0.24808061254780375
## 0.2441953132982817
## 0.24057130065488982
## 0.2371903206633321
## 0.23403539578330254
## 0.23109073563287788
## 0.22834165397427683
## 0.22577449150454584
## 0.22337654404525148
## 0.22113599575364543
## 0.21904185700416282
## 0.2170839066136745
## 0.21525263810674786
## 0.2135392097384097
## 0.21193539801166192
## 0.2104335544453725
## 0.209026565365253
## 0.20770781450652795
## 0.20647114823168267
## 0.20531084318042545
## 0.20422157618178793
## 0.20319839627017727
## 0.20223669865825925
## 0.2013322005298366
## 0.2004809185254555
## 0.19967914780237356
## 0.19892344255879904
## 0.19821059792000922
## 0.19753763309111594
## 0.19690177568790568
## 0.19630044716337505
## 0.19573124925334387
## 0.1951919513698831
## 0.19468047887628182
## 0.19419490218190888
## 0.1937334265996366
## 0.19329438291250436
## 0.192876218600024
## 0.19247748967800374
## 0.1920968531089874
## 0.19173305974340785
## 0.19138494775434317
## 0.1910514365313603
## 0.19073152100134264
## 0.19042426634644535
## 0.19012880309140615
## 0.1898443225343861
## 0.18957007249731683
## 0.18930535337341367
## 0.18904951445107168
## 0.18880195049482287
## 0.1885620985653743
## 0.1883294350620115
## 0.18810347297181784
## [[0.63577887]
##  [1.06481842]
##  [0.35278444]
##  [0.49702146]]
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
## 0.6257538535768574
## 0.15226461943531708
## 0.14573080915532183
## 0.14038174019871447
## 0.1357713007268658
## 0.1317890295802218
## 0.12834148570985976
## 0.12534945212826856
## 0.12274575826090645
## 0.1204734362554304
## 0.11850279171835217
## [[ 0.60253875]
##  [ 1.00409182]
##  [ 0.71288828]
##  [-0.43603821]]
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
## 6.054056387858538
## 0.10213863117048172
## 0.09801195217985904
## 0.0970423506685451
## 0.09672902952465014
## 0.09658040253979552
## 0.09649081094291291
## 0.09643156638553299
## 0.0963912579417566
## 0.0963636103430162
## 0.09634461987106427
## [[ 1.79314134]
##  [ 0.66651165]
##  [ 0.71763717]
##  [-0.57117203]]
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
## 21.925553101782093
## 0.10707860395037187
## 0.10349314518233517
## 0.10120761379694859
## 0.09966941947453367
## 0.09861758943667381
## 0.09789512731319158
## 0.09739828395257144
## 0.09705648461882989
## 0.09682132461221309
## 0.09665966240535519
## [[ 1.67268262]
##  [ 0.69650523]
##  [ 0.73412148]
##  [-0.59978351]]
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
## 0.9034138751418307
## 0.1432588615329011
## 0.12140574123744843
## 0.11121130821219301
## 0.10613945986906217
## 0.10338628293959977
## 0.1017331180425958
## 0.10063883431516693
## 0.0998547821414152
## 0.09926062002147777
## 0.09879791824728336 - final score
## [[ 1.37490772]
##  [ 0.77395536]
##  [ 0.75231934]
##  [-0.60923102]]
```

* lesson #18: in mini-batch processing it is comfortable to use placeholders


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
## If you want the future behaviour and silence this warning, you can specify "categories='auto'".
## In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
##   warnings.warn(msg, FutureWarning)
```

```python
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
## 0.250339957498485
## 0.22321047662610255
## 0.22158124922335146
## 0.22072156246823213
## 0.22010995943548498
## 0.21959723840541806
## 0.2191275955114837
## 0.21867712177090953
## 0.2182352053908528
## 0.2177974727879094
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
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
## If you want the future behaviour and silence this warning, you can specify "categories='auto'".
## In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
##   warnings.warn(msg, FutureWarning)
```

```python
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
## 0.2999745859153437
## 0.1445341159122835
## 0.10480632796627741
## 0.08837311355646629
## 0.07610644769648213
## 0.06660072141874983
## 0.05910497260698746
## 0.05311104101867989
## 0.048278270580648465
## 0.044356719609718744
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
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
## If you want the future behaviour and silence this warning, you can specify "categories='auto'".
## In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
##   warnings.warn(msg, FutureWarning)
```

```python
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
## 0.24168211890969238
## 0.14995626743055615
## 0.11013791552669502
## 0.09312001280428404
## 0.07983561247849161
## 0.06963812007520152
## 0.061647622751834025
## 0.055255277986801764
## 0.05008224575785478
## 0.04586894177598112
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.9666666666666667
```


```python
learning_rate = 0.01
iris = load_iris()
X_np, y_np = iris.data, iris.target
ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
## If you want the future behaviour and silence this warning, you can specify "categories='auto'".
## In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
##   warnings.warn(msg, FutureWarning)
```

```python
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
## 0.2727944387429683
## 0.05517311597100655
## 0.024169896132494605
## 0.01818011479862492
## 0.015724123115681384
## 0.014354724761808725
## 0.013455100933521653
## 0.012806911171404778
## 0.012309824604451614
## 0.011913171423626199
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.98
```
