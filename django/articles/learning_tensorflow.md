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
## [[0.46274322]
##  [0.34535956]
##  [0.12199711]
##  [0.9349485 ]]
## [[0.50394258]
##  [0.47185822]
##  [0.27376649]
##  [0.98217181]]
## 7.748357983967208
## 3.2123600272695865
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
## 2.013420402935302
## 1.0596345513116183
## 0.8419045926018387
## 0.7697923159039369
## 0.7281012773367034
## 0.6942438033080738
## 0.6636863976354637
## 0.6354225178503167
## 0.6091413893681581
## 0.5846757702681026
## 0.5618937979148039
## 0.5406774008489892
## 0.520917613211537
## 0.5025132158285698
## 0.4853700564626999
## 0.46940053122830877
## 0.4545231242378201
## 0.4406619835589561
## 0.4277465274774112
## 0.415711078280737
## 0.40449452151903764
## 0.3940399889465764
## 0.38429456349391716
## 0.3752090047389349
## 0.36673749345223317
## 0.3588373938916414
## 0.35146903261256585
## 0.3445954926466718
## 0.3381824219810968
## 0.3321978553445906
## 0.32661204837601154
## 0.3213973233148504
## 0.3165279254132247
## 0.3119798893244173
## 0.3077309147747794
## 0.3037602508739921
## 0.30004858846348853
## 0.2965779599445399
## 0.293331646066322
## 0.29029408919037486
## 0.2874508125814763
## 0.2847883453062119
## 0.28229415234961636
## 0.2799565695873332
## 0.27776474327593
## 0.27570857374744606
## 0.27377866301606074
## 0.27196626602506785
## 0.27026324528122575
## 0.2686620286411268
## 0.267155570030585
## 0.26573731289325137
## 0.26440115617883314
## 0.26314142269446045
## 0.26195282965501177
## 0.2608304612796116
## 0.2597697432921323
## 0.2587664191934092
## 0.2578165281820696
## 0.2569163846094301
## 0.25606255886187385
## 0.25525185957152646
## 0.25448131706293947
## 0.2537481679499043
## 0.2530498408024836
## 0.25238394280990195
## 0.2517482473701037
## 0.25114068254159166
## 0.25055932029763595
## 0.25000236652710317
## 0.2494681517300325
## 0.24895512235968426
## 0.24846183276614725
## 0.24798693769970537
## 0.24752918533507404
## 0.24708741078031352
## 0.24666053003674543
## 0.24624753437853647
## 0.24584748512279025
## 0.24545950876301376
## 0.2450827924407134
## 0.2447165797316251
## 0.24436016672471855
## 0.24401289837363369
## 0.24367416510161968
## 0.2433433996423644
## 0.2430200741003241
## 0.2427036972153013
## 0.24239381181708192
## 0.24208999245692345
## 0.24179184320360902
## 0.241498995592632
## 0.24121110671787174
## 0.24092785745585957
## 0.24064895081342494
## 0.2403741103901451
## 0.24010307894762653
## 0.23983561707819148
## 0.23957150196606525
## 0.23931052623463558
## [[0.35542835]
##  [1.17900628]
##  [0.25105562]
##  [0.75999903]]
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
## 1.2275395813631353
## 0.12892251556198686
## 0.12342884580245299
## 0.11944873358598293
## 0.1160745041064279
## 0.11321231530564717
## 0.11078332850454849
## 0.10872088646535673
## 0.10696862935146181
## 0.10547890245505692
## 0.10422310158951868
## [[ 1.34860626]
##  [ 0.81610867]
##  [ 0.62819206]
##  [-0.30565261]]
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
## 26.28245649672255
## 0.14853358224508414
## 0.1156044516823384
## 0.10638688335689764
## 0.10254627634672858
## 0.10036817635349245
## 0.09895493411074609
## 0.09799768552656701
## 0.09734133395488694
## 0.09688978738176844
## 0.09657911344688519
## [[ 1.59023222]
##  [ 0.71846913]
##  [ 0.7459023 ]
##  [-0.62325895]]
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
## 9.761564447292104
## 0.10452623965847693
## 0.10070933039613321
## 0.09899938823139837
## 0.09799280633894669
## 0.09733135891630275
## 0.09688168405401017
## 0.09657304550262902
## 0.09636065467200852
## 0.09621439352432087
## 0.09611373585131124
## [[ 1.70003518]
##  [ 0.69104903]
##  [ 0.73094457]
##  [-0.59724266]]
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
## 0.2302049427189175
## 0.11650408698287386
## 0.11027475353684027
## 0.1066779093588757
## 0.10437068005478495
## 0.10275185443204894
## 0.10153973113480862
## 0.10059313539434209
## 0.0998347902588752
## 0.0992179902628101
## 0.0987163347189228 - final score
## [[ 1.32887785]
##  [ 0.78614124]
##  [ 0.76368502]
##  [-0.63571121]]
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
## 0.2847753291394739
## 0.22257607487219008
## 0.2180598731174556
## 0.21684703306730338
## 0.21628414806598367
## 0.21591714959800756
## 0.21562513946298048
## 0.2153676234794652
## 0.21512908286112056
## 0.21490301860269898
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
## 0.28105510858594823
## 0.1344922356593923
## 0.10815536039529484
## 0.09118954905842484
## 0.07790418840736872
## 0.0679491131359242
## 0.06028569949396143
## 0.0542414727880494
## 0.049395501171880504
## 0.04546056755976462
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
## 0.2177565034714002
## 0.1156304719849687
## 0.09472057255335657
## 0.07996758146153313
## 0.06907428429063953
## 0.060807234985156125
## 0.05438419498668668
## 0.04930802851855549
## 0.04523807817447255
## 0.04192967086711092
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
## 0.3868771293408352
## 0.034033477542872885
## 0.019637079137531405
## 0.015870527609139127
## 0.014148791464928479
## 0.013141111218144557
## 0.012465283947972979
## 0.011973313838342044
## 0.011595084490277213
## 0.011292958646966543
```

```python
print(calculate_accuracy(preds))  # maybe overfitting?
```

```
## 0.98
```
