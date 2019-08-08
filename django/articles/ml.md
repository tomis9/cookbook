---
title: "basic machine learning algorithms"
date: 2019-04-22T18:05:21+02:00
draft: false
categories: ["Machine learning"]
tags: []
---






## 1. What is machine learning and why would you use it? 

* it's a rather complicated, yet beautiful tool for boldly going where no man has gone before.

* in other words, it enables you to extract valuable information from data.

## 2. Examples of the most popular machine learning algorithms in Python and R 

We'll be working on `iris` dataset, which is easily available in Python (`from sklearn import datasets; datasets.load_iris()`) and R (`data(iris)`).

We will use a few of the most popular machine learning tools: 

* R base, 
  
* R caret, 
  
    * [a short introduction to caret](https://cran.r-project.org/web/packages/caret/vignettes/caret.html)

    * [The *caret* package](http://topepo.github.io/caret/index.html)

* Python scikit-learn,
  
* Python API to [tensorflow](http://tomis9.com/tensorflow). `tensorflow` was used in very few cases, because it is designed mainly for neural networks, and we would have to implement the algorithms from scratch.

Let's prepare data for our algorithms. You can read more about data preparation in [this blog post](http://tomis9.com/useful_processing).

### data preparation 

*Python*

```python
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
boston = datasets.load_boston()
# I will divide boston dataset to train and test later on
```

*tensorflow*

You can use functions from [tensorflow_datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3) module, but... does anybody use them?

*base R*

```r
# unfortunately base R does not provide a function for train/test split
train_test_split <- function(test_size = 0.33, dataset) {
    smp_size <- floor(test_size * nrow(dataset))
    test_ind <- sample(seq_len(nrow(dataset)), size = smp_size)

    test <- dataset[test_ind, ]
    train <- dataset[-test_ind, ]
    return(list(train = train, test = test))
}

library(gsubfn)
list[train, test] <- train_test_split(0.5, iris)
```

*caret R*

```r
# docs: ..., the random sampling is done within the
# levels of ‘y’ when ‘y’ is a factor in an attempt to balance the class
# distributions within the splits.
# I provide package's name before function's name for clarity
```

```r
trainIndex <- caret::createDataPartition(iris$Species, p=0.7, list = FALSE, 
                                         times = 1)
train <- iris[trainIndex,]
test <- iris[-trainIndex,]
```

### SVM 

The best description of SVM I found is in *Data mining and analysis - Zaki, Meira*. In general, I highly recommend this book. 

*Python*

```python
from sklearn.svm import SVC  # support vector classification
svc = SVC()
svc.fit(X_train, y_train)
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
##   "avoid this warning.", FutureWarning)
```

```python
print(accuracy_score(svc.predict(X_test), y_test))
```

```
## 1.0
```

TODO: [plotting svm in scikit](https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html)

*"base" R*

```r
svc <- e1071::svm(Species ~ ., train)

pred <- as.character(predict(svc, test[, 1:4]))
mean(pred == test["Species"])
```

```
## [1] 1
```

*caret R*

```r
svm_linear <- caret::train(Species ~ ., data = train, method = "svmLinear")
mean(predict(svm_linear, test) == test$Species)
```

```
## [1] 1
```

### decision trees 

*Python*

```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print(accuracy_score(y_test, dtc.predict(X_test)))
```

```
## 0.98
```

TODO: [an article on drawing decision trees in Python](https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176)

*"base" R*

```r
dtc <- rpart::rpart(Species ~ ., train)
print(dtc)
```

```
## n= 105 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 105 70 setosa (0.33333 0.33333 0.33333)  
##   2) Petal.Length< 2.45 35  0 setosa (1.00000 0.00000 0.00000) *
##   3) Petal.Length>=2.45 70 35 versicolor (0.00000 0.50000 0.50000)  
##     6) Petal.Width< 1.75 39  5 versicolor (0.00000 0.87179 0.12821) *
##     7) Petal.Width>=1.75 31  1 virginica (0.00000 0.03226 0.96774) *
```

```r
rpart.plot::rpart.plot(dtc)
```

![plot of chunk unnamed-chunk-9](./articles/figures/ml/unnamed-chunk-9-1.png)

```r
pred <- predict(dtc, test[,1:4], type = "class")
mean(pred == test[["Species"]])
```

```
## [1] 1
```

*caret R*

```r
c_dtc <- caret::train(Species ~ ., train, method = "rpart")
print(c_dtc)
```

```
## CART 
## 
## 105 samples
##   4 predictors
##   3 classes: 'setosa', 'versicolor', 'virginica' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 105, 105, 105, 105, 105, 105, ... 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa 
##   0.0000  0.9092    0.8616
##   0.4143  0.7318    0.6215
##   0.5000  0.4682    0.2515
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.
```

```r
rpart.plot::rpart.plot(c_dtc$finalModel)
```

![plot of chunk unnamed-chunk-10](./articles/figures/ml/unnamed-chunk-10-1.png)

I described working with decision trees in R in more detail in [another blog post](http://tomis9.com/decision_trees/).


### random forests 

*Python*

```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
##   "10 in version 0.20 to 100 in 0.22.", FutureWarning)
```

```python
print(accuracy_score(rfc.predict(X_test), y_test))
```

```
## 0.98
```

*"base" R*

```r
rf <- randomForest::randomForest(Species ~ ., data = train)
mean(predict(rf, test[, 1:4]) == test[["Species"]])
```

```
## [1] 1
```

*caret R*

```r
c_rf <- caret::train(Species ~ ., train, method = "rf")
print(c_rf)
```

```
## Random Forest 
## 
## 105 samples
##   4 predictors
##   3 classes: 'setosa', 'versicolor', 'virginica' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 105, 105, 105, 105, 105, 105, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##   2     0.9229    0.8822
##   3     0.9224    0.8815
##   4     0.9203    0.8784
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
print(c_dtc$finalModel)
```

```
## n= 105 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 105 70 setosa (0.33333 0.33333 0.33333)  
##   2) Petal.Length< 2.45 35  0 setosa (1.00000 0.00000 0.00000) *
##   3) Petal.Length>=2.45 70 35 versicolor (0.00000 0.50000 0.50000)  
##     6) Petal.Width< 1.75 39  5 versicolor (0.00000 0.87179 0.12821) *
##     7) Petal.Width>=1.75 31  1 virginica (0.00000 0.03226 0.96774) *
```


### knn 

*Python*

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, y)
print(accuracy_score(y, knn.predict(X)))
```

```
## 0.9666666666666667
```

*R*

```r
kn <- class::knn(train[,1:4], test[,1:4], cl = train[,5], k = 3) 
mean(kn == test[,5])
```

```
## [1] 1
```

TODO: caret r knn


### kmeans 

K-means can be nicely plotted in two dimensions with help of [PCA](http://tomis9.com/dimensionality/#pca).

*R*

```r
pca <- prcomp(iris[,1:4], center = TRUE, scale. = TRUE)
# devtools::install_github("vqv/ggbiplot")
ggbiplot::ggbiplot(pca, obs.scale = 1, var.scale = 1, groups = iris$Species, 
                   ellipse = TRUE, circle = TRUE)
```

![plot of chunk unnamed-chunk-16](./articles/figures/ml/unnamed-chunk-16-1.png)

```r
iris_pca <- scale(iris[,1:4]) %*% pca$rotation 
iris_pca <- as.data.frame(iris_pca)
iris_pca <- cbind(iris_pca, Species = iris$Species)

ggplot2::ggplot(iris_pca, aes(x = PC1, y = PC2, color = Species)) +
  geom_point()
```

![plot of chunk unnamed-chunk-16](./articles/figures/ml/unnamed-chunk-16-2.png)

```r
plot_kmeans <- function(km, iris_pca) {
  # we choose only first two components, so they could be plotted
  plot(iris_pca[,1:2], col = km$cluster, pch = as.integer(iris_pca$Species))
  points(km$centers, col = 1:2, pch = 8, cex = 2)
}
par(mfrow=c(1, 3))
# we use 3 centers, because we already know that there are 3 species
sapply(list(kmeans(iris_pca[,1], centers = 3),
            kmeans(iris_pca[,1:2], centers = 3),
            kmeans(iris_pca[,1:4], centers = 3)),
       plot_kmeans, iris_pca = iris_pca)
```

![plot of chunk unnamed-chunk-16](./articles/figures/ml/unnamed-chunk-16-3.png)

```
## [[1]]
## NULL
## 
## [[2]]
## NULL
## 
## [[3]]
## NULL
```

interesting article - [kmeans with dplyr and broom](https://cran.r-project.org/web/packages/broom/vignettes/kmeans.html)

TODO: caret r - kmeans

TODO: python - kmeans


### linear regression 

*Python*

```python
from sklearn.linear_model import LinearRegression
from sklearn import datasets
X, y = boston.data, boston.target
lr = LinearRegression()
lr.fit(X, y)
print(lr.intercept_)
```

```
## 36.45948838509017
```

```python
print(lr.coef_)
# TODO calculate this with iris dataset
```

```
## [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
##  -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
##   3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
##  -5.24758378e-01]
```

*tensorflow*

- data preparation


```python
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
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
```

- using normal equations


```python
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

- using gradient descent and mini-batches

```python
learning_rate = 0.01
n_iter = 1000
X_train, y_train = get_data(tensorflow=False)
X = tf.placeholder("float64", shape=(None, 4))  # placeholder -
y = tf.placeholder("float64", shape=(None, 1))
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
```

```
## --- Logging error ---
## Traceback (most recent call last):
##   File "/usr/lib/python3.5/logging/__init__.py", line 982, in emit
##     stream.write(msg)
## ValueError: I/O operation on closed file
## Call stack:
##   File "<string>", line 1, in <module>
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 213, in __call__
##     return cls._variable_v1_call(*args, **kwargs)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 176, in _variable_v1_call
##     aggregation=aggregation)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 155, in <lambda>
##     previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variable_scope.py", line 2495, in default_variable_creator
##     expected_shape=expected_shape, import_scope=import_scope)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 217, in __call__
##     return super(VariableMetaclass, cls).__call__(*args, **kwargs)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 1395, in __init__
##     constraint=constraint)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/variables.py", line 1547, in _init_from_args
##     validate_shape=validate_shape).op
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/state_ops.py", line 223, in assign
##     validate_shape=validate_shape)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/gen_state_ops.py", line 64, in assign
##     use_locking=use_locking, name=name)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py", line 784, in _apply_op_helper
##     with _MaybeColocateWith(must_colocate_inputs):
##   File "/usr/lib/python3.5/contextlib.py", line 59, in __enter__
##     return next(self.gen)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py", line 263, in _MaybeColocateWith
##     with ops.colocate_with(inputs[0]), _MaybeColocateWith(inputs[1:]):
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/util/deprecation.py", line 323, in new_func
##     instructions)
##   File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/platform/tf_logging.py", line 166, in warning
##     get_logger().warning(msg, *args, **kwargs)
## Message: 'From %s: %s (from %s) is deprecated and will be removed %s.\nInstructions for updating:\n%s'
## Arguments: ('/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py:263', 'colocate_with', 'tensorflow.python.framework.ops', 'in a future version', 'Colocations handled automatically by placer.')
```

```python
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(mse)
batch_indexes = np.arange(150).reshape(5,30)
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
## 0.20996323569403932
## 0.13080024229227408
## 0.12419480345543107
## 0.11914881531775269
## 0.11515888444090067
## 0.11193960404720589
## 0.10931182590358006
## 0.1071524423953464
## 0.10537086422104061
## 0.10389730814029338
## 0.10268758165256191 - final score
## [[ 1.07633931]
##  [ 0.84744059]
##  [ 0.79871093]
##  [-0.69760022]]
```

*base R*


```r
model <- lm(Sepal.Length ~ ., train)
summary(model)
```

```
## 
## Call:
## lm(formula = Sepal.Length ~ ., data = train)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.7029 -0.2265  0.0273  0.1883  0.7501 
## 
## Coefficients:
##                   Estimate Std. Error t value             Pr(>|t|)
## (Intercept)          1.984      0.329    6.04          0.000000027
## Sepal.Width          0.568      0.101    5.61          0.000000182
## Petal.Length         0.807      0.081    9.96 < 0.0000000000000002
## Petal.Width         -0.380      0.178   -2.13                0.035
## Speciesversicolor   -0.519      0.295   -1.76                0.081
## Speciesvirginica    -0.786      0.405   -1.94                0.055
## 
## Residual standard error: 0.313 on 99 degrees of freedom
## Multiple R-squared:  0.867,	Adjusted R-squared:  0.86 
## F-statistic:  129 on 5 and 99 DF,  p-value: <0.0000000000000002
```

`lm()` function automatically converts factor variables to one-hot encoded features.

*R caret*

```r
library(caret)

m <- train(Sepal.Length ~ ., data = train, method = "lm")
summary(m)  # exactly the same as lm()
```

```
## 
## Call:
## lm(formula = .outcome ~ ., data = dat)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.7029 -0.2265  0.0273  0.1883  0.7501 
## 
## Coefficients:
##                   Estimate Std. Error t value             Pr(>|t|)
## (Intercept)          1.984      0.329    6.04          0.000000027
## Sepal.Width          0.568      0.101    5.61          0.000000182
## Petal.Length         0.807      0.081    9.96 < 0.0000000000000002
## Petal.Width         -0.380      0.178   -2.13                0.035
## Speciesversicolor   -0.519      0.295   -1.76                0.081
## Speciesvirginica    -0.786      0.405   -1.94                0.055
## 
## Residual standard error: 0.313 on 99 degrees of freedom
## Multiple R-squared:  0.867,	Adjusted R-squared:  0.86 
## F-statistic:  129 on 5 and 99 DF,  p-value: <0.0000000000000002
```


### logistic regression 

In these examples I will present classification of a dummy variable.

*Python*

```python
from sklearn.linear_model import LogisticRegression
cond = iris.target != 2
X = iris.data[cond]
y = iris.target[cond]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
##   FutureWarning)
```

```python
accuracy_score(lr.predict(X_test), y_test)
```

*base R*

```r
species <- c("setosa", "versicolor")
d <- iris[iris$Species %in% species,]
d$Species <- factor(d$Species, levels = species)
library(gsubfn)
list[train, test] <- train_test_split(0.5, d)

m <- glm(Species ~ Sepal.Length, train, family = binomial)
# predictions - if prediction is bigger than 0.5, we assume it's a one, 
# or success
y_hat_test <- predict(m, test[,1:4], type = "response") > 0.5

# glm's doc:
# For ‘binomial’ and ‘quasibinomial’ families the response can also
# be specified as a ‘factor’ (when the first level denotes failure
# and all others success) or as a two-column matrix with the columns
# giving the numbers of successes and failures.
# in our case - species[1] ("setosa") is a failure (0) and species[2] 
# ("versicolor") is 1 (success)
# successes:
y_test <- test$Species == species[2]

mean(y_test == y_hat_test)
```

```
## [1] 0.94
```

*R caret*

```r
library(caret)
m2 <- train(Species ~ Sepal.Length, data = train, method = "glm", family = binomial)
mean(predict(m2, test) == test$Species)
```

```
## [1] 0.94
```


### xgboost 

*Python*

```python
from xgboost import XGBClassifier
cond = iris.target != 2
X = iris.data[cond]
y = iris.target[cond]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
accuracy_score(xgb.predict(X_test), y_test)
```

*"base" R*

```r
species <- c("setosa", "versicolor")
d <- iris[iris$Species %in% species,]
d$Species <- factor(d$Species, levels = species)
library(gsubfn)
list[train, test] <- train_test_split(0.5, d)

library(xgboost)
m <- xgboost(
  data = as.matrix(train[,1:4]), 
  label = as.integer(train$Species) - 1,
  objective = "binary:logistic",
  nrounds = 2)
```

```
## [1]	train-error:0.000000 
## [2]	train-error:0.000000
```

```r
mean(predict(m, as.matrix(test[,1:4])) > 0.5) == (as.integer(test$Species) - 1)
```

```
##  [1] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [12] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [23] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [34] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [45] FALSE FALSE FALSE FALSE FALSE FALSE
```

TODO: R: xgb.save(), xgb.importance()

*caret R*

TODO: [tuning xgboost with caret](https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret)
