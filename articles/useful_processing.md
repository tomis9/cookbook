---
title: "useful processing"
date: 2019-05-17T15:44:38+02:00
draft: false
categories: ["Machine learning", "Data engineering"]
tags: []
---






## 1. What is useful processing? 

* many machine learning algorithms require the same kinds of data preprocessing in order for them to work properly. In other words, theses kinds of processing are useful.

## 2. Examples 

### one-hot encoding 

*R*

```r
# data.table
dt_iris <- data.table::as.data.table(iris)
mltools::one_hot(dt_iris)
```

```
##      Sepal.Length Sepal.Width Petal.Length Petal.Width Species_setosa
##   1:          5.1         3.5          1.4         0.2              1
##   2:          4.9         3.0          1.4         0.2              1
##   3:          4.7         3.2          1.3         0.2              1
##   4:          4.6         3.1          1.5         0.2              1
##   5:          5.0         3.6          1.4         0.2              1
##  ---                                                                 
## 146:          6.7         3.0          5.2         2.3              0
## 147:          6.3         2.5          5.0         1.9              0
## 148:          6.5         3.0          5.2         2.0              0
## 149:          6.2         3.4          5.4         2.3              0
## 150:          5.9         3.0          5.1         1.8              0
##      Species_versicolor Species_virginica
##   1:                  0                 0
##   2:                  0                 0
##   3:                  0                 0
##   4:                  0                 0
##   5:                  0                 0
##  ---                                     
## 146:                  0                 1
## 147:                  0                 1
## 148:                  0                 1
## 149:                  0                 1
## 150:                  0                 1
```

```r
# caret
library(caret)
dummy <- caret::dummyVars(" ~ .", data = iris)
head(predict(dummy, iris))
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species.setosa
## 1          5.1         3.5          1.4         0.2              1
## 2          4.9         3.0          1.4         0.2              1
## 3          4.7         3.2          1.3         0.2              1
## 4          4.6         3.1          1.5         0.2              1
## 5          5.0         3.6          1.4         0.2              1
## 6          5.4         3.9          1.7         0.4              1
##   Species.versicolor Species.virginica
## 1                  0                 0
## 2                  0                 0
## 3                  0                 0
## 4                  0                 0
## 5                  0                 0
## 6                  0                 0
```

```r
# dplyr is not that clever
library(dplyr)
iris %>%
  mutate("Species_setosa" = ifelse(Species == "setosa", 1, 0)) %>%
  mutate("Species_virgninica" = ifelse(Species == "virgninica", 1, 0)) %>%
  mutate("Species_versicolor" = ifelse(Species == "versicolor", 1, 0)) %>%
  head()
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species Species_setosa
## 1          5.1         3.5          1.4         0.2  setosa              1
## 2          4.9         3.0          1.4         0.2  setosa              1
## 3          4.7         3.2          1.3         0.2  setosa              1
## 4          4.6         3.1          1.5         0.2  setosa              1
## 5          5.0         3.6          1.4         0.2  setosa              1
## 6          5.4         3.9          1.7         0.4  setosa              1
##   Species_virgninica Species_versicolor
## 1                  0                  0
## 2                  0                  0
## 3                  0                  0
## 4                  0                  0
## 5                  0                  0
## 6                  0                  0
```

As you can see, `caret` recognised dummy variables (`Species`) and processed them to binary variables.

TODO: `library(dummies); library(onehot)`

*Python*

```python
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
data = datasets.load_iris()
y = [data.target_names[i] for i in data.target]
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded[:5])
# binary encode
```

```
## [0 0 0 0 0]
```

```python
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded[:5])
```

```
## [[1. 0. 0.]
##  [1. 0. 0.]
##  [1. 0. 0.]
##  [1. 0. 0.]
##  [1. 0. 0.]]
```

### scaling 

In R it's extremely simple.
*base R*

```r
scale(1:5)
```

```
##         [,1]
## [1,] -1.2649
## [2,] -0.6325
## [3,]  0.0000
## [4,]  0.6325
## [5,]  1.2649
## attr(,"scaled:center")
## [1] 3
## attr(,"scaled:scale")
## [1] 1.581
```

```r
mean(1:5)
```

```
## [1] 3
```

```r
sd(1:5)
```

```
## [1] 1.581
```

```r
sc <- scale(iris[,1:4])
head(sc)
```

```
##      Sepal.Length Sepal.Width Petal.Length Petal.Width
## [1,]      -0.8977     1.01560       -1.336      -1.311
## [2,]      -1.1392    -0.13154       -1.336      -1.311
## [3,]      -1.3807     0.32732       -1.392      -1.311
## [4,]      -1.5015     0.09789       -1.279      -1.311
## [5,]      -1.0184     1.24503       -1.336      -1.311
## [6,]      -0.5354     1.93331       -1.166      -1.049
```

```r
attributes(sc)
```

```
## $dim
## [1] 150   4
## 
## $dimnames
## $dimnames[[1]]
## NULL
## 
## $dimnames[[2]]
## [1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width" 
## 
## 
## $`scaled:center`
## Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
##        5.843        3.057        3.758        1.199 
## 
## $`scaled:scale`
## Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
##       0.8281       0.4359       1.7653       0.7622
```

*Python*

```python
from sklearn.preprocessing import scale, StandardScaler
from sklearn import datasets
import numpy as np
# simple approach
sc = scale(np.arange(1, 6))
```

```
## /usr/local/lib/python3.5/dist-packages/sklearn/utils/validation.py:444: DataConversionWarning: Data with input dtype int64 was converted to float64 by the scale function.
##   warnings.warn(msg, DataConversionWarning)
```

```python
print(np.std(sc))
```

```
## 0.9999999999999999
```

```python
print(np.mean(sc))
# and a way compatible with pandas
```

```
## 0.0
```

```python
data = datasets.load_iris()
X, y = data.data, data.target
scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)
print(scaled_df.mean(axis=0))
```

```
## [-1.69031455e-15 -1.63702385e-15 -1.48251781e-15 -1.62314606e-15]
```

```python
print(scaled_df.std(axis=0))
```

```
## [1. 1. 1. 1.]
```

### splitting your dataset into train and test subsets 

The idea for the following solution comes from [this post at stackoverflow](https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function).

*base R*

```r
train_test_split <- function(test_proportion = 0.75, dataset) {
    smp_size <- floor(test_proportion * nrow(dataset))
    train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)

    train <- dataset[train_ind, ]
    test <- dataset[-train_ind, ]
    return(list(train = train, test = test))
}
library(gsubfn)
list[train, test] <- train_test_split(0.8, iris)
```

*caret R*

```r
library(caret)
# ..., the random sampling is done within the
# levels of ‘y’ when ‘y’ is a factor in an attempt to balance the class
# distributions within the splits.
# I provide package's name before function's name for clarity
trainIndex <- caret::createDataPartition(iris$Species, p=0.7, list = FALSE, 
                                         times = 1)
train <- iris[trainIndex,]
test <- iris[-trainIndex,]
```

*Python - sklearn*

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
data = datasets.load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
```

*Python - pandas*

```python
import pandas as pd
# data = pd.DataFrame(data)
# train = data.sample(frac=0.8)
# test = data.drop(train.index)
```

### sklearn pipeline 

- [a short article about sklearn pipelines](https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976)


```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
data = datasets.load_iris()
X, y = data.data, data.target
# one way
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svc = SVC()
svc.fit(X_scaled, y)
# or another - with pipeline
from sklearn.pipeline import Pipeline
svc = Pipeline([('scaler', StandardScaler()), ('SVM', SVC())])
svc.fit(X, y)
```

TODO: `pd.get_dummies()`
