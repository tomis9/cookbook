---
title: "ml"
date: 2019-04-22T18:05:21+02:00
draft: true
categories: []
tags: []
---


# 1. What is machine learning and why would you use it?

* it's a rather complicated, yet beautiful tool to boldly go where no man has gone before.

# 2. Examples of the most popular machine learning algorithms in Python and R.

We'll be working on `iris` dataset, which is easily available in Python (`from sklearn import datasets; datasets.load_iris()`) and R (`iris`).

## SVM

*Data mining and analysis - Zaki, Meira*

```{python}
from sklearn.svm import SVC  # support vector classification
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

svc = SVC()
svc.fit(X, y)

accuracy_score(svc.predict(X), y)
# sum(svc.predict(X) == y) / len(y)  # the same as accuracy_score

# 0.986667  # pretty good
```

```{r}
svc <- e1071::svm(Species ~ ., iris)

pred <- as.character(predict(svc, iris[, 1:4]))
mean(pred == iris["Species"])
# 0.9733  # pretty good  as well
```

## decision trees

```{python}
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

dtc = DecisionTreeClassifier()
dtc.fit(X, y)

accuracy_score(y, dtc.predict(X))  # clearly overfitted
```

[link to my blog post](https://tomis9.github.io/decision_trees/)
```{r}
dtc <- rpart::rpart(Species ~ ., iris)
print(dtc)
# rpart.plot::rpart.plot(dtc)
```


*to be continued...*
