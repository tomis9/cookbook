---
title: "sklearn regression"
date: 2018-10-11T21:12:29+01:00
draft: false
image: "sklearn_regression.jpg"
categories: ["python", "statistics"]
tags: ["python", "statistics"]
---

## What is regression and why would you use it?

* regression is probably the most popualr algorithm in data analysis;

* it is very simple but yet very powerful;

* there are many kinds of regression, but here we will concentrate on the simplest and the most popular one: **linear regression**.

To be honest, I recommend using R programming language for calculating regression. I have several thoughts when you should use Python or R and it's a subject for a longer discussion.

## 1. Prerequisites

You have to have python 3 installed on your machine. You can use Python 2 if you want, but c'mon, it's 2018.

In theory you could calculate linear regression in python 'by hand' using numpy or pandas and it's matrix multiplication functions, but `sklearn` provides you with a nice interface and additional statistics. Let's install it:

```{bash}
sudo pip3 install sklearn
```

## 2. A 'Hello world' example

Once you have you libraries loaded:
```{python}
import sklearn.linear_model
import pandas as pd
import io
import requests
```

you can import a dataset that we're going to use. We'll do it from Python by providing it's url.

```{python}
url = "https://raw.githubusercontent.com/gagolews/Analiza_danych_w_jezyku_Python/master/zbiory_danych/winequality-all.csv"
s = requests.get(url).content
wine = pd.read_csv(io.StringIO(s.decode('utf-8')), comment="#")
```

*The dataset that we're going to work on was published by Marek Gągolewski as a example dataset for execrises for his book 'Przetwarzanie i analiza danych w języku Python`. I am afraid the book wasn't transalted to English, so I can highly recommend it only to Polish readers.*

Let's choose only white wine and the first eleven variables of the dataset.
```{python}
white_wine = wine[wine.color == "white"]
white_wine = white_wine.iloc[:, 0:11]
```

Now let's create a dependent and independent variables based on our dataset:
```{python}
y = white_wine.iloc[:, -1]
X = white_wine.iloc[:, :-1]
```

And now - the regression part.
```{python}
ls = sklearn.linear_model.LinearRegression()
ls.fit(X, y)
print(ls.intercept_)
print(ls.coef_)
```
