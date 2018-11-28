---
title: "spark"
date: 2018-11-23T12:58:49+02:00
draft: false
categories: ["data-engineering", "python"]
tags: ["draft", "spark", "data-engineering", "python"]
---

## 1. What is spark and why would use use it?

* Spark is a smooth framework for working with big data, i.e. [hdfs](https://tomis9.github.io/hdfs);

* it can be accessed from Python, R, scala (spark is actually written in scala) and java;

* it is probably the most popular big data tool nowadays for data scientists.

## A good tutorial:

But before watching the film you should install pyspark:

```{python}
sudo pip3 install pyspark
```
<iframe width="1620" height="595" src="https://www.youtube.com/embed/wi_PPloqRe0?list=PLE50-dh6JzC5zo2whIGqJ02CIhP3ysQLX" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

My notes from the film above:

Importing pyspark.sql, which let's you work with dataframes and launching SparkSession, which is a connection with spark. If you do not specifically provide details of spark installation on cluster/server/laptop you use, pyspark will use it's own, development spark session.

```{python}
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("saprk.come.config.option", "som-value") \
    .getOrCreate()
```

Let's download a dataset called 'creditcard fraud detection' from kaggle and read it into spark.
```{python}
df = spark.read.csv('creditcard.csv')
```

Basic dataframe operations:
```{python}
# prints columns' names and datatypes
df.printSchema()

df.head()

df.count()

# prints nothing interesting
df.describe()

# but... this reveals some information:
df.describe().show()

# do we have any null values?
df.dropna().count()

# what if we had nas?
df.fillna(-1).show(5)
```

columns

select

filter

data from mysql

