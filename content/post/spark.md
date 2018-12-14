---
title: "spark"
date: 2018-11-23T12:58:49+02:00
draft: false
categories: ["data-engineering", "python"]
tags: ["spark", "data-engineering", "python"]
---

## 1. What is spark and why would use use it?

* Spark is a smooth framework for working with big data, i.e. [hdfs](https://tomis9.github.io/post/hadoop);

* it can be accessed from Python, R, scala (spark is actually written in scala) and java;

* it is probably the most popular big data tool nowadays for data scientists.

## 2. A few "Hello World" examples

### Prerequisites

#### Installation of pyspark
In this tutorial we will work on a development python version of spark. You can istall it with:

```{python}
sudo pip3 install pyspark
```

You will not notice any difference between this spark version and a production version installed on a cluster, except for performance.

#### Example file

During the tutorial we will work on an example file:

example.csv:
```
UserCode,GroupCode
1051,123
1150,234
1152,345
```

### Initialisation

First, let's import SparkSession from pyspark.sql module, which enables us to ceoonect with spark from python. If you do not specifically provide details of spark installation on cluster/server/laptop you use, pyspark will use it's own, development spark session.

```
from pyspark.sql import SparkSession
```

In order to work with spark, we have to initialize the spark session and give it a nice name
```
spark = SparkSession \
    .builder \
    .appName("Python Spark basic example") \
    .getOrCreate()
```
or an ugly name. Up to you.

From now you can watch your tasks execution in a web interface available at http://127.0.0.1:4040.

### Basic information about dataframe

Let's read some data to spark and enjoy it's incredibly fast performance.

```
df = spark.read.csv('example.csv', header=True)
```

Printing columns' names and datatypes are not spark jobs yet, co you can not observer their execution in spark's web interface.
```
df.printSchema()
df.columns
```

These are spark jobs, so open up web interface and check out the "jobs" tab.

head of our dataframe
```
df.head()
```

number of rows
```
df.count()
```

some statistics of our dataframe, similar to R's `summary`
```
df.describe().show()
```

### SQL queries

How about being able to use sql to query our table?

First we have to decalre our data as a table.

```
df.createOrReplaceTempView("example")
```
It's time to check the "SQL" tab in GUI.

And the  we can move on to sql.

General queries
```
df2 = spark.sql("select * from example")
df2.show()
```

### Query expressions

Selecting
```
user_code3 = df.select('UserCode', 'UserCode', 'Usercode')
user_code3.show()
```

Filtering
```
filtered = df.filter("UserCode = 1051")
filtered.show()
```


### Saving dataframes

In general, you will save your data to parquet files, as they are optimised for reading from writing to spark.
```
df.write.parquet('file.parquet')
```

But you can always save the data to csv.
```
df.write.csv('file.csv')
```
#### saving tips & tricks

You will often want to write your files in a specific way. Here is a list of the most popular parameters:
```
df.coalesce(1). \
    write. \
    mode('overwrite'). \
    option("header", "true"). \
    csv("result.csv")
```

and their descriptions:

* use partition or save to one file

* let's move to the writing file part

* overwrite if exists

* write the header

* file extension

## 3. Useful links

* a nice introductory article https://dzone.com/articles/introduction-to-spark-with-python-pyspark-for-begi


## 4. Subjects still to cover

* MLlib (TODO)

* importing table directly from database - jdbc (TODO)

* communication with hdfs (TODO)

* sparkR (TODO)
