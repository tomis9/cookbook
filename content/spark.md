---
title: "spark"
date: 2018-11-23T12:58:49+02:00
draft: false
categories: ["data-engineering", "python", "R"]
tags: ["spark", "data-engineering", "python", "R"]
---

<center>

**Contents:**

[1. What is spark and why would you use it?](#what) 

[2. A few "Hello World" examples](#hello) 

[pyspark](#pyspark)

[sparklyr](#sparklyr)

[SparkR](#sparkr)

[3. Useful links](#useful)

[4. Subjects still to cover](#subjects)

</center>

## 1. What is spark and why would use use it? {#spark}

* Spark is a smooth framework for working with big data, i.e. [hdfs](https://tomis9.github.io/hadoop);

* it can be accessed from Python, R, scala (spark is actually written in scala) and java;

* it is probably the most popular big data tool nowadays for data scientists.

## 2. A few "Hello World" examples {#hello}
### a) pyspark {#pyspark}
#### Prerequisites

##### Installation of pyspark
In this tutorial we will work on a development python version of spark. You can istall it with:

```{python}
sudo pip3 install pyspark
```

You will not notice any difference between this spark version and a production version installed on a cluster, except for performance.

##### Example file

During the tutorial we will work on an example file:

example.csv:
```
UserCode,GroupCode
1051,123
1150,234
1152,345
```

Clearly not a big data case.

#### Initialisation

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

#### Basic information about dataframe

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

#### SQL queries

How about being able to use sql to query our table? First we have to declare our data as a table in spark.

```
df.createOrReplaceTempView("example")
```
It's time to check the "SQL" tab in GUI and then we can move on to sql. An example query:

```
df2 = spark.sql("select * from example")
df2.show()
```

#### Query expressions

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


#### Saving dataframes

In general, you will save your data to parquet files, as they are optimised for reading from and for writing to spark.
```
df.write.parquet('file.parquet')
```

But you can always save the data to csv.
```
df.write.csv('file.csv')
```

##### saving tips & tricks

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

### b) sparklyr (spark + dplyr) {#sparklyr}

There are two popular R libraries, which enable you to connect to spark from R: SparkR and sparklyr. I found sparklyr much nicer, as it is compatible with all the fancy functions from [dplyr](https://tomis9.github.io/tidyverse), which makes manipulating dataframes familiar and easy (+ 1 big point for dplyr in it's fight against [data.table](https://tomis9.github.io/data.table)).

Here's a quick example of reading a parquet file from [hdfs](https://tomis9.github.io/hadoop) into spark.

First, check if you use a proper version of spark:
```
Sys.getnev("SPARK_HOME")
```

If the function above returns an empty string, you may set the path of the spark installation manually with:

```
Sys.setenv(SPARK_HOME="<spark_home_path>")
```

so that R will use production version of spark installed on the cluster. If you do not set it, sparklyr will use the default, testing version of spark (as long as you installed it with `spark_install()`) or will throw an error if it does not find any available version of spark.

Let's get to the point:
```{r}
Sys.setenv(SPARK_HOME="<spark_home_path>")

library(dplyr)
library(sparklyr)

sc <- spark_connect(master = "local[*]")

parquet_path <- "hdfs:///user/..."
d <- spark_read_parquet(sc, name = "my_df_name", path = parquet_path)
```

> Tip: you can check the details of spark conenction by typing `sc`.

After reading the dataframe to the variable `d`, you can run any `dplyr` function on this dataframe, e.g.:

```
d %>% count()
```

You will find more useful information on [datacamp sparklyr course](https://www.datacamp.com/courses/introduction-to-spark-in-r-using-sparklyr).

### c) SparkR {#sparkr}

A short example of setting up SparkR:

```{r}
SPARK_HOME <- ""
lib.loc <- file.path(SPARK_HOME, "R", "lib")
Sys.setenv(SPARK_HOME = SPARK_HOME) 
# library(rJava)  # may be necessary
library(SparkR, lib.loc = lib.loc)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))
```

and reading a parquet file from hdfs:

```{r}
parquet_path <- "hdfs:///user/..."
df <- read.parquet(parquet_path)
# collect() downloads SparkDataFrame to local memory; beware of function 
# names' conflict: tidyverse::collect() and SparkR::collect()
SparkR::collect(df)  
```

[Here](https://spark.apache.org/docs/latest/sparkr.html) you can find a SparkR programming guide.

## 3. Useful links {#useful}

* [a nice introductory article](https://dzone.com/articles/introduction-to-spark-with-python-pyspark-for-begi)

* [a goo book on pyspark](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf)

* [SparkR vs sparklyr](https://eddjberry.netlify.com/post/2017-12-05-sparkr-vs-sparklyr/)

## 4. Subjects still to cover {#subjects}

* MLlib (TODO)

* importing table directly from database - jdbc (TODO), both for pyspark and sparklyr (https://rdrr.io/cran/sparklyr/man/spark_read_jdbc.html)

[https://stackoverflow.com/questions/45420958/how-to-use-a-predicate-while-reading-from-jdbc-connection]

[https://stackoverflow.com/questions/41966814/transfer-data-from-database-to-spark-using-sparklyr]

* pyspark communication with hdfs (TODO)

* spark-submit (TODO)

* sparklyr + sql (TODO) [https://spark.rstudio.com/]

* (TODO) sparkR and sparklyr comparison [https://eddjberry.netlify.com/post/2017-12-05-sparkr-vs-sparklyr/]

* collect() (TODO)
