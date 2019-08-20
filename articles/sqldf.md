---
title: "sqldf"
date: 2017-03-01T13:07:39+01:00
draft: false
image: "sqldf.jpg"
categories: ["R"]
tags: ["R", "SQL"]
---






## 1. What is sqldf and why would you use it?

`sqldf` package lets you treat any data.frame object as an sql table. You can write queries as if you were in a database. Pretty useless, comparing to, say, data.table or dplyr + tidyverse.

Despite it's uselessness, it works like a charm.


## 2. A few basic examples:

Load the package:

```r
library(sqldf)
```


Selecting specific columns:

```r
sqldf('select mpg, cyl, disp from mtcars where cyl = 6')
```

```
##    mpg cyl  disp
## 1 21.0   6 160.0
## 2 21.0   6 160.0
## 3 21.4   6 258.0
## 4 18.1   6 225.0
## 5 19.2   6 167.6
## 6 17.8   6 167.6
## 7 19.7   6 145.0
```

A simple where clause:

```r
sqldf('select * from mtcars where mpg > 21')
```

```
##     mpg cyl  disp  hp drat    wt  qsec vs am gear carb
## 1  22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
## 2  21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
## 3  24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
## 4  22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
## 5  32.4   4  78.7  66 4.08 2.200 19.47  1  1    4    1
## 6  30.4   4  75.7  52 4.93 1.615 18.52  1  1    4    2
## 7  33.9   4  71.1  65 4.22 1.835 19.90  1  1    4    1
## 8  21.5   4 120.1  97 3.70 2.465 20.01  1  0    3    1
## 9  27.3   4  79.0  66 4.08 1.935 18.90  1  1    4    1
## 10 26.0   4 120.3  91 4.43 2.140 16.70  0  1    5    2
## 11 30.4   4  95.1 113 3.77 1.513 16.90  1  1    5    2
## 12 21.4   4 121.0 109 4.11 2.780 18.60  1  1    4    2
```

Group by and order by:

```r
sqldf('select cyl, count(*) number from mtcars group by cyl order by cyl')
```

```
##   cyl number
## 1   4     11
## 2   6      7
## 3   8     14
```

Joins:

```r
customers <- data.frame(
  id_customers = c(1, 2, 3),
  name = c("Zidane", "Figo", "Beckham")
)
orders <- data.frame(
  id_orders = c(1, 2, 3, 4, 5),
  id_customers = c(1, 1, 2, 3, 1),
  products = c("fotball", "shoes", "t-shirt", "shoes", "energy drink")
)

sqldf("select * from customers c join orders o on c.id_customers = o.id_customers")
```

```
##   id_customers    name id_orders id_customers     products
## 1            1  Zidane         1            1      fotball
## 2            1  Zidane         2            1        shoes
## 3            1  Zidane         5            1 energy drink
## 4            2    Figo         3            2      t-shirt
## 5            3 Beckham         4            3        shoes
```

Subqueries:

```r
sqldf('select a.mpg from (select mpg, cyl, disp from mtcars where cyl = 6) as a')
```

```
##   a.mpg
## 1  21.0
## 2  21.0
## 3  21.4
## 4  18.1
## 5  19.2
## 6  17.8
## 7  19.7
```

As you can see, all the sql operations are available in `sqldf`.

## 3. Links

A longer `sqldf` tutorial is available [here](http://jasdumas.com/tech-short-papers/sqldf_tutorial.html#).