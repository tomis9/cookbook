---
title: "data.table"
date: 2017-03-02T15:45:27+01:00
draft: false
categories: ["R"]
tags: ["R", "data.table"]
---






## 1. What is data.table and why would you use it? 

* data.table is an R packge which let's you work on tabular datasets quickly and easily;

* comparing to base R or [dplyr](http://tomis9.com/tidyverse/#/dplyr) it's significantly faster;

* data.table has a concise and SQL-like syntax.

## 2. Basic functionalities 

### Creating a data.table 


```r
library(data.table)

df <- data.frame(x = c("b","b","b","a","a"),
                 v = rnorm(5))

dt <- data.table(x = c("b","b","b","a","a"),
                 v = rnorm(5))
```

is exactly the same as creating a data.frame. The method `as.data.table()` works exaclty the same as `as.data.frame()`.

### Filtering 

Let's create a sample dataset first, baased on mtcars table: 

```r
sample_dataset <- as.data.table(datasets::mtcars)
```
Yes, you already have *datasets* package installed.


```r
sample_dataset[cyl == 6]
```

```
##     mpg cyl  disp  hp drat    wt  qsec vs am gear carb
## 1: 21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
## 2: 21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
## 3: 21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
## 4: 18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
## 5: 19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
## 6: 17.8   6 167.6 123 3.92 3.440 18.90  1  0    4    4
## 7: 19.7   6 145.0 175 3.62 2.770 15.50  0  1    5    6
```

What happened? We chose only those cars, which have 6 cylinders. Data.table already knew that we mean a column named `cyl`, not an object from outside of the square brackets.

### Selecting columns 


```r
sample_dataset[, .(mpg, cyl, disp)][1:5]
```

```
##     mpg cyl disp
## 1: 21.0   6  160
## 2: 21.0   6  160
## 3: 22.8   4  108
## 4: 21.4   6  258
## 5: 18.7   8  360
```

What happened here?

* we used a special fucntion from data.table package: `.()`, which works just like vectors, but inside data.tables square brackets it treats columns as separate objects, so to work on column `mpg`, you simply type `mpg` instead of `"mpg"` or `sample_dataset$mpg`

* in square brackets we first provided a comma, as the first argument is always filtering. If we want to skip filtering, we simply write a comma;

* we chose the first five elements from our dataset. We could write even more square brackets after the whole statement and it would work as a pipe, but this would be too dplyr-ish.

### Grouping 


```r
sample_dataset[, .(mean_mpg = mean(mpg), count = .N), cyl]
```

```
##    cyl mean_mpg count
## 1:   6    19.74     7
## 2:   4    26.66    11
## 3:   8    15.10    14
```
* group by is the last statement inside the square brackets. In the example above, we group by column cyl;

* in the select clause we do exactly the same thing as in SQL statements;

* `.N` means *number of* or simply *count*.

### Reading and writing data 

data.table has the fastest reading and writing functions available in R. These are:


```r
fwrite(x = mtcars, file = 'mtcars.csv')
ds <- fread(file = 'mtcars.csv')
```

`fread` is pretty clever. It recognises if a file has headers, columns datatypes and separators. What I like the most in these functions is that I literally *never* have to provide any details about the file. Object and file names are always enough for data.table.

### Ordering data 

Very easy.


```r
sample_dataset[order(-gear, cyl)][1:5]
```

```
##     mpg cyl  disp  hp drat    wt qsec vs am gear carb
## 1: 26.0   4 120.3  91 4.43 2.140 16.7  0  1    5    2
## 2: 30.4   4  95.1 113 3.77 1.513 16.9  1  1    5    2
## 3: 19.7   6 145.0 175 3.62 2.770 15.5  0  1    5    6
## 4: 15.8   8 351.0 264 4.22 3.170 14.5  0  1    5    4
## 5: 15.0   8 301.0 335 3.54 3.570 14.6  0  1    5    8
```

### Updating data 

In order to update our dataset we use the `:=` operator:


```r
sample_dataset[mpg > 30, carb := -1]
```

### Creating a new column 

In the same way as updating we can create a new column in place:


```r
sample_dataset[, new_column := 0]
print(sample_dataset[1:5])
```

```
##     mpg cyl disp  hp drat    wt  qsec vs am gear carb new_column
## 1: 21.0   6  160 110 3.90 2.620 16.46  0  1    4    4          0
## 2: 21.0   6  160 110 3.90 2.875 17.02  0  1    4    4          0
## 3: 22.8   4  108  93 3.85 2.320 18.61  1  1    4    1          0
## 4: 21.4   6  258 110 3.08 3.215 19.44  1  0    3    1          0
## 5: 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2          0
```

But we don't have to do it in place:


```r
sample_dataset[, .(mpg, cyl, new_column2 = 0)][1:5]
```

```
##     mpg cyl new_column2
## 1: 21.0   6           0
## 2: 21.0   6           0
## 3: 22.8   4           0
## 4: 21.4   6           0
## 5: 18.7   8           0
```


## 3. Subjects still to cover: 

* `.I`(TODO)

* `.SD` + lapply (TODO)

* `merge()` (TODO)

* `setkey()` (TODO)
