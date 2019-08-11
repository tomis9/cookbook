---
title: "reshape2"
date: 2017-03-01T13:07:39+01:00
draft: false
image: "reshape.jpg"
categories: ["R"]
tags: ["R", "data.table", "dataframes"]
---






## 1. What is reshape2 and why would you use it?

`reshape2` is an R package that let's you change the shape of any dataframe, i.e. to pivot and to "unpivot".

Keep in mind that if your favourite R package for dataframes manipulation is [data.table](http://tomis9.com/data.table), functions *dcast* and *melt* are already in this package and work exactly the same as those in `reshape2`.

## 2. A "Hello World" example

In fact there are only two functions worth mentioning: *dcast*, which is equivalent to MS Excel pivot table, and *melt*, which does the opposite or unpivots a table.

An example dataframe:

```r
d <- data.frame(
  account_no = paste(rep(7, 5), 1:5, sep=""),
  Jan = rnorm(5, 10, 1),
  Feb = rnorm(5, 10, 2),
  Mar = rnorm(5, 10, 3)
)
print(d)
```

```
##   account_no    Jan    Feb    Mar
## 1         71 10.305 10.480 12.024
## 2         72  8.779  8.142  7.288
## 3         73  9.755 10.625  8.084
## 4         74  9.629 10.276  7.047
## 5         75 10.496  8.520 10.338
```

Transormation into a normalized table (unpivot):

```r
dn <- reshape2::melt(
  data = d, 
  id.vars = "account_no", 
  variable.name = "month", 
  value.name = "revenue"
)
print(dn)
```

```
##    account_no month revenue
## 1          71   Jan  10.305
## 2          72   Jan   8.779
## 3          73   Jan   9.755
## 4          74   Jan   9.629
## 5          75   Jan  10.496
## 6          71   Feb  10.480
## 7          72   Feb   8.142
## 8          73   Feb  10.625
## 9          74   Feb  10.276
## 10         75   Feb   8.520
## 11         71   Mar  12.024
## 12         72   Mar   7.288
## 13         73   Mar   8.084
## 14         74   Mar   7.047
## 15         75   Mar  10.338
```

And back to the previous format using a pivot:

```r
reshape2::dcast(
  data = dn, 
  formula = account_no ~ month, 
  value.var = "revenue"
)
```

```
##   account_no    Jan    Feb    Mar
## 1         71 10.305 10.480 12.024
## 2         72  8.779  8.142  7.288
## 3         73  9.755 10.625  8.084
## 4         74  9.629 10.276  7.047
## 5         75 10.496  8.520 10.338
```

## 3. Links

A pretty nice and much longer tutorial is available [here](https://seananderson.ca/2013/10/19/reshape/).
