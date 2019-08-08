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
##   account_no    Jan   Feb    Mar
## 1         71 10.062  7.55  8.232
## 2         72  8.960 10.58 10.175
## 3         73  8.456 10.12  7.708
## 4         74  9.022 11.81 12.517
## 5         75 11.831 11.32 10.451
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
## 1          71   Jan  10.062
## 2          72   Jan   8.960
## 3          73   Jan   8.456
## 4          74   Jan   9.022
## 5          75   Jan  11.831
## 6          71   Feb   7.550
## 7          72   Feb  10.576
## 8          73   Feb  10.117
## 9          74   Feb  11.812
## 10         75   Feb  11.316
## 11         71   Mar   8.232
## 12         72   Mar  10.175
## 13         73   Mar   7.708
## 14         74   Mar  12.517
## 15         75   Mar  10.451
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
##   account_no    Jan   Feb    Mar
## 1         71 10.062  7.55  8.232
## 2         72  8.960 10.58 10.175
## 3         73  8.456 10.12  7.708
## 4         74  9.022 11.81 12.517
## 5         75 11.831 11.32 10.451
```

## 3. Links

A pretty nice and much longer tutorial is available [here](https://seananderson.ca/2013/10/19/reshape/).
