---
title: "lubridate"
date: 2017-03-03T13:46:45+01:00
draft: false
image: "lubridate.jpg"
categories: ["R"]
tags: ["lubridate", "R"]
---






## 1. What is lubridate and why would you use it?

* it's an R package that makes working with dates easy;

* because in basic, no-frills R working with dates may be a little bit daunting

## 2. A few "Hello World" examples


Load the package

```r
library(lubridate)
```


Convert a string to class `Date`:

```r
# the base way
d <- as.Date("2017-03-03")
class(d)
## [1] "Date"
# the lubridate way
d <- ymd("2017-03-03")
class(d)
## [1] "Date"
```

Extract year, month and day

```r
year(d)
## [1] 2017
month(d)
## [1] 3
day(d)
## [1] 3
```

You can also modify the date on the fly:

```r
year(d) <- 1410
month(d) <- 7
day(d) <- 15
print(d)
## [1] "1410-07-15"
class(d)
## [1] "Date"
```

Now, analogical to python's `datetime.now()`:

```r
# the base way
n <- Sys.time()
# the lubridate way
n <- now()
```

Extracting hour, minute and second

```r
hour(n)
## [1] 19
minute(n)
## [1] 24
second(n)
## [1] 57.48
```

Days of the week

```r
wday(d)  # numbering starts from Sunday! you can adjust it, read ?wday
## [1] 1
```

Adding / subtracting dates:

```r
print(d)
## [1] "1410-07-15"
d + years(1)  # yearS - plural form
## [1] "1411-07-15"
d - months(2)
## [1] "1410-05-15"
```
