---
title: "stringr and regex"
date: 2017-03-02T23:01:35+01:00
draft: true
categories: ["R"]
tags: ["R", "regex", "tidyvers"]
---






## 1. What is stringr and why would you use it?

* stringr is a simple and fast replacement for deafult R regex functions (grepl, gsub, substr etc.);

* but still traditional regex is available in every programming language;

* stringr is considered a member of [tidyverse]{http://tomis9.com/tidyverse} family.

To be honest, I don't use stringr, because it's basic R regex functions are more than enough for my everyday work.

## 2. A few basic operations

Let's load `stringr` library:

```r
library(stringr)
```

It's available on CRAN, so you can dowload it with `install.packages()`.

* string concatenation

```r
str_c("a", "b", "c")
```

```
## [1] "abc"
```

```r
paste0(c("a", "b", "c"), collapse = "")
```

```
## [1] "abc"
```

* numerb of characters

```r
str_length("abcdefg")
```

```
## [1] 7
```

```r
nchar("abcdefg")
```

```
## [1] 7
```

## 3. A few regex-like functions

* "is in"

```r
str_detect("The quick brown fox", "qui")
```

```
## [1] TRUE
```

```r
grepl("qui", "The quick brown fox")
```

```
## [1] TRUE
```

* substring

```r
str_sub("tomis9", 2, 4)
```

```
## [1] "omi"
```

```r
substr("tomis9", 2, 4)
```

```
## [1] "omi"
```

* replacing

```r
str_replace("John Doe", "Doe", "Kowalski")
```

```
## [1] "John Kowalski"
```

```r
gsub("Doe", "Kowalski", "John Doe")
```

```
## [1] "John Kowalski"
```

* trimming (removing excess spaces)

```r
str_trim("   abc  ")
```

```
## [1] "abc"
```

## 4. Useful links

* a very nice tutorial on stringr is available [here](https://cran.r-project.org/web/packages/stringr/vignettes/stringr.html);

* if you work a lot on text data, consider reading [Mastering Regular Expressions](http://shop.oreilly.com/product/9780596528126.do).
