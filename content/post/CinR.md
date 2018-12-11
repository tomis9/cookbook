---
title: "C in R"
date: 2018-02-06T22:07:51+01:00
draft: false
categories: ["R"]
tags: ["R", "C"]
---

## 1. Why would you extend R with C language?

* some parts of your program may run too slowly. One of the possible solutions is to rewrite them into C;

* if you create a library and you want it to be extremely fast, you will probably end up writing most of your functions in C.;

* it's worth learning even the basic example, as most basic R functions are written in C.

## A "Hello World" example

Let's create a trivial C function, which sums elements of a given vector.

```
#include <R.h> 
#include <Rmath.h> 
#include <math.h>
#include <stdio.h>

void my_sum(double *x, int *n, double *s) {
    
    double sum = 0;

    for (int i=0; i<*n; i++) {
        sum = sum + x[i];
    }

    *s = sum;
}
```

As you can see:

* we do not provide a return clause; instead, we modify one of function's arguments, or, being more precise, a place in computer's memory at which this argument points;

* as every scalar is in fact a vector in R, the easiest way to pass elements from R to C is by using pointers.

Having the above function saved as `my_sum.c`, let's compile it with `R CMD SHLIB my_sum.c`. Comparing to a standard compilation with `gcc`, ,`R CMD SHLIB` prodides additional information about special R libraries directories. Compilation will create two new files: `my_sum.o` and `my_sum.so`. How do we use them in R?

my_func_test.R
```
dyn.load("my_sum.so")

x <- c(1, 2, 3, 4)
sum_ <- .C("my_sum", x = x, n = length(x), s = 0)

sum_
```

```
## $x
## [1] 1 2 3 4
## 
## $n
## [1] 4
## 
## $s
## [1] 10

```

As you can see, the result is a list of function arguments, but some of their values (in this case *s*) were updated by our C function.

## 3. Usful links

As 'C' and 'R' are not the easient terms to search on the internet, [here's](http://www.biostat.jhsph.edu/~rpeng/docs/interface.pdf) a short tutorial.
