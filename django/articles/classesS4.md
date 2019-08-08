---
title: "classes - S4"
date: 2018-02-04T12:00:04+01:00
draft: false
image: "classes.jpg"
categories: ["R"]
tags: ["R", "OOP"]
---






## 1. Why would you use OOP in R? 

Object oriented programming in R is unfortunately rather complicated comparing to Python (which seems to be the only reasonable alternative for data science programming). 

However, there are certain cases when OOP may come up helpful:

* when you write your own package and you want users to work on an object that your function returns (print, summary, plot);

* learning OOP in R is a good investment, because it let's you understand better how functions like print, plot etc. work.

## 2. Creating a new class in S4 

Let's define a new class:

```r
setClass(
  Class = 'Polygon', 
  slots = c(x = 'numeric', y = 'numeric'),
)
```

```
## Error in setClass(Class = "Polygon", slots = c(x = "numeric", y = "numeric"), : could not find function "setClass"
```
where `Polygon` is it's name and `slots` are it's attributes.

A new instance of class `Polygon`:


```r
w1 <- new('Polygon', x = 1:3, y = c(1, 1, 2))
```

```
## Error in new("Polygon", x = 1:3, y = c(1, 1, 2)): could not find function "new"
```

## 3. Attributes 

If you've ever wondered what `@` in R means, here's the answer:


```r
slotNames(w1)
```

```
## Error in slotNames(w1): could not find function "slotNames"
```

```r
w1@x  
```

```
## Error in eval(expr, envir, enclos): object 'w1' not found
```

```r
w1@y
```

```
## Error in eval(expr, envir, enclos): object 'w1' not found
```

It provides access to inctance attributes. You can also overwrite current attributes simply with:


```r
w1@x <- c(3, 2, 1)
```

```
## Error in w1@x <- c(3, 2, 1): object 'w1' not found
```

## 4. Methods 

Let's create a simple method `show` which prints a description of our object.


```r
setMethod(
  f = 'show',
  signature = c(object = 'Polygon'),
  function(object) {
    cat(sprintf('This is a polygon with %d sides.\n', length(object@x)))
    cat(sprintf('(%g, %g)', object@x, object@y))
    cat('\n')
  }
)
```

```
## Error in setMethod(f = "show", signature = c(object = "Polygon"), function(object) {: could not find function "setMethod"
```

```r
print(w1)
```

```
## Error in print(w1): object 'w1' not found
```

Yes, so far OOP in R looks extremely complicated and non-intuitive comparing to Python.

## 5. Inheritance 

A simple example of inheritance:

```r
setClass(
  'Triangle',
  contains = 'Polygon',
  validity = function(object)
    if (length(object@x) != 3) "a triangle has three sides" else TRUE
)
```

```
## Error in setClass("Triangle", contains = "Polygon", validity = function(object) if (length(object@x) != : could not find function "setClass"
```

```r
t1 <- new('Triangle', x=c(1, 2, 3), y=c(1, 1, 3))
```

```
## Error in new("Triangle", x = c(1, 2, 3), y = c(1, 1, 3)): could not find function "new"
```

```r
class(t1)
```

```
## Error in eval(expr, envir, enclos): object 't1' not found
```

```r
show(t1)
```

```
## Error in show(t1): could not find function "show"
```

You can see that a new class `Trinagle` ingerits `x` and `y` attributes and `show` method from class `Polygon`. It also adds a validity check, which produces an error if the number of vertices is different than 3.
