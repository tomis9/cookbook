---
title: "decorators"
date: 2018-08-12T15:30:35+02:00
draft: true
image: "decorators.jpg"
categories: ["python"]
tags: ["python", "decorators"]
---

## 1. What are decorators and why would you use them?

* decorators in Python are special functions that take a function as an argument and slightly change it's behaviour, e.g. it's return value;

* you can write your own decorators, which is rather easy (I highly recommend [Fluent Python](http://shop.oreilly.com/product/0636920032519.do) as a reference)

* but there are already many useful decorators available in Python.

I am not going to describe here how to write your own decorator as, to be honest, I used them only twice in my career. In fact, I didn't have to do that, I just wanted to try them out ;)

## 2. Useful decorators

### @property

[@property](https://www.programiz.com/python-programming/property) - a pythonic way to use getters and setters.

```
class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
```
