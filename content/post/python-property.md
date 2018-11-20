---
title: "python @property"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["python"]
---

## @property

@property - pythonic way to use getters and setters.
[link](https://www.programiz.com/python-programming/property)

```
class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
```


