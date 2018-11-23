---
title: "airflow"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["python", "data-engineering"]
tags: ["draft"]
---

## 1. What is airflow and why would you use it?

* airflow lets you manage your dataflow as a graph (direct acyclic graph or DAG), wchich consists of separate Tasks, and schedule them

*Wait*, you may say, *I can do that with cron!*

Yes, you can, but:

#### Installation

```
pip install airflow
```

or you can use virtualenv. Or pyenv + virtualenv, which I recommend.

## 3. Useful links

[tutorial 1](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/)

[tutorial 2](https://airflow.apache.org/tutorial.html)
