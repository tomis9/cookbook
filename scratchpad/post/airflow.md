---
title: "airflow"
date: 2018-11-09T23:01:35+01:00
draft: true
categories: ["python", "data-engineering"]
tags: ["draft"]
---

## 1. What is airflow and why would you use it?

* airflow lets you manage your dataflow as a graph (direct acyclic graph or DAG), wchich consists of separate Tasks, and schedule them

*Wait*, you may say, *I can do that with cron!*

Yes, you can, but with airflow:

* you can easily divide your app into smaller tasks and monitor their reliability and execution duration;

* the performance is more transparent;

* simple rerunning;

* simple alerting with emails;

* as the pipelines' definitions are kept in code, you can generate them, or even let the user do it;

* you can (and should!) keep your pipelines' code in a git repository;

* you keep the logs in one place. Unless you use ELK stack, then you don't use this functionality;

* integration with [mesos](https://tomis9.github.io/mesos), which I never used, but you can.

Convinced? ;)

## Installation

```{python}
pip install airflow
```

Not very complicated.

Installation will create a directory named ~/airflow with a structure:

```
├── airflow.cfg
├── airflow.db
├── connections.py
├── dags
│   ├── check_ssh.py
│   ├── __pycache__
│   │   ├── check_ssh.cpython-35.pyc
│   │   └── sql_test.cpython-35.pyc
│   └── sql_test.py
├── logs
│   ├── check_mysql
│   │   └── test_sql
│   │       ├── 2018-10-28T12:00:00
│   │       ├── 2018-10-29T21:43:14.051520
│   │       └── 2018-10-29T21:46:45.016707
│   ├── check_ssh
│   │   │   ├── 2018-10-28T12:00:00
│   │   │   └── 2018-10-29T21:44:04.345091
│   │   ├── task1
│   │   └── task_check_ssh
│   └── scheduler
└── unittests.cfg
```


## 3. Useful links

[tutorial 1](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/)

[tutorial 2](https://airflow.apache.org/tutorial.html)
