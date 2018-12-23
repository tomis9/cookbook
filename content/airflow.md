---
title: "airflow"
date: 2018-08-14T11:51:12+02:00
draft: false
categories: ["data-engineering"]
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

## 2. Installation

```{python}
pip install airflow
```

Not very complicated.

## 3. Best practises

### Softlinks

You may feel tempted to create a git repository in your DAG folder, however this is not the best solution. It's much easier and more logical to keep your DAG file in a repo where your project lives and softlink it with

```
ln -s /path-to-your-project-repo/my_project_dag.py /home/me/airflow/dags/
```

### DAG names and DAG file names

* keep only one DAG in a file;

* DAG should have the same name as the file it's in;

* DAG's and file's name should begin with project's name.


## 3. Useful links

[a good tutorial](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/)

[another good tutorial](https://airflow.apache.org/tutorial.html)

Airflow's purpose is rather straightforward, so the best way to learn it is learning-by-doing.
