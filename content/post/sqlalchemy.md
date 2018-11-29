---
title: "sqlAlchemy"
date: 2018-11-09T23:01:35+01:00
draft: false
categories: ["python", "SQL"]
tags: ["draft", "python", "SQL"]
---

## 1. What is sqlAlchemy and why would you use it?

* sqlAlchemy is a python module that enables you to connect to and use sql databases without writing code in sql;

* using sqlalchemy has several advantages: 

    * you will avoid using long sql strings in your code, which are difficult to read without syntax highlighting (unless you kepp you sql queries in separate sql files);

    * you are not vulnerable to sql injection attacks anymore;

    * you can map your sql tables to pythonic objects with ORM (Object Relational Mapper). Most probably you will not use this functionality as a data scientis, but as a back-end or full-stack developer, you would use it a lot. Thanks to ORM you ca create a whole logic of your database in Python, which produces much less mess at the end of the day;

* Basics of sqlAlchemy are very simple and you can just use it instead of pymysql or any other module you use. So, the last reason is: *why wouldn't you use it?*

## 2. Minimal examples

#### downloading data (querying, projection, selection, joins, subqueries)

In order to get the access to the database, you have to provide credentials.
```{python}
# happy people kkeep their mysql password im .my.cnf
creds_path = os.path.join(os.getenv("HOME"), '.my.cnf')
with open(creds_path) as c:
    creds = c.read().splitlines()
    user, password = (x[x.find("=")+1:] for x in creds[1:])
```

Import sqlalchemy by import sqlalchemy:
```{python}
import sqlalchemy
import os
```

A connection string consists of all the information needed to connect to a database you want.
```{python}
connection_string = 'mysql://' + user + ':' + password + '@localhost/test'
```

Here comes the tricky part. We `map` an sql table `t1` to a python variable `t1`. Metadata provides all the information about the structure of that table. Then we create a statement, which in this case is just `select * from`, but we can also add a `.where()` clause, joins, 'groupbys' etc. Having the statement created, we execute it and fetch the result. And disconnect from the database. Don't forget about it.
```{python}
# one way
meta = sqlalchemy.MetaData(connection_string)
t1 = sqlalchemy.Table('t1', meta, autoload=True)
stmt = t1.select()
result = stmt.execute()
print(result.fetchall())
connection.close()
```

Here's another way of retrieving data from a database, with our old friend, sql. As you can see, this solution does not differ much from what pymysql proposes.
```{python}
# or another
engine = sqlalchemy.create_engine(connection_string)
connection = engine.connect()
result = connection.execute("select * from t1")
print(result.fetchall())
connection.close()
```

#### uploading data (insert, update)

#### running queries, e.g. calling a procedure

## 3. Useful links

http://danielweitzenfeld.github.io/passtheroc/blog/2014/10/12/datasci-sqlalchemy/

https://medium.freecodecamp.org/sqlalchemy-makes-etl-magically-easy-ab2bd0df928

https://medium.com/hacking-datascience/sqlalchemy-python-tutorial-abcc2ec77b57

<iframe width="1620" height="595" src="https://www.youtube.com/embed/rTVDlBMaI7I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

https://it.toolbox.com/blogs/garyeastwood/why-every-data-scientist-should-learn-sqlalchemy-103118

https://github.com/dropbox/PyHive

https://www.safaribooksonline.com/library/view/mastering-geospatial-analysis/9781788293334/44f54df4-f6c9-467d-8057-dee20c3d1f33.xhtml

https://datascienceplus.com/leveraging-hive-with-spark-using-python/
