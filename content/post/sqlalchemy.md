---
title: "sqlAlchemy"
date: 2018-11-09T23:01:35+01:00
draft: false
image: "sqlalchemy.jpg"
categories: ["python", "SQL"]
tags: ["python", "SQL"]
---

## 1. What is sqlAlchemy and why would you use it?

* sqlAlchemy is a python module that enables you to connect to and use sql databases without writing code in sql;

* using sqlAlchemy has several advantages: 

    * you will avoid using long sql strings in your code, which are difficult to read without syntax highlighting (unless you keep you sql queries in separate sql files);

    * you are not vulnerable to sql injection attacks anymore;

    * you can map your sql tables to pythonic objects with ORM (Object Relational Mapper). Most probably you will not use this functionality as a data scientist, but as a back-end or full-stack developer, you would use it a lot. Thanks to ORM you can create a whole logic of your database in Python, which produces much less mess at the end of the day;

* Basics of sqlAlchemy are very simple and you can just use it instead of pymysql or any other module you use. So, the last reason is: *why wouldn't you use it?*

## 2. Minimal examples

#### connection string

First of all, you need to connect to the database, so you have to:

* tell which database you want to connect to

* and provide your credentials.

Read credentials from a file. Happy people keep their mysql password in .my.cnf.

```{python}
creds_path = os.path.join(os.getenv("HOME"), '.my.cnf')
with open(creds_path) as c:
    creds = c.read().splitlines()
    user, password = (x[x.find("=")+1:] for x in creds[1:])
```

and then create a connection string.
```{python}
connection_string = 'mysql://' + user + ':' + password + '@localhost/test'
```

#### creating a table

```{python}
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, MetaData

meta = sqlalchemy.MetaData(connection_string)
test1 = Table(
    'test1', meta,
    Column('id', Integer, primary_key=True),
    Column('text', String(30))
)

engine = sqlalchemy.create_engine(connection_string)
meta.create_all(engine)
```

Great, you've just created you first table with sqlAlchemy! As you can see: 

* we haven't written a single line of code in SQL;

* we didn't have to provide a separate DDL file with SQL's create statement;

* we could have define more tables before the `create_all` execution;

* defining tables in sqlAlchemy does not differ much from creating them with plain SQL. 

#### 'talking about' the table in code

In sqlAlchemy we `map` an sql table (e.g. `test1`) to a python variable (e.g. `test1`) and then we refer to that variable when we want to select/insert/update/delete/do anything on that table. 

```{python}
test1 = sqlalchemy.Table('test1', meta, autoload=True)
```

In the example in section 'creating a table' we have already defined that variable, so we don't have to download it's metadata again, but in general case, we want to inform Python about the structure of the table.

After that, we simply execute the statement. Have a look:

#### inserting data to the table

There are several ways to do that.

First one:

```{python}
data = [{'id': 1, 'text': 'how'},
        {'id': 2, 'text': 'you'},
        {'id': 3, 'text': "doin'"},
        {'id': 4, 'text': '?'}]
stmt = test1.insert().values(data)
stmt.execute()
```
And the data is in the database.

You can also do this with pandas, but let's clear the table before reinserting the data:
```{python}
stmt = test1.delete()
stmt.execute()
```

You can always write execute and the end of line:
```{python}
test1.delete().execute()
```

```{python}
import pandas as pd
df = pd.DataFrame(data)
df.to_sql('test1', engine, if_exists='append', index=False)
```

And the data is in the database.

#### selecting data

```{python}
stmt = test1.select()
result = stmt.execute()
result.fetchall()
```
or

```{python}
test1.select().execute().fetchall()
```

or even

```{python}
pd.read_sql(test1.select(), engine)
```
#### using plain old SQL

You can still use plain old sql queries, if you don't feel comfortable with working on databases in Python:

```{python}
engine = sqlalchemy.create_engine(connection_string)
connection = engine.connect()
result = connection.execute("select * from test1;")
print(result.fetchall())

connection.close()
```

## 3. Useful or interesting links

There is a whole discussion whether sqlAlchemy is not a overkill for data science. 

* http://danielweitzenfeld.github.io/passtheroc/blog/2014/10/12/datasci-sqlalchemy/

* https://medium.freecodecamp.org/sqlalchemy-makes-etl-magically-easy-ab2bd0df928

* https://medium.com/hacking-datascience/sqlalchemy-python-tutorial-abcc2ec77b57

* https://it.toolbox.com/blogs/garyeastwood/why-every-data-scientist-should-learn-sqlalchemy-103118

* https://github.com/dropbox/PyHive

* https://www.safaribooksonline.com/library/view/mastering-geospatial-analysis/9781788293334/44f54df4-f6c9-467d-8057-dee20c3d1f33.xhtml

* https://datascienceplus.com/leveraging-hive-with-spark-using-python/
