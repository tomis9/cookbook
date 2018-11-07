import sqlalchemy
import os


# happy people kkeep their mysql password im .my.cnf
creds_path = os.path.join(os.getenv("HOME"), '.my.cnf')
with open(creds_path) as c:
    creds = c.read().splitlines()
    user, password = (x[x.find("=")+1:] for x in creds[1:])

connection_string = 'mysql://' + user + ':' + password + '@localhost/test'

# one way
meta = sqlalchemy.MetaData(connection_string)
t1 = sqlalchemy.Table('t1', meta, autoload=True)
stmt = t1.select()
result = stmt.execute()
print(result.fetchall())

# or another
engine = sqlalchemy.create_engine(connection_string)
connection = engine.connect()
result = connection.execute("select * from t1")
print(result.fetchall())

connection.close()
