'''
example.csv:
UserCode,GroupCode
1051,123
1150,234
1152,345
1154,456
'''

'''
total basics
'''
from pyspark.sql import SparkSession

# https://www.youtube.com/watch?v=C2dxoCMhcFQ

# in order to work with spark, we have to initialize as spark session and give
# it a name
spark = SparkSession \
    .builder \
    .appName("Python Spark basic example") \
    .getOrCreate()

# now you can watch your tasks with a nice GUI available at http://127.0.0.1:4040

df = spark.read.csv('example.csv', header=True)

# prints columns' names and datatypes - these are not spark jobs yet
df.printSchema()
df.columns

# these are spark jobs, you may watch their execution statistics in GUI in
# "jobs" tab
df.head()

df.count()

# but... this reveals some information:
df.describe().show()


# now we want to be able to use sql to query our table
# but first we have to write our data as a table

df.createOrReplaceTempView("example")
# check the "SQL" tab in GUI!

df2 = spark.sql("select * from example")
df2.show()

user_code3 = df.select('UserCode', 'UserCode', 'Usercode')
user_code3.show()

filtered = df.filter("UserCode = 1051")
filtered.show()

# saving data to common file extensions

df.write.parquet('file.parquet')

df.write.csv('file.csv')

# saving to one file
df.coalesce(1).write(...)

df.coalesce(1). \
    write. \
    mode('overwrite'). \
    option("header", "true"). \
    csv("result.csv")

## use partition or save to one file
## let's move to the writing file part
## overwrite if exists
## write the header
## file extension

# a nice introductory article https://dzone.com/articles/introduction-to-spark-with-python-pyspark-for-begi

# TODO importing table directly from database
