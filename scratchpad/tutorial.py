from pyspark.sql import SparkSession

# https://www.youtube.com/watch?v=C2dxoCMhcFQ

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("saprk.come.config.option", "som-value") \
    .getOrCreate()

df = spark.read.csv('creditcard.csv')

# prints columns' names and datatypes
df.printSchema()

df.head()

df.count()

# prints nothing interesting
df.describe()

# but... this reveals some information:
df.describe().show()

# do we have any null values?
df.dropna().count()

# what if we had nas?
df.fillna(-1).show(5)
