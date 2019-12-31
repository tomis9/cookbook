from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("saprk.come.config.option", "som-value") \
    .getOrCreate()

df = spark.read.csv('mtcars.csv', header=True)
df = df.withColumnRenamed("_c0", "car")

df.printSchema()
df.count()
df.show()

df.coalesce(1).write.parquet("file.parquet")

