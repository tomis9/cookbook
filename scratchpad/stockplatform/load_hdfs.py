from pyspark.sql import SparkSession
import pandas as pd
import os

sparkSession = SparkSession.builder.appName("test_app").getOrCreate()

filename = "20170830.csv"
path_file = os.path.join("internet", filename)
d = pd.read_csv(path_file)
df = sparkSession.createDataFrame(d)

shortname = os.path.splitext(filename)[0]
path = os.path.join("hdfs://127.0.0.1:9000/user/tomek/stock", shortname)
df.write.save(path, format="parquet")
