from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)

lines = sc.textFile('hello_file')

print(lines.count())
print(lines.first())
