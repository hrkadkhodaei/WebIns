from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_replace, udf  # , to_timestamp, lag, unix_timestamp, lit
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

path = r'G:/WebInsight Datasets2/1M/1M pickle dataset 384323 instances doina/'
fn = 'numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8_without_sv.csv'
df = spark.read.option("header","true").option("inferSchema","true").csv(path + fn).cache()
df = df.withColumn("avg_diffExternalOutLinks",sum([col('diffExternalOutLinks-'+str(i+1)) for i in range(8)])/8.0)
df = df.withColumn("avg_diffInternalOutLinks",sum([col('diffInternalOutLinks-'+str(i+1)) for i in range(8)])/8.0)
df = df.withColumn('avg_diffOutLinks',sum([col('avg_diffExternalOutLinks'),col('avg_diffInternalOutLinks')]))
df = df.repartition(1)
fn_write = path + 'aaaaa.csv'
df.write.option('header','true').csv(fn_write)