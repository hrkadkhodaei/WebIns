# import findspark
# findspark.init()
from pyspark.sql import SparkSession
# from pyspark.sql.functions import udf, row_number, lit  # , to_timestamp, lag, unix_timestamp, lit
import pyspark.sql.types as T
from pyspark.sql.functions import col
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import json

#    time spark-submit --master yarn --deploy-mode client --executor-cores 2 --conf spark.dynamicAllocation.minExecutors=25 --conf spark.dynamicAllocation.initialExecutors=25 --driver-memory 6G --executor-memory 6G features_static-target_newOutlinks.py

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


def getCleanText(vector):
    vector = vector.replace('[', '').replace(']', '')
    return vector


udf_getCleanText = F.udf(getCleanText, T.StringType())


def parse_array_from_string(x):
    res = json.loads(x)
    return res


retrieve_array = F.udf(parse_array_from_string, T.ArrayType(T.FloatType()))

w = Window().orderBy(F.lit('A'))

# fileNameList = r"F:\University\PHD\Postdoc\Me\My Postdoc\Sent\20210113 991024 Netherland University of Twente\StartWork\Dataset\1M.2020-09-14-aa_2\mini.txt"
#fileNameList = r"F:\University\PHD\Postdoc\Me\My Postdoc\Sent\20210113 991024 Netherland University of Twente\StartWork\Dataset\1M.2020-09-14-aa_2\1M.2020-09-14-aa_2"
fileNameList = [r"/data/doina/WebInsight/2020-07-13/1M.2020-07-13-a" + chr(i) + ".gz" for i in range(97, 113)]
features = spark.read.json(fileNameList) \
    .where(
    (col('url').isNotNull()) &
    (col('fetch.contentLength') > 0) & ~col('fetch.semanticVector').contains('not set')) \
    .select('url', udf_getCleanText(col('fetch.semanticVector')).alias('semanticVector')) \
    .cache()
# .withColumn('id', row_number().over(w)) \
print("data ready: total instances are: " + str(features.count()))

features = features.repartition(1)
#f2 = f1.withColumn("ss1", retrieve_array(F.col("semanticVector")))
#f3 = f1.na.replace('"', '')
print("Start writing to HDFS")
f_write = "dataset/similar_pages/1M_url_sv_Spark_single_file.csv"
features.write.csv(f_write)
print("finished")
