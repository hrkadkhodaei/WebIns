import findspark

findspark.init()
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, udf, concat_ws, row_number, lit, \
    collect_set  # , to_timestamp, lag, unix_timestamp, lit
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType

from re import sub
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from urllib.parse import urlparse
from time import strptime

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
import heapq
import numpy as np
from pyspark.sql.window import Window
from scipy.spatial import distance

path = r"F:\University\PHD\Postdoc\Me\My Postdoc\Sent\20210113 991024 Netherland University of Twente\StartWork\Source Code\MyPythonWI\temp\\"
file_name1 = "mini_url_indices.json"
file_name2 = "ALL_DATA_0713_SINGLE_FILE.json"
# file_name = "mini.json"
df1 = pd.read_json(path + file_name1, lines=True)
df2 = pd.read_json(path + file_name2, lines=True)


path = 'dataset/similar_pges/20210602/'

dic1 = {'sumCS': 0}
for index, row in df1.iterrows():
    url = row['url']
    similar_page_indices = row['indices']
    row_indexer = df1.iloc()
    for idx in similar_page_indices:
        similar_row = row_indexer[idx - 1]
        similar_url = similar_row['url']
        df2.where()
        # f1 = features.where(col('url').eqNullSafe(similar_url))
        # f2 = f1.select('contentLength').collect()
        # ToDo sum with features
        # dic1['sumCS'] += sumCS
    print(type(row['url']), row['url'])

with open(path + file_name) as fi:
    lines = fi.readlines()
    for line in lines:
        features.where(col('url').eqNullSafe(line))
        features.where

print("data ready: total instances are: " + str(features.count()))
features = features.where(~col('semanticVector').contains('not set'))
