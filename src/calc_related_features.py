import pandas as pd
import time
# import numpy as np
# from data_exploration import read_dataset, df_from_pickle
# from Definitions import *
import logging

LOG_FORMAT = "%(levelname)s %(asctime)s in %(funcName)s line %(lineno)d- %(message)s"
logging.basicConfig(filename="log/log.log", level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

logger.critical("")
logger.critical("---------------------------------------------------------------------")

# path = 'dataset/1M/Pickle/'
path = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\\'
path = r''
# path = 'dataset/similar_pages/20210602/'

file_name1 = "mini_url_indices.json"
file_name1 = "1M_url_indices_distances_similarities.json"
# file_name1 = "1M_url_indices_distances_similarities.pkl"
file_name2 = "linkChangeRate_dataset-SVflat.pkl"
file_name2 = "1M_all_with_diffs_avg_linkChangeRate.pkl"
# file_name2 = "ALL_DATA_0713_SINGLE_FILE.pkl"
# file_name2 = r"F:\University\PHD\Postdoc\Me\My Postdoc\Sent\20210113 991024 Netherland University of Twente\StartWork\Dataset\linkChangeRate_dataset-SVflat.pkl"

df1 = pd.read_json(path + file_name1, lines=True)
# df1 = pd.read_pickle(path + file_name1)
logger.debug("file: " + file_name1 + " successfully loaded")
# df2 = pd.read_json(path + file_name2, lines=True)
df2 = pd.read_pickle(path + file_name2)
logger.debug("file: " + file_name2 + " successfully loaded")
# df2 = df2.drop(['SV' + str(i) for i in range(192)], axis='columns')
df2 = df2.set_index('url')
# df2 = read_dataset(path + file_name2, ['url'] + features + target)


# atts = ['avg_contentLength', 'avg_textSize', 'avg_textQuality', 'avg_numInternalOutLinks', 'avg_numExternalOutLinks',
#         'avg_numInternalInLinks', 'avg_numExternalInLinks', 'avg_pathDepth', 'avg_domainDepth', 'avg_trustRank']


atts = [f'avg_diffInternalOutLinks-{i}' for i in range(1, 9)] + [f'avg_diffExternalOutLinks-{i}' for i in range(1, 9)]
atts += [f'avg_numInternalInLinks-{i}' for i in range(1, 9)] + [f'avg_numExternalInLinks-{i}' for i in range(1, 9)]
atts += ['avg_numInternalInLinks', 'avg_numExternalInLinks', 'avg_diffInternalOutLinks', 'avg_diffExternalOutLinks']

dic1 = {}
for att in atts:
    df2[att] = 0

df2[atts] = df2[atts].astype(float)
df2['isValid'] = False

cnt = 0
print("Datasets are ready")
print("time is: " + str(time.time()))
start1 = time.time()
start2 = time.time()
for index, row in df1.iterrows():
    try:
        cnt += 1
        if not row['is_valid']:
            continue
        url = row['url']
        similar_page_indices = row['indices']
        similarities_weights = row['similarities']
        row_indexer = df1.loc()
        for att in atts:
            dic1[att] = 0
    except Exception as e:
        err = getattr(e, 'message', repr(e))
        logger.critical(err)
        print(err)
    # avgCL = avgTS = avgTQ = avgNIntOL = avgNExtOL = avgNIntIL = avgNExtIL = 0
    for i in range(len(similar_page_indices)):
        try:
            idx = similar_page_indices[i]
            weight = similarities_weights[i]
            similar_row = row_indexer[idx - 1]
            similar_url = similar_row['url']
        except Exception as e:
            err = getattr(e, 'message', repr(e))
            logger.critical(err)
            print(err)
        try:
            row2 = df2.loc[[similar_url]]
            for att in atts:
                dic1[att] += (weight * row2[att.replace('avg_', '')].values[0])
        except Exception as e:
            err = getattr(e, 'message', repr(e))
            logger.critical(err)
            print(err)
    try:
        for att in atts:
            df2.at[url, att] = round(dic1[att], 3)
            # df2.at[url, att] = dic1[att] if ('Qual' in att) or ('trust' in att) else round(dic1[att])  # textQuality is float
            # df2.at[url, att] = round(dic1[att])
    except Exception as e:
        err = getattr(e, 'message', repr(e))
        logger.critical(err)
        print(err)
    try:
        df2.at[url, 'isValid'] = True
        if cnt % 100 == 0:
            stop1 = time.time()
            c = str(cnt / 100)
            print(c + ": duration for 100 instances: " + str(round(stop1 - start1, 2)))
            logger.debug(c + ": duration for 100 instances: " + str(round(stop1 - start1, 2)))
            start1 = time.time()
    except Exception as e:
        err = getattr(e, 'message', repr(e))
        logger.critical(err)
        print(err)
    # break
try:
    df2 = df2.reset_index()
    # df2.to_json(path + '1M_all_with_avg_atts.json', lines=True, orient='records')
    df2.to_csv(path + 'All_new.csv', index=False, header=True)
except Exception as e:
    err = getattr(e, 'message', repr(e))
    logger.critical(err)
    print(err)
print("finished")
# print(type(row['url']), row['url'])

# with open(path + file_name) as fi:
#     lines = fi.readlines()
#     for line in lines:
#         features.where(col('url').eqNullSafe(line))
#         features.where
#
# print("data ready: total instances are: " + str(features.count()))
# features = features.where(~col('semanticVector').contains('not set'))
