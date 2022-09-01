import numpy as np
import pandas as pd
import pkg_resources
import itertools
import shap
import math
import logging
import multiprocessing
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotnine
from plotnine import *
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

plotnine.options.figure_size = (20, 10)

import xgboostlss
import pickle
from xgboostlss.model import *
# from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.NegativeBinomial import NBI
from xgboostlss.datasets.data_loader import load_simulated_data

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')

# distribution = Gaussian
distribution = NBI


def read_data(InEx, percent_zeros=0):
    print(datetime.now().strftime("%H:%M:%S"), "\n")
    # target = f'link{InEx}ChangeRate'
    target = f'diff{InEx}OutLinks'
    # atts += [f'{prefix}num{InEx}InLinks-{i}' for i in range(1, 9)]
    prefix = 'related_'
    a = [f'{prefix}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{prefix}diff{InEx}OutLinks']
    atts = [f'{prefix}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{prefix}diff{InEx}OutLinks']
    atts += [f'{prefix}linkInternalChangeRate'] + [f'{prefix}linkExternalChangeRate']
    atts += [f'related_avg_diff{InEx}OutLinks']
    # atts += [f'diff{InEx}OutLinks-{i}' for i in range(1, 9)]
    atts += [f'avg_diff{InEx}OutLinks']
    # atts += [f'diffInternalOutLinks', 'diffExternalOutLinks']

    path = r'D:\XGB-Lss\\'
    fn_orders = path + fr'{InEx}_orders-NGBCC.csv'
    fn = path + r'1M_Final.pkl'

    df_orders = pd.read_csv(fn_orders)
    test_urls = list(df_orders['URL'])
    url_orders = set(test_urls)

    df_orders_non_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] > 0]
    df_orders_zero = df_orders.loc[df_orders['num_new_outlinks_week_10'] == 0].sample(frac=percent_zeros,
                                                                                      random_state=123)
    url_orders_non_zero = list(df_orders_non_zero['URL']) + list(df_orders_zero['URL'])

    df = pd.read_pickle(fn)
    # df = df.loc[df[f'diff{InEx}OutLinks'] > 1].sample(n=1000, replace=False)
    # df = pd.read_csv(path + 'test2.csv')
    df[f'related_avg_diff{InEx}OutLinks'] = df[a].mean(axis=1)
    df = df[atts + ['url'] + [target]]
    url_all = set(df['url'])

    url_train = url_all - url_orders
    df.set_index('url', inplace=True)
    df_train = df.loc[url_train]
    df_train_zeros = df_train.loc[df_train[target] == 0].sample(frac=percent_zeros, random_state=123)
    df_train = df_train.loc[df_train[target] > 0]
    df_train = pd.concat([df_train, df_train_zeros], axis=0)
    X_train = df_train[atts]
    y_train = df_train[target]

    df_result = pd.DataFrame()
    df_result['url'] = test_urls

    df_test = df.loc[url_orders_non_zero]
    X_test = df_test[atts]
    y_test = df_test[target]
    return X_train, y_train, X_test, y_test


def read_data2(InEx):
    print(datetime.now().strftime("%H:%M:%S"), "\n")
    # target = f'link{InEx}ChangeRate'
    target = f'diff{InEx}OutLinks'
    # atts += [f'{prefix}num{InEx}InLinks-{i}' for i in range(1, 9)]
    prefix = 'related_'
    a = [f'{prefix}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{prefix}diff{InEx}OutLinks']
    atts = [f'{prefix}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{prefix}diff{InEx}OutLinks']
    atts += [f'{prefix}linkInternalChangeRate'] + [f'{prefix}linkExternalChangeRate']
    atts += [f'{prefix}avg_diff{InEx}OutLinks']
    atts += [f'diff{InEx}OutLinks-{i}' for i in range(1, 9)]
    atts += [f'avg_diff{InEx}OutLinks']
    # atts += [f'diffInternalOutLinks', 'diffExternalOutLinks']

    path = r'D:\XGB-Lss\\'
    fn = path + r'1M_Final.pkl'

    df = pd.read_pickle(fn)
    # df = df.loc[df[f'diff{InEx}OutLinks'] > 0]  # .sample(n=500000, replace=False, random_state=0)
    df[f'related_avg_diff{InEx}OutLinks'] = df[a].mean(axis=1)
    df = df[atts + ['url'] + [target]]
    X_train, X_test, y_train, y_test = train_test_split(df[atts], df[target], test_size=0.3, random_state=state)
    # X_train = df[atts]
    # y_train = df[target]
    #
    # df_result = pd.DataFrame()
    #
    # X_test = df[atts]
    # y_test = df[target]
    return X_train, y_train, X_test, y_test


state = 123
# train, test = load_simulated_data()
# path = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\\'
InExs = ['Internal', 'External']
# fn_orders = path + 'orders\\' + fr'{InEx}_orders-NGB.csv'
# fn = path + r'1M_Final.csv'
print("start reading data")
for InEx in InExs:
    df_res_stats = pd.DataFrame()
    for i in [0.4]: #np.arange(0, 0.3, 0.1):
        print(f"start {i} percent zeros")
        X_train, y_train, X_test, y_test = read_data(InEx, i)

        _, X_tune, _, y_tune = train_test_split(X_train, y_train, test_size=0.2, random_state=state)

        # train = pd.DataFrame()
        # testst = pd.DataFrame()
        n_cpu = multiprocessing.cpu_count()

        # X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
        # X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
        dtune = xgb.DMatrix(X_tune, label=y_tune, nthread=n_cpu)
        dtest = xgb.DMatrix(X_test, nthread=n_cpu)

        # distribution = Gaussian  # Estimates both location and scale parameters of the Gaussian simultaneously.
        distribution = NBI  # Estimates both location and scale parameters of the Gaussian simultaneously.
        distribution.stabilize = "None"  # Option to stabilize Gradient/Hessian. Options are "None", "MAD", "L2"
        quant_sel = [0.05, 0.95]

        np.random.seed(state)

        # Specifies the parameters and their value range. The structure is as follows: "hyper-parameter": [lower_bound, upper_bound]. Currently, only the following hyper-parameters can be optimized:
        params = {"eta": [1e-5, 1],
                  "max_depth": [1, 10],
                  "gamma": [1e-8, 40],
                  "subsample": [0.2, 1.0],
                  "colsample_bytree": [0.2, 1.0],
                  "min_child_weight": [0, 500]
                  }

        opt_params = xgboostlss.hyper_opt(params,
                                          dtrain=dtune,
                                          dist=distribution,
                                          num_boost_round=400,  # Number of boosting iterations.
                                          max_minutes=70,
                                          nfold=5,
                                          # Time budget in minutes, i.e., stop study after the given number of minutes.
                                          n_trials=None,
                                          # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                                          silence=False)  # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.

        print(opt_params)

        n_rounds = opt_params["opt_rounds"]
        del opt_params["opt_rounds"]

        # Train Model with optimized hyper-parameters
        xgboostlss_model = xgboostlss.train(opt_params,
                                            dtrain,
                                            dist=distribution,
                                            num_boost_round=n_rounds)

        with open(fr'd:/xgblss/model_{InEx}_zeros_{i}.pkl', 'wb') as f:
            pickle.dump(xgboostlss_model, f)
        # Number of samples to draw from predicted distribution
        # n_samples = 100

        # Using predicted distributional parameters, sample from distribution
        # pred_y = xgboostlss.predict(xgboostlss_model,
        #                             dtest,
        #                             dist=distribution,
        #                             pred_type="response",
        #                             n_samples=n_samples,
        #                             seed=123)
        # Using predicted distributional parameters, calculate quantiles
        # pred_quantiles = xgboostlss.predict(xgboostlss_model,
        #                                     dtest,
        #                                     dist=distribution,
        #                                     pred_type="quantiles",
        #                                     quantiles=quant_sel,
        #                                     seed=123)

        # Returns predicted distributional parameters
        pred_params = xgboostlss.predict(xgboostlss_model,
                                         dtest,
                                         dist=distribution,
                                         pred_type="parameters")

        # df_results = pd.DataFrame(pred_y)
        df_results = pd.DataFrame()
        df_results["url"] = X_test.index.values
        df_results["y_true"] = y_test.values
        # df_results = pd.concat([df_results, pred_quantiles], axis=1)
        df_results = pd.concat([df_results, pred_params], axis=1)

        df_results.to_csv(fr'd:/xgblss/{InEx}_zeros_{i}.csv', index=False, header=True)
        # df_res_stats
        # pred_y
