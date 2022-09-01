import numpy as np
import pandas as pd
from ngboost.distns import Poisson
from scipy.stats import norm
from ngboost import NGBRegressor
import random
import definitions
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pickle
from pathlib import Path
from datetime import datetime
import time
import sys
import math
import os

saved_dataset_folder = r'saved_datasets/'
result_folder = r'results/'
if not os.path.exists(saved_dataset_folder):
    os.mkdir(saved_dataset_folder)
if not os.path.exists('preds'):
    os.mkdir('preds')
if not os.path.exists('models'):
    os.mkdir('models')
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

distribution = 'Poisson'

index_args = ('-lr' in sys.argv) and (sys.argv.index('-lr') > 0)
learningRate = float(sys.argv[sys.argv.index('-lr') + 1]) if index_args else 0.01
print("lr = ", learningRate)

index_args = ('-target' in sys.argv) and (sys.argv.index('-target') > 0)
total_target = sys.argv[sys.argv.index('-target') + 1] if index_args else 'External'
print("target = ", total_target)

index_args = ('-n' in sys.argv) and (sys.argv.index('-n') > 0)
num_estimators = int(sys.argv[sys.argv.index('-n') + 1]) if index_args else 800
print("num_estimators = ", num_estimators)

use_scaling = True
min_samples_leaf = 2
max_depth = 5
gap = 20000
seed = 0
# config_str = total_target + ' ' + distribution + ' numEst_' + str(num_estimators) + ' LRate_' + str(
#     learningRate) + ' Scaling_' + str(use_scaling) + '_min_samples_leaf' ' gap_' + str(gap) + ' seed_' + str(seed)

config_str = f'{total_target} {distribution} numEst_{num_estimators} LRate_{learningRate}' \
             f' Scaling_{use_scaling} min_samples_leaf_{min_samples_leaf} max_depth_{max_depth} gap_{gap} seed_{seed}'
path = config_str + '.log'
sys.stdout = open(path, 'w')

print("time is: ", datetime.now().strftime("%H:%M:%S"))

print(config_str, flush=True)
start1 = time.time()
print("start time is: ", start1, flush=True)

target_feature = 'diff' + total_target + 'OutLinks'
which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
Y_mean_features = ['diff' + total_target + 'OutLinks-' + str(i + 1) for i in range(8)]
features = [f for fs in which_features for f in definitions.feature_sets[fs]]
random.seed(seed)
tuning_fraction, test_fraction = 1. / 6, 1. / 4
# tuning_fraction, test_fraction = 1. / 4, 1. / 4
path = r'~/dataset/1M/Pickle/'
path = r''

# path = r'G:/WebInsight Datasets2/1M/1M pickle dataset 384323 instances doina/'
fn = '1M_all_with_diffs_avg_linkChangeRate.pkl'
fn_external_order = total_target + '_orders-NGB.csv'
# fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\new\All_data3.csv'
Xy = pd.read_pickle(fn)
bb = np.asarray(Xy['url'])
Xy = Xy.set_index('url')
ext = pd.read_csv(fn_external_order)
ext = np.asarray(ext['URL'])
bb = set(bb) - set(ext)
train = Xy.loc[bb]
train = train[train['isValid'] == True]
test = Xy.loc[ext]
test = test.reset_index('url')

# Xy = Xy.sample(frac=1, random_state=seed)
# train, test, = train_test_split(Xy, train_size=1 - test_fraction, shuffle=True, random_state=seed)
# train.to_csv('train.csv', index=False, header=True)
# test.to_csv('test.csv', index=False, header=True)

# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# test, dev, = train_test_split(test, train_size=1 - tuning_fraction, shuffle=True, random_state=seed)

# train = train.set_index('url')
df_best_iter = pd.DataFrame()
df_max_iter = pd.DataFrame()
df_best_iter['url'] = test['url']
df_max_iter['url'] = test['url']
test = test.set_index('url')
X_test_orig = test[features]
y_test = test[target_feature]
y_test = y_test.to_numpy()
y_test = y_test.astype(np.float64)
df_best_iter['y_true'] = y_test

# X_dev = dev[features]
# y_dev = dev[target_feature]
# y_dev = y_dev.to_numpy()
# y_dev = y_dev.astype(np.float64)
# df['y_true'] = dev


train_zero = train[train[target_feature] == 0]
train_nonzero = train[train[target_feature] > 0]
train_zero.insert(0, 'index_zeros', random.sample(range(0, len(train_zero)), len(train_zero)))
# train_zero['index'] = random.sample(range(0, len(train_zero)), len(train_zero))
path_saved_models = Path(__file__).parent.resolve() / 'models'
for i in range(0, len(train_zero), gap):
    starttime = time.time()
    if (len(train_zero) - i) < gap:
        i = len(train_zero)
    train_zero_selected = train_zero[train_zero['index_zeros'] < i].drop(['index_zeros'], axis='columns')
    # train_zero_aux.drop(['index'], axis='columns', inplace=True)
    train_final = pd.concat([train_nonzero, train_zero_selected]).sample(frac=1, random_state=seed)
    train_final.to_csv(saved_dataset_folder + 'train dataset ' + config_str + ' zeros_' + str(i) + '.csv', header=True,
                       index_label='url')
    y_train = train_final[target_feature]
    y_train = y_train.to_numpy()
    y_train = y_train.astype(np.float64)
    X_train = train_final[features]
    X_test_final = pickle.loads(pickle.dumps(X_test_orig))
    if use_scaling:
        scalar = StandardScaler()
        scalar = scalar.fit(X=X_train)
        X_train = scalar.transform(X_train)
        X_test_final = scalar.transform(X_test_final)
        print("scaling finished", flush=True)
    print("Total # of X_train is: ", len(X_train), ' --> i is: ', i, flush=True)
    base_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=0.0,
        max_depth=max_depth,
        splitter="best",
        random_state=None,
    )
    sample_weights = np.sqrt(y_train)
    ngb = None
    ngb = NGBRegressor(Dist=Poisson, n_estimators=num_estimators + 2, Base=base_learner, learning_rate=learningRate,
                       verbose=True,
                       verbose_eval=20, random_state=seed).fit(X_train, y_train, X_val=X_test_final, Y_val=y_test,
                                                               sample_weight=sample_weights)
    path_saved_models2 = path_saved_models / ('model_' + config_str + ' zeros_' + str(i) + '.p')
    # model_file_path = 'models/model_' + config_str + ' zeros_' + str(i) + '.p'
    with path_saved_models2.open("wb") as f:
        pickle.dump(ngb, f)
    print("best loss obtained at iteration ", ngb.best_val_loss_itr, flush=True)
    preds_best_iter = ngb.predict(X_test_final, max_iter=ngb.best_val_loss_itr)
    preds_max_iter = ngb.predict(X_test_final)
    lst_r2 = []
    lst_mse = []
    df_preds_i = pd.DataFrame({'y_true': y_test, 'preds_best_iter': preds_best_iter, 'preds_max_iter': preds_max_iter})
    for j in range(1, num_estimators + 1, 50):
        # lst_iter.append(i)
        preds_iter = ngb.predict(X_test_final, max_iter=j)
        df_preds_i['preds_at_' + str(j) + '_iter'] = preds_iter
        # preds = model.predict(X_test)
        lst_r2.append(r2_score(y_test, preds_iter))
        lst_mse.append(mean_squared_error(y_test, preds_iter))
        print("j= ", j, flush=True)

    df_preds_i.to_csv(result_folder + 'partial result zeros_' + str(i) + ' best_iter_at_' +
                      str(ngb.best_val_loss_itr) + ' ' + config_str + '.csv', index=False, header=True)

    df_r2_mse_iteration = pd.DataFrame({'r2': lst_r2, 'mse': lst_mse})
    df_r2_mse_iteration.to_csv(saved_dataset_folder + 'r2 mse zeros_' + str(i) + ' best_iter_at_' +
                               str(ngb.best_val_loss_itr) + ' ' + config_str + '.csv', index=False,
                               header=True)
    df_best_iter['y_pred_zeros_' + str(i)] = preds_best_iter
    df_max_iter['y_pred_zeros_' + str(i)] = preds_max_iter
    print("time is: ", datetime.now().strftime("%H:%M:%S"),
          " --> Total time: ", time.time() - starttime, " seconds", flush=True)
    print('-' * 80, flush=True)

df_best_iter.to_csv(result_folder + 'best iteration df  ' + config_str + '.csv', header=True, index=False)
df_max_iter.to_csv(result_folder + 'max iteration df  ' + config_str + '.csv', header=True, index=False)

print("Total execution time for all iterations: ", time.time() - start1, " seconds", flush=True)

print("finished", flush=True)
