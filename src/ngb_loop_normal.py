import numpy as np
import pandas as pd
from ngboost.distns import Poisson
from ngboost.distns import Normal
# from scipy.stats import norm
from ngboost import NGBRegressor
import random
import definitions
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split




total_target = 'External'
target_feature = 'diff' + total_target + 'OutLinks'
seed = 0
which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
Y_mean_features = ['diff' + total_target + 'OutLinks-' + str(i + 1) for i in range(8)]
features = [f for fs in which_features for f in definitions.feature_sets[fs]]
random.seed(seed)
tuning_fraction, test_fraction = 1. / 4, 1. / 4
# tuning_fraction, test_fraction = 1. / 4, 1. / 4
path = r'~/dataset/1M/Pickle/'
path2 = r'~/src/WebIns/paper2/2/'
path = r''
path2 = r''
# path = r'G:/WebInsight Datasets2/1M/1M pickle dataset 384323 instances doina/'
fn = path + '1M_all_with_diffs_avg_linkChangeRate.pkl'
# fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\new\All_data3.csv'
Xy = pd.read_pickle(fn)
Xy = Xy[Xy['isValid'] == True]
train, test, = train_test_split(Xy, train_size=1 - test_fraction, shuffle=True, random_state=seed)
train = train.set_index('url')

df = pd.DataFrame()
df['url'] = test['url']
test = test.set_index('url')
X_test = test[features]
y_test = test[target_feature]
y_test = y_test.to_numpy()
df['y_true'] = y_test

train_zero = train[train[target_feature] == 0]
train_nonzero = train[train[target_feature] > 0]
train_zero['index'] = random.sample(range(0, len(train_zero)), len(train_zero))
gap = 20000
for i in range(0, len(train_zero), gap):
    if (len(train_zero) - i) < gap:
        i = len(train_zero)
    train_zero_selected = train_zero[train_zero['index'] < i].drop(['index'], axis='columns')
    # train_zero_aux.drop(['index'], axis='columns', inplace=True)
    train_final = pd.concat([train_nonzero, train_zero_selected]).sample(frac=1)
    # train_final.to_csv(path2 + 'internal_ngb_analysis_trainset_zeros_' + str(i) + '.csv', header=True, index_label='url')
    y_train = train_final[target_feature]
    y_train = y_train.to_numpy()
    X_train = train_final[features]
    print("Total # of X_train is: ", len(X_train), ' --> i is: ', i)
    ngb = None
    ngb = NGBRegressor(Dist=Normal, n_estimators=500, verbose=True, verbose_eval=5).fit(X_train, y_train)
    preds = ngb.predict(X_test)
    df['y_pred_zeros_' + str(i)] = preds
    print('-' * 80)

df.to_csv(path2 + 'all_external_Normal.csv', header=True, index=False)
print("finished")