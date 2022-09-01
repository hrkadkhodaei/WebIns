from ngboost import NGBRegressor
from ngboost.distns import Poisson
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split

import pickle

scale_dict = {
    'contentLength': 'log',
    'textSize': 'symlog',
    'trustRank': 'symlog',
    'numInternalOutLinks': 'symlog',
    'numExternalOutLinks': 'symlog',
    'numInternalInLinks': 'symlog',
    'numExternalInLinks': 'symlog',
    'diffInternalOutLinks-1': 'symlog',
    'diffExternalOutLinks-1': 'symlog',
    'diffInternalOutLinks-2': 'symlog',
    'diffExternalOutLinks-2': 'symlog'
}

feature_label_dict = {
    'contentLength': 'page content size',
    'textSize': 'page text size',
    'numInternalOutLinks': '# internal outlinks',
    'numExternalOutLinks': '# external outlinks',
    'numInternalInLinks': '# internal inlinks',
    'numExternalInLinks': '# external inlinks',
    'textQuality': 'page text quality',
    'pathDepth': 'URL path depth',
    'domainDepth': 'URL domain depth',
    'trustRank': 'TrustRank',
    'diffInternalOutLinks': '# new internal outlinks',
    'diffExternalOutLinks': '# new external outlinks',
}
target_label_dict = {
    'diffInternalOutLinks': 'Prob(1+ new internal outlinks)',
    'diffExternalOutLinks': 'Prob(1+ new external outlinks)',
}
history_label_dict = dict([
    (f + "-" + str(i + 1), feature_label_dict[f] + " (-" + str(i + 1) + ")") for f in feature_label_dict.keys()
    for i in range(8)
])

pretty_label_dict = feature_label_dict.copy()
pretty_label_dict.update(target_label_dict)
pretty_label_dict.update(history_label_dict)

pretty_class_names = {
    0: "0",
    1: "1+"
}

static_page_features = ['contentLength', 'textSize', 'textQuality', 'pathDepth', 'domainDepth', 'numInternalOutLinks',
                        'numExternalOutLinks']  #
static_page_semantics = ["SV" + str(i) for i in range(192)]
static_network_features = ['numInternalInLinks', 'numExternalInLinks', 'trustRank']
static_network_features += ['related_' + f for f in static_page_features + static_network_features]
dynamic_network_features = ['numInternalInLinks-', 'numExternalInLinks-', 'trustRank-']
dynamic_page_features = ['contentLength-', 'textSize-', 'textQuality-', 'diffInternalOutLinks-',
                         'diffExternalOutLinks-']

feature_sets = {
    'SP': static_page_features,
    'v': static_page_semantics,
    'SN': static_network_features,
}

feature_sets.update(dict([
    ('DP' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_page_features])
    for i in range(8)
]))

feature_sets['DPRate'] = ['related_linkExternalChangeRate', 'related_linkInternalChangeRate']

feature_sets.update(dict([
    ('DN' + str(i + 1), [f + str(j + 1) for j in range(i + 1) for f in dynamic_network_features])
    for i in range(8)
]))  # 'DP/N8' will contain all -1, ... -8 dynamic page features


def int_to_categorical(x):
    if x == 0:
        return 0
    # elif x >= 1 and x <= 5:
    # 	return 1
    else:
        return 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    # _________________________________________________________________________________________________
    # Read data, form the target variable separately
    # Xy = read_dataset("datasets/numNewOutlinks_dataset-between_09-07_and_09-14-SVflat-with_history_8-sample.pkl",
    fn = r'~/dataset/1M/Pickle/1M_all_with_diffs_avg_linkChangeRate.pkl'

    total_target = 'External'
    target_feature = 'diff' + total_target + 'OutLinks'
    target_feature = 'link' + total_target + 'ChangeRate'
    seed = 0
    which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
    Y_mean_features = ['diff' + total_target + 'OutLinks-' + str(i + 1) for i in range(8)]
    features = [f for fs in which_features for f in feature_sets[fs]]

    tuning_fraction, test_fraction = 1. / 4, 1. / 4
    Xy = pd.read_pickle(fn)

    Xy = Xy[Xy['isValid'] == True]
    Xy = Xy.set_index('url')
    y = Xy[target_feature]  # pd.Series
    # y = y.to_numpy()  # np.array
    X = Xy.drop([target_feature], axis='columns')
    # X = X[]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                        random_state=seed)
    X_train = X_train[features + Y_mean_features]
    X_test = X_test[features + Y_mean_features]

    # X_train = X_train.reset_index()
    # X_train = X_train.to_numpy()
    # y_train = y_train.to_numpy()
    # X_test = X_test.to_numpy()
    # y_test = y_test.to_numpy()
    ngboost_save = r'/home/kadkhodaeihr/dataset/1M/Pickle/ngb_model_' + '_'.join(
        which_features) + '_' + total_target + '.pkl'
    # ngboost_save = r'/home/kadkhodaeihr/dataset/1M/ngb_.pkl'
    # ngboost_save=r'F:\Netherlands Project\WebInsight\Source Code\PythonApplication2\PythonApplication2\Nhung\NGBoost\ngb_' + '_'.join(
    saving = True  # True: create and save new pickle, False: load saved pickle
    if saving is True:
        ngb = NGBRegressor(Dist=Poisson, verbose_eval=10, n_estimators=500)
        print("Start training...")
        ngb.fit(X_train, y_train)
        print('saving ...')
        file_path = ngboost_save
        with open(file_path, "wb") as f:
            pickle.dump(ngb, f)
    else:
        print('loading ...')
        file_path = ngboost_save
        with open(file_path, "rb") as f:
            ngb = pickle.load(f)

    Y_preds = ngb.predict(X_test)  # return parameters
    Y_dists = ngb.pred_dist(X_test)  # return distribution (type object)
    Y_gener = np.random.poisson(lam=tuple(Y_preds.reshape(1, -1)[0]))

    df = pd.read_csv(r'/home/kadkhodaeihr/dataset/1M/Pickle/orders/' + total_target + '_orders_SP_SN_DN8_DP8_DPRate.csv')
    df['NGB_preds'] = Y_preds
    df['NGB_gener'] = Y_gener
    df.to_csv(r'/home/kadkhodaeihr/dataset/1M/Pickle/orders/orders-NGB_' + total_target + '.csv', index=False)
    print("Finished")

