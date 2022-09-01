import itertools
from sys import exit
from sklearn.experimental import enable_hist_gradient_boosting  # it is needed for the HistGradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, median_absolute_error, mean_absolute_error, r2_score, make_scorer
from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, \
    ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split, ShuffleSplit, learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from matplotlib.patches import Rectangle
from scipy import stats


def int_to_categorical(x):
    if x == 0:
        return 0
    else:
        return 1

print("-" * 100)

fn = r'~/dataset/1M/Pickle/1M_all_with_diffs_avg.csv'
new_target = 'diffInternalOutLinks'

random_state = 0
tuning_fraction, test_fraction = 1. / 3, 1. / 4
Xy = pd.read_csv(fn)
print("dataset ready")

Xy = Xy.set_index('url')
y = Xy[new_target]  # pd.Series
y = y.to_numpy()  # np.array
X = Xy.drop([new_target, 'language'], axis='columns')
X = X.to_numpy()
y = np.array([int_to_categorical(yi) for yi in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                    random_state=random_state)
print("start training")
model = ExtraTreesClassifier(n_jobs=-1, random_state=random_state, n_estimators=500, min_samples_leaf=2)
model.fit(X_train, y_train)
print("model is ready. start predicting")
y_prob = model.predict_proba(X_test)

y_pred = np.rint(y_prob[:, 1])  # prob of appearing new outlink
print(stats.spearmanr(y_test, y_pred))

print("finished")
