import pickle

import numpy as np
import pandas as pd
from ngboost.distns import Poisson
from ngboost.distns import Normal
# from scipy.stats import norm
from ngboost import NGBRegressor
import random

# from torch import feature_dropout

import definitions
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, median_absolute_error, mean_absolute_error, r2_score, make_scorer

total_target = 'External'
target_feature = 'diff' + total_target + 'OutLinks'
seed = 0
which_features = ['SP', 'SN', 'DN8', 'DP8', 'DPRate']
Y_mean_features = ['diff' + total_target + 'OutLinks-' + str(i + 1) for i in range(8)]
features = [f for fs in which_features for f in definitions.feature_sets[fs]]
random.seed(seed)
tuning_fraction, test_fraction = 1. / 2, 1. / 4

train = pd.read_csv('train.csv')
train = train[train[target_feature] > 0]
# test = pd.read_csv('test.csv')
train.set_index('url')
X_train = train[features]
y_train = train[target_feature]

X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=tuning_fraction, shuffle=True,
                                        random_state=0)

untuned_model = NGBRegressor(Dist=Poisson, n_estimators=500, random_state=0, verbose=True, verbose_eval=20)
params = {'learning_rate': [0.06, 0.07, 0.08]}



print("\nHyperparameter tuning on", len(y_tune), "samples")

my_scorer = make_scorer(r2_score)
tuned_model = GridSearchCV(untuned_model, params, cv=5, scoring=my_scorer, n_jobs=-1)
tuned_model.fit(X_tune, y_tune)

print("\n\tScores on the development set:\n")
means = tuned_model.cv_results_['mean_test_score']
stds = tuned_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, tuned_model.cv_results_['params']):
    print("\tmean %0.5f, stdev %0.05f for parameters %r" % (mean, std, params))

print("\n\tBest parameters on the development set:", tuned_model.best_params_)
model = tuned_model.best_estimator_

# with open("tundedModel_halfData", "wb") as f:
#     pickle.dump(model, f)

print("finish")
