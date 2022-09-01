import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, \
    fbeta_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, cross_validate, cross_val_predict, \
    learning_curve, train_test_split

fig_path = 'figures_spearman_correlation_orders/'


def int_to_categorical(x):
    if x == 0:
        return 0
    else:
        return 1


def calc_link_change_rate(row):
    s = 0
    for i in range(8):
        if row['diffExternalOutLinks-' + str(i + 1)] != 0:
            s += 1
    return s / 8


def plot_bar(avg_prediction, avg_prediction_stddev, regressor_prediction, regressor_prediction_stddev, file_name):
    labels = ['Internal OutLinks', 'External OutLinks']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    # axes = plt.gca()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylim([.40, 1])
    rects1 = ax.bar(x - width / 2, avg_prediction, width, color='xkcd:coral pink',
                    label='Using average of history')
    rects2 = ax.bar(x + width / 2, regressor_prediction, width,
                    color='xkcd:soft green', label='ExtraTreeClassifier')
    # rects1 = ax.bar(x - width / 2, avg_prediction, width, yerr=avg_prediction_stddev, label='Avg of history')
    # rects2 = ax.bar(x + width / 2, regressor_prediction, width, yerr=regressor_prediction_stddev,
    #                 label='ExtraTreeClassifier')
    # rects1 = ax.bar(x - width/2, a, width, label='avg of history',color='xkcd:coral pink')
    # rects2 = ax.bar(x + width/2, b, width, label='ExtraTreeClassifier',color='xkcd:soft green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Spearman correlation')
    # ax.set_title('Spearman correlation of estimated number \n of new outlinks and the real value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=0)
    ax.bar_label(rects2, padding=0)

    fig.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()
    # plt.show()


def hyperparameter_tuning(model, params, X_tune, y_tune, cv=5, n_jobs=-1):
    print("\nHyperparameter tuning on", len(y_tune), "samples")

    my_scorer = make_scorer(balanced_accuracy_score)
    tuned_model = GridSearchCV(model, params, cv=cv, verbose=5, scoring=my_scorer, n_jobs=n_jobs)
    tuned_model.fit(X_tune, y_tune)

    print("\n\tScores on the development set:\n")
    means = tuned_model.cv_results_['mean_test_score']
    stds = tuned_model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, tuned_model.cv_results_['params']):
        print("\tmean %0.5f, stdev %0.05f for parameters %r" % (mean, std, params))

    print("\n\tBest parameters on the development set:", tuned_model.best_params_)
    model = tuned_model.best_estimator_

    return model


if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

fn = r'~/dataset/1M/Pickle/1M_all_with_diffs_avg_linkChangeRate.csv'
fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\1M_all_with_diffs_avg_linkChangeRate.csv'
fn = r'1M_all_with_diffs_avg_linkChangeRate.csv'
tuning_fraction, test_fraction = 1. / 4, 1. / 4
Xy = pd.read_csv(fn)
# Xy['externalLinkChangeRate'] = Xy.apply(lambda row: calc_link_change_rate(row), axis=1)
Xy = Xy.set_index('url')
targets = ['diffInternalOutLinks', 'diffExternalOutLinks']
seeds = [0, 1]

avg_estimation_mean = []
regressor_estimation_mean = []
avg_estimation_stddev = []
regressor_estimation_stddev = []

avg_estimation_mean2 = []
regressor_estimation_mean2 = []
avg_estimation_stddev2 = []
regressor_estimation_stddev2 = []

untuned_models = {
    'ET': [ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=0),
           # {'n_estimators': [50, 200, 300, 400, 500],
           #  'min_samples_leaf': [2, 5, 10, 15, 20, 25]}],
           # {'n_estimators': [50, 200, 300, 400, 500],
           #  'min_samples_leaf': [2, 5, 10]}],
           {'n_estimators': [50, 100, 200],
            'min_samples_leaf': [2, 5]}],
}

for target in targets:
    avg_estimation_temp = []
    regressor_estimation_temp = []
    avg_estimation_temp2 = []
    regressor_estimation_temp2 = []
    model = untuned_models['ET'][0]
    params = untuned_models['ET'][1]
    is_tuned = False
    for seed in seeds:
        y = Xy[target]  # pd.Series
        # y = y.to_numpy()  # np.array
        X = Xy.drop([target, 'language'], axis='columns')
        # X = X.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                            random_state=seed)
        X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=tuning_fraction, shuffle=True,
                                                random_state=seed)  # for lack of a simpler split function

        X_test_link_changeRate = X_test[
            'internalLinkChangeRate' if 'Internal' in target else 'externalLinkChangeRate'].to_numpy()
        y_train = np.array([int_to_categorical(yi) for yi in y_train])
        y_tune = np.array([int_to_categorical(yi) for yi in y_tune])
        # y_test = np.array([int_to_categorical(yi) for yi in y_test])

        avg_estimation_temp.append(stats.spearmanr(y_test, X_test_link_changeRate)[0])
        avg_estimation_temp2.append(stats.spearmanr(y_test, X_test_link_changeRate)[0])

        print("\nRetraining on", len(y_train), "samples")
        # model = ExtraTreesClassifier(n_jobs=-1, random_state=seed, n_estimators=400, min_samples_leaf=2)
        if not is_tuned:
            tuned_model = hyperparameter_tuning(model, params, X_tune, y_tune, cv=3, n_jobs=-1)
            is_tuned = True
        tuned_model.fit(X_train, y_train)
        #

        y_pred = tuned_model.predict_proba(X_test)
        y_pred1 = y_pred[:, 0]
        regressor_estimation_temp2.append(stats.spearmanr(y_test, y_pred1)[0])

        y_pred2 = y_pred[:, 1]
        regressor_estimation_temp.append(stats.spearmanr(y_test, y_pred2)[0])

    avg_estimation_mean.append(np.mean(avg_estimation_temp))
    regressor_estimation_mean.append(np.mean(regressor_estimation_temp))
    avg_estimation_stddev.append(np.std(avg_estimation_temp))
    regressor_estimation_stddev.append(np.std(regressor_estimation_temp))

    avg_estimation_mean2.append(np.mean(avg_estimation_temp2))
    regressor_estimation_mean2.append(np.mean(regressor_estimation_temp2))
    avg_estimation_stddev2.append(np.std(avg_estimation_temp2))
    regressor_estimation_stddev2.append(np.std(regressor_estimation_temp2))

fn1 = fig_path + 'spearman_corr_orders_probOfNewOutlinks1.png'
fn2 = fig_path + 'spearman_corr_orders_probOfNewOutlinks2.png'
plot_bar(fn1, avg_estimation_mean, avg_estimation_stddev, regressor_estimation_mean, regressor_estimation_stddev)
plot_bar(fn2, avg_estimation_mean2, avg_estimation_stddev2, regressor_estimation_mean2, regressor_estimation_stddev2)
print("avg_history", avg_estimation_mean, avg_estimation_stddev)
print("ETRegressor", regressor_estimation_mean, regressor_estimation_stddev)
print("finished")
