import itertools
from sys import exit
from sklearn.experimental import enable_hist_gradient_boosting  # it is needed for the HistGradientBoostingRegressor
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import Definitions
from Evaluation import Evaluation
from MLTools import MLTools
from Plots import Plots

if __name__ == '__main__':
    random_state = 0
    n_jobs = -1
    which_model = 'ET'  # ET and HGB are most competitive, quick, and configurable; GB far too slow on large data
    # HGB ~ LGB (same alg.); HGB is experimental, so its configuration may need porting in the future
    which_features = ['SP','SN']
    # which_features = ['SP']
    num_SV_clusters = 20
    tuning_fraction, test_fraction = 1. / 3, 1. / 4

    # for _content change rate_; don't need this for this manuscript (it used to be Sec. 4.1, but I commented it out)
    # target, new_target = ['changeCount', 'fetchCount'], 'changeRate'
    # dataset_filename = "datasets/changeRate_dataset-SVflat.pkl"

    # for _link change rate_
    target, new_target = ['linkInternalChangeRate'], 'linkInternalChangeRate'
    # target, new_target = ['linkExternalChangeRate'], 'linkExternalChangeRate'
    dataset_filename = r"F:\Netherlands Project\WebInsight\Dataset\1M pickle dataset 384323 instances doina\1M_all_with_avg_atts.pkl"
    # dataset_filename = r"d:/WebInsight/datasets/1M_all_with_avg_atts.pkl"
    # dataset_filename = r"dataset/1M/Pickle/1M_all_with_avg_atts.pkl"

    untuned_models = {
        'ET': [ExtraTreesRegressor(n_jobs=n_jobs, random_state=random_state),
               {'n_estimators': [200, 300, 400, 500],
                'min_samples_leaf': [2, 5, 10, 15, 20, 25]}],
        'HGB': [HistGradientBoostingRegressor(max_depth=None, max_leaf_nodes=None, random_state=random_state),
                {'max_iter': [200, 300, 400, 500],
                 'min_samples_leaf': [10, 15, 20, 25],
                 'learning_rate': [0.1, 0.08, 0.06, 0.04, 0.02]}],
        'LGB': [LGBMRegressor(max_depth=-1, random_state=random_state, n_jobs=n_jobs),
                {'n_estimators': [3000, 4000, 5000],
                 'min_child_samples': [2, 5, 7],
                 'learning_rate': [0.1]}],
        'GB': [GradientBoostingRegressor(max_depth=-1, random_state=random_state),
               {'n_estimators': [400, 600, 800, 1000],
                'min_samples_leaf': [2, 5],
                'learning_rate': [0.1]}],
        'ETtest': [ExtraTreesRegressor(n_jobs=n_jobs, random_state=random_state),
                   {'n_estimators': [400], 'min_samples_leaf': [2]}],
    }
    pretuned_models = {
        'ET': ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=n_jobs, random_state=random_state),
        'HGB': HistGradientBoostingRegressor(max_iter=400, min_samples_leaf=20, learning_rate=0.02, max_depth=None,
                                             max_leaf_nodes=None, random_state=random_state)
    }

    title = "target " + new_target + "-features " + "_".join(which_features) + "-model " + which_model + "-seed " + str(
        random_state)
    print("\n" + "_" * 80 + "\nRandom state:", random_state)

    # export_scores([0.68, 0.16, 0.11], ['R2', 'MAE', 'MedAE'], title)
    # exit()

    # _________________________________________________________________________________________________
    # Read data, form the target variable
    features = [f for fs in which_features for f in Definitions.feature_sets[fs]]
    # Xy = read_dataset(dataset_filename, ['url'] + features + target)  # Xy = pd.DataFrame with url as index
    Xy = pd.read_pickle(dataset_filename)  # Xy = pd.DataFrame with url as index
    Xy = Xy[Xy['isValid'] == True]
    Xy = Xy[['url'] + features + target]
    Xy = Xy.set_index('url')

    # 	-> content change rate
    # y = Xy['changeCount'] / (Xy['fetchCount'] - 1) # pd.Series
    # 	-> link change rate
    y = Xy[new_target]  # pd.Series

    y = y.to_numpy()  # np.array
    print("\nTarget:", new_target)
    X = Xy#.drop(target, axis='columns')  # X = pd.DataFrame with url as index; still needed to separate SV features

    # _________________________________________________________________________________________________
    # (Optional, not ML) Run the dynamic baseline: y_pred = 0 (or any other value)
    # baseline_y_pred = [0 for i in range(len(y))]
    # _ = baseline_score(baseline_y_pred, y)

    # _________________________________________________________________________________________________
    # (Optional) Dimensionality reduction for the semantic vector
    if 'v' in which_features:
        agglo = FeatureAgglomeration(affinity='cosine', linkage='complete',
                                     n_clusters=num_SV_clusters)  # n_clusters should be hypertuned
        X_SV = agglo.fit_transform(X[Definitions.feature_sets['v']])  # X_SV = np.array
        print("Features reduced to", X_SV.shape)

        # merge back into a pd.DataFrame X[without SV features] with the now reduced X_SV
        X = X.drop(Definitions.feature_sets['v'], axis='columns')  # X = pd.DataFrame
        new_columns = ["SVCluster" + str(i) for i in range(num_SV_clusters)]
        X_SV = pd.DataFrame(X_SV, index=X.index.values, columns=new_columns)
        X[new_columns] = X_SV[new_columns]

    # _________________________________________________________________________________________________
    # (Optional) Any stats?
    # plot_univariate_distribution(y, 9, (0, 1), pretty_label_dict.get(new_target, new_target), "figures-changeRate/distribution_"+new_target+".png")
    # scatterplot_2D(X, y, ['textSize', 'numInternalOutLinks'], pretty_label_dict.get(new_target, new_target))

    Plots.corr_matrix(X, [Definitions.pretty_label_dict.get(f, f) for f in X.columns.values],
                      'spearman', "corrmatrix-target_" + new_target + "-features " + "_".join(
            which_features) + ".png")
    # exit()  # remove this when needed

    # _________________________________________________________________________________________________
    # Build a dictionary of feature names, for later reference
    feature_names = X.columns.values
    feature_index_to_name = dict(enumerate(X.columns.values))
    print("\nFinal enumerated feature set (" + "_".join(which_features) + "):\n\t", feature_index_to_name, "\n")
    X = X.to_numpy()  # np.ndarray; X column names are lost from X itself now

    # _________________________________________________________________________________________________
    # Define the model, with feature scaling; the semantic vector needs no scaling, but it's hard to separate it
    # model = Pipeline(steps=[	('scaler', PowerTransformer()),
    # 							('model', untuned_models[which_model])])
    # ...or without if no scaling needed
    model = untuned_models[which_model][0]
    params = untuned_models[which_model][1]
    print("Model:", which_model)

    # (Alternative) transformed target, but not effective on this changeRate distribution (too discrete)
    # model = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='uniform'))

    # _________________________________________________________________________________________________
    # (Step 0) Set aside a fraction of test data
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, train_size=1 - test_fraction, shuffle=True,
                                                    random_state=random_state)

    # _________________________________________________________________________________________________
    # (Step 1) Tune hyperparameters on a fraction of the development data
    X_tune, _, y_tune, _ = train_test_split(X_dev, y_dev, train_size=tuning_fraction, shuffle=True,
                                            random_state=random_state)  # for lack of a simpler split function
    tuned_model = MLTools.hyperparameter_tuning(model, params, X_tune, y_tune, cv=5, n_jobs=n_jobs)

    # (Alternative) preturned model, based on prior runs with tuning
    # tuned_model = pretuned_models[which_model]

    # _________________________________________________________________________________________________
    # (Step 2) Refit a single model on all development data
    print("\nRetraining on", len(y_dev), "samples")
    tuned_model.fit(X_dev, y_dev)

    # _________________________________________________________________________________________________
    # (Step 3) Test it on the test data
    scoring = Evaluation.test_and_score(tuned_model, X_test, y_test)
    Plots.export_scores(list(scoring.values()), list(scoring.keys()), title)

    # _________________________________________________________________________________________________
    # (Step 4, optional) Get permutation feature importance scores, or paste it from a previous run
    # top_feature_indices = [4, 3, 6, 2, 1, 5, 0] # manual
    top_feature_indices = Plots.plot_permutation_feature_importance(tuned_model,
                                                                    title, feature_names, new_target, X_test, y_test,
                                                                    random_state)
    # top_feature_indices = [3, 0, 5, 21]
    # _________________________________________________________________________________________________
    # (Optional) Refit a model on 2 top features only, to visualise the decision boundaries
    print("\nRefitting on two features for decision boundaries")
    # top_feature_pairs = itertools.permutations(top_feature_indices[-4:], r=2)
    top_feature_pairs = itertools.combinations(top_feature_indices[-4:], r=2)  # (0,3) & (3,0) are equal
    top_feature_pairs = [list(t) for t in top_feature_pairs]  # tuple to list
    # b = [(min(r), max(r)) for r in top_feature_pairs]
    # c = set(b)
    # d = [list(r) for r in c]
    # top_feature_pairs = d
    for top_feature_indices in top_feature_pairs:
        top_feature_names = np.array(feature_names)[top_feature_indices]
        tuned_model.fit(X_dev[:, top_feature_indices], y_dev)
        y_pred = tuned_model.predict(X_test[:, top_feature_indices])
        score, score_name = mean_absolute_error(y_test, y_pred), 'MAE'
        print("\t", top_feature_indices, score_name + ':', score)
        Plots.plot_two_top_features(tuned_model, score, score_name,
                                    title, X[:, top_feature_indices], y, top_feature_names,
                                    Definitions.pretty_label_dict.get(new_target, new_target))
