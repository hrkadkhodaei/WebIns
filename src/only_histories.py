import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from datetime import datetime


def hyperparameter_tuning(model, parameters, X, y, cv=5):
    """
    Tune the hyperparameters of an ExtraTreesRegressor model using GridSearchCV.
    """
    # Create the parameters list you wish to tune
    # parameters = {'n_estimators': [400, 500],
    #               'max_features': ['auto', 'sqrt', 'log2'],
    #               'max_depth': [None, 5, 3, 1]}

    # Create the model
    # model = ExtraTreesRegressor()

    # # Make an f1 scoring function for the GridSearchCV model
    # scoring_fnc = make_scorer(r2_score, greater_is_better=True)
    #
    # # Make the GridSearchCV object
    # grid = GridSearchCV(model, parameters, scoring_fnc, cv=5, n_jobs=-1)
    #
    # # Fit it to the training data
    # grid = grid.fit(X, y)
    #
    # # Return the best estimator
    # return grid.best_estimator_
    print("\nHyperparameter tuning on", len(y), "samples")

    my_scorer = make_scorer(r2_score)
    tuned_model = GridSearchCV(model, parameters, cv=cv, scoring=my_scorer, n_jobs=-1)
    tuned_model.fit(X, y)

    print("\n\tScores on the development set:\n")
    means = tuned_model.cv_results_['mean_test_score']
    stds = tuned_model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, tuned_model.cv_results_['params']):
        print("\tmean %0.5f, stdev %0.05f for parameters %r" % (mean, std, params))

    print("\n\tBest parameters on the development set:", tuned_model.best_params_)
    model = tuned_model.best_estimator_

    return model


InExs = ['External', 'Internal']
for InEx in InExs:
    print(datetime.now().strftime("%H:%M:%S"), "\n")
    target = f'link{InEx}ChangeRate'
    target = f'diff{InEx}OutLinks'

    # atts += [f'{prefix}num{InEx}InLinks-{i}' for i in range(1, 9)]
    prefix = 'related_'
    atts = [f'{prefix}diff{InEx}OutLinks-{i}' for i in range(1, 9)] + [f'{prefix}diff{InEx}OutLinks']
    atts += [f'diff{InEx}OutLinks-{i}' for i in range(1, 9)]
    atts += [f'avg_diff{InEx}OutLinks']
    atts += [f'related_linkInternalChangeRate'] + [f'related_linkExternalChangeRate']
    # atts += [f'diffInternalOutLinks', 'diffExternalOutLinks']

    state = 0
    path = r''
    fn_orders = path + fr'{InEx}_orders-NGB.csv'
    fn = path + r'1M_Final.pkl'

    df_orders = pd.read_csv(fn_orders)
    test_urls = list(df_orders['URL'])
    url_orders = set(test_urls)

    df = pd.read_pickle(fn)
    # df = pd.read_csv(path + 'test2.csv')
    df = df[atts + ['url'] + [target]]
    url_all = set(df['url'])

    url_train = url_all - url_orders
    df.set_index('url', inplace=True)
    df_train = df.loc[url_train]
    X_train = df_train[atts]
    y_train = df_train[target]

    df_result = pd.DataFrame()
    df_result['url'] = test_urls

    df_test = df.loc[test_urls]
    X_test = df_test[atts]
    y_test = df_test[target]
    df_result['y_true'] = y_test.values

    # weights = [1] * len(X_train)
    # weights = 1+np.sqrt(y_train.values)
    untuned_models = {
        'ET': [ExtraTreesRegressor(n_jobs=-1, n_estimators=500, random_state=state),
               {'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 5, 1]}]}

    model1 = untuned_models['ET'][0]
    params1 = untuned_models['ET'][1]

    # X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=1 / 4, shuffle=True,
    #                                         random_state=state)  # for lack of a simpler split function

    # print("hyperparameter_tuning")
    # tuned_model = hyperparameter_tuning(model1, params1, X_train, y_train)
    tuned_model = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=state)
    print("training first model")
    tuned_model.fit(X_train, y_train) #, sample_weight=weights)

    filename = path + f'{InEx}_finalized_model.sav'
    with open(filename, 'wb') as f:
        pickle.dump(tuned_model, f)
    y_pred = tuned_model.predict(X_test)

    df_result['y_pred'] = y_pred
    df_result.to_csv(path + rf'results\{InEx}_{target}_results_histories.csv', index=False, header=True)

    print(f"{InEx} Fininshed")
    print(datetime.now().strftime("%H:%M:%S"), "\n")


print(datetime.now().strftime("%H:%M:%S"))
