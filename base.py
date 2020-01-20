import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


# Data structure to store the results
fold = []


# Selecting data folder
data_dir = os.getcwd() + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)


# Loading original data files
for station in files:
    attr = []
    station_name = station[:-5]
    df = pd.read_excel(data_dir + "\\" + station, header=1)

    # Data preprocessing
    df.columns = ['date', 'wind speed', 't max', 't min', 'humidity max',
                  'humidity min', 'vpd', 'evaporation', 'solar radiation']
    aux = df['date']
    df = df.set_index('date')
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekday name'] = df.index.weekday_name

    # Data split
    X_train, X_test = df.loc['1998':'2008'], df.loc['2009':'2012']
    y_train, y_test = X_train['solar radiation'], X_test['solar radiation']
    X_train = X_train.drop(['year', 'month', 'weekday name', 'solar radiation']
                           , axis=1)
    X_test = X_test.drop(['year', 'month', 'weekday name', 'solar radiation'],
                         axis=1)

    # Pipeline creation
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures()),
        ('lr_reg', LinearRegression())
    ])

    # Values for GridSearchCV to iterate over
    param_grid = {
        'poly_features__degree': [1, 2, 3],
        'poly_features__interaction_only': [True, False],
        'poly_features__include_bias': [True, False],
        'lr_reg__fit_intercept': [True, False],
        'lr_reg__normalize': [True, False]
    }

    # Scores
    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    # Do 10 CV for each of the 6 possible combinations of the parameter values
    grid = GridSearchCV(pipeline, cv=10, param_grid=param_grid, refit='r2',
                        scoring=scoring, iid=False, return_train_score=True)
    grid.fit(X_train, y_train)

    # Summarize results
    pd.DataFrame(grid.cv_results_)
    print("Best Estimator:", grid.best_estimator_)
    print("Best Score:", grid.best_score_)
    print("Best Params:", grid.best_params_)
    print("Best Index:", grid.best_index_)

    # Predict test instances using the best configs found in the CV step
    y_pred = grid.predict(X_test)

    # Metric calculations
    print("MSE:", metrics.mean_squared_error(y_test, y_pred, squared=True))
    print("RMSE:", metrics.mean_squared_error(y_test, y_pred, squared=False))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("R²:", metrics.r2_score(y_test, y_pred))

    # Save the results for each station
    attr.append(station_name)
    attr.append(grid.best_score_)
    attr.append(grid.best_params_)
    attr.append(grid.best_index_)
    attr.append(grid.best_estimator_)
    attr.append(pd.DataFrame(grid.cv_results_))
    attr.append(metrics.mean_squared_error(y_test, y_pred, squared=True))
    attr.append(metrics.mean_squared_error(y_test, y_pred, squared=False))
    attr.append(metrics.mean_absolute_error(y_test, y_pred))
    attr.append(metrics.r2_score(y_test, y_pred))

    fold.append(attr)


# Save the simulation results into a pickle file
with open('results.pickle', 'wb') as f:
    pickle.dump(fold, f)


# Load the results saved
# with open('results.pickle', 'rb') as f:
#     results = pickle.load(f)
