import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import time
t = time.time()


# Data structure to store the final results
L = []


# Selecting data folder
data_dir = os.getcwd() + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)


# Independent and generalized executions (executions = 30)
for exec in range(1, 31):

    print("Execução:", exec)
    # Setting random number generator seed
    semente = ((exec + 100) * 77) + 2**exec

    # Iteration over the different stations
    for station in files:
        d = {}

        station_name = station[:-5]
        df = pd.read_excel(data_dir + "\\" + station, header=1)

        print("Estação:", station_name)

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
        X_train = X_train.drop(['year', 'month', 'weekday name',
                                'solar radiation'], axis=1)
        X_test = X_test.drop(['year', 'month', 'weekday name',
                              'solar radiation'], axis=1)

        # Pipeline creation
        pipeline = Pipeline([
            ('poly_features', PolynomialFeatures()),
            ('ridge_reg', Ridge())
        ])

        # Values for GridSearchCV to iterate over
        param_grid = {
            'poly_features__degree': [1, 2, 3],
            'poly_features__interaction_only': [True, False],
            'poly_features__include_bias': [True, False],
            'ridge_reg__alpha': [0, 1, 2],
            'ridge_reg__fit_intercept': [True, False],
            'ridge_reg__normalize': [True, False]
        }

        # Scores
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        # Do CV for each of the possible combinations of the parameter values
        grid = GridSearchCV(pipeline, cv=KFold(n_splits=10, shuffle=True,
                                               random_state=semente),
                            param_grid=param_grid,
                            refit='neg_mean_squared_error', scoring=scoring,
                            iid=False, return_train_score=True)
        grid.fit(X_train, y_train)

        # Predict test instances using the best configs found in the CV step
        y_pred = grid.predict(X_test)

        # Save the results for each station
        d = {
             'Exec': exec,
             'Seed': semente,
             'Station': station_name,
             'Best Params': grid.best_params_,
             'Best Score': grid.best_score_,
             'Best Estimator': grid.best_estimator_,
             'y_pred': y_pred
        }

        L.append(d)


# Sava final results into a .csv file
results = pd.DataFrame(L)
results.to_csv('results.csv', index=False)


# Displays the elapsed time
elapsed = time.time() - t
h = int(elapsed / 3600)
m = int((elapsed % 3600) / 60)
s = (elapsed % 3600) % 60
print(h, 'HORA(S)', m, 'MINUTO(S)', s, 'SEGUNDO(S)')
