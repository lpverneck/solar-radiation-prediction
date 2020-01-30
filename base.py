import os
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import time


t = time.time()
warnings.filterwarnings("ignore")


# Data structure to store the final results
L = []


# Selecting data folder
data_dir = os.getcwd() + "\\data\\raw"
# G.col instructions
# data_dir = os.getcwd() + "/drive/My Drive/data/raw"
files = os.listdir(data_dir)
files.pop(0)  # G.col (-)


# Independent and generalized executions (executions = 30)
for exec in range(1, 31):

    print("=============================================\nExecution nº:", exec)
    # Setting random number generator seed
    semente = ((exec + 100) * 77) + 2**exec

    # Iteration over the different stations
    for station in files:

        d = {}

        station_name = station[:-5]
        df = pd.read_excel(data_dir + "\\" + station, header=1)

        print("'\tStation:", station_name, end='')

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
            'ridge_reg__alpha': [0, 1],
            'ridge_reg__fit_intercept': [True, False],
            'ridge_reg__normalize': [True, False]
        }

        # Scores
        scoring = ['neg_root_mean_squared_error', 'neg_mean_squared_error',
                   'neg_mean_absolute_error', 'r2']

        # Cross-Validation procedure
        grid = GridSearchCV(pipeline, cv=KFold(n_splits=10, shuffle=True,
                                               random_state=semente),
                            param_grid=param_grid,
                            refit='neg_root_mean_squared_error',
                            scoring=scoring,
                            iid=False, return_train_score=True)
        grid.fit(X_train, y_train)

        # Predict test instances using the refit model
        y_pred = grid.predict(X_test)

        # Save the results for each station
        d = {
             'exec': exec,
             'seed': semente,
             'station': station_name,
             'best params': grid.best_params_,
             'best score': grid.best_score_,
             'best estimator': grid.best_estimator_,
             'y_pred': y_pred,
             'best index': grid.best_index_,
             'cv_results': pd.DataFrame(grid.cv_results_)
        }

        L.append(d)
        print(" - Ok.")


print("Done!")


# Save the final results into a .csv file
results = pd.DataFrame(L)
results.to_csv('results.csv', index=False)


# Display the elapsed time
print("=============================================\nelapsed time:\n")
elapsed = time.time() - t
print(int(elapsed / 3600), 'hours', int((elapsed % 3600) / 60),
      'minutes', (elapsed % 3600) % 60, 'seconds')
print("=============================================\n")
