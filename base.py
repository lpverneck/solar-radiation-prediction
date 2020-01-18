import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


# Selecting data folder
data_dir = os.getcwd() + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)


# Loading original data files
# for station in files: ############################
#    station_name = station[:-5] ##################
station_name = files[0][:-5]
df = pd.read_excel(data_dir + "\\" + files[0], header=1)


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
X_train = X_train.drop(['year', 'month', 'weekday name', 'solar radiation'],
                       axis=1)
X_test = X_test.drop(['year', 'month', 'weekday name', 'solar radiation'],
                     axis=1)


# Pipeline creation
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures()),
    ('lr_reg', LinearRegression())
])


# Values for GridSearchCV to iterate over
param_grid = {
    'poly_features__degree': [1, 2, 3, 4, 5],
    'poly_features__interaction_only': [True, False]
}


# Scores
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']


# Do 10 CV for each of the 6 possible combinations of the parameter values
grid = GridSearchCV(pipeline, cv=10, param_grid=param_grid, scoring=scoring,
                    refit='r2')
grid.fit(X_train, y_train)


# Summarize results
pd.DataFrame(grid.cv_results_)
print("Best Estimator:", grid.best_estimator_)  # Best estimator config
print("Best Score:", grid.best_score_)  # Mean CV score of the best_estimator
print("Best Params:", grid.best_params_)


# now train and predict test instances
# using the best configs found in the CV step
##############################################################
