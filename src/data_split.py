"""Data Split

Preprocess, standardizes and splits the raw data into subsets for
training the models and validate them.
"""


import os
import numpy as np
import pandas as pd


# Selecting data folder
data_dir = os.getcwd()[:-6] + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)


for file_name in files:

    station_name = file_name[:-5]
    df = pd.read_excel(data_dir + "\\" + file_name, header=1)

    # Data preprocessing
    df.columns = ['date', 'wind speed', 't max', 't min', 'humidity max',
                  'humidity min', 'vpd', 'evaporation', 'solar radiation']
    aux = df['date']
    df = df.set_index('date')
    # df['year'] = df.index.year
    # df['month'] = df.index.month
    # df['weekday name'] = df.index.weekday_name

    # Data split
    X_train, X_test = df.loc['1998':'2008'], df.loc['2009':'2012']
    y_train, y_test = X_train['solar radiation'], X_test['solar radiation']
    X_train = X_train.drop('solar radiation', axis=1)
    X_test = X_test.drop('solar radiation', axis=1)

    # Save data separately
    X_train.to_json(data_dir[:-3] + "splitted\\" + station_name +
                    "_xtrain.json")
    y_train.to_json(data_dir[:-3] + "splitted\\" + station_name +
                    "_ytrain.json")
    X_test.to_json(data_dir[:-3] + "splitted\\" + station_name + "_xtest.json")
    y_test.to_json(data_dir[:-3] + "splitted\\" + station_name + "_ytest.json")
