"""Model Rebuild

Loads a specific previously trained model to access his attributes and
to predict new data.
"""


import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# Take stations name
data_dir = os.getcwd()[:-6] + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)

# Choose the run number and station
run = 1
station = files[0][:-5]

# Load model
loaded_model = joblib.load(station + "_" + str(run) + "_model.pkl")

# X_test = pd.read_json("Bobo Dioulasso" + "_xtest.json")
# y_test = pd.read_json("Bobo Dioulasso" + "_ytest.json", typ='series')

# Predict new data
y_p = loaded_model.predict()

# Attributes
len(loaded_model.named_steps['ridge_reg'].coef_)
loaded_model.named_steps['ridge_reg'].coef_
loaded_model.named_steps['poly_features'].n_input_features_
loaded_model.named_steps['poly_features'].n_output_features_
loaded_model.named_steps['features_select'].scores_
len(loaded_model.named_steps['features_select'].scores_)
len(loaded_model.named_steps['features_select'].pvalues_)
