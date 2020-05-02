import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


loaded_model = joblib.load("model6.pkl")

X_test = pd.read_json("Bobo Dioulasso" + "_xtest.json")

y_test = pd.read_json("Bobo Dioulasso" + "_ytest.json", typ='series')

y_p = loaded_model.predict(X_test)


len(loaded_model.named_steps['ridge_reg'].coef_)
loaded_model.named_steps['ridge_reg'].coef_
loaded_model.named_steps['poly_features'].n_input_features_
loaded_model.named_steps['poly_features'].n_output_features_
loaded_model.named_steps['features_select'].scores_
len(loaded_model.named_steps['features_select'].scores_)
len(loaded_model.named_steps['features_select'].pvalues_)


metrics.mean_squared_error(y_test, y_p, squared=False)
metrics.mean_squared_error(y_test, np.asarray(livro), squared=False)