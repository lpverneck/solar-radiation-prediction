"""
Support Vector Regression:

Param:
- X_train: ####
- y_train: ####
- X_test:  ####
- y_test:  ####
- dg:      degree
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn import metrics

def supvr (X_train, y_train, X_test, y_test, dg):
    
    reg_svm = SVR(
        kernel='rbf',
        gamma='auto',
        degree=dg,
        C=1.0,
        epsilon=0.1
    )
    reg_svm.fit(X_train, y_train)
    y_pred_svm = reg_svm.predict(X_test)
    print("SVM: \n==========")
    print("R²:", metrics.r2_score(y_test, y_pred_svm))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred_svm), "\n-----")

    return y_pred_svm, reg_svm
