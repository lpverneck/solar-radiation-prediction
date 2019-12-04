import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


def param_search(X_train, y_train, model, param, metric):
    """Choose the best model parameters in a range.

    Arguments:
    X_train -- Training data.
    y_train -- Training label.
    model   -- Model estimator object.
    param   -- Dictionary with the parameters values.
    metric  -- Metrics for the score computing.

    Return:
    best_scores -- Best scoring values.
    best_param  -- Best parameters values.
    """
    opt = GridSearchCV(
        estimator=model,
        param_grid=param,
        scoring=metric,
        #cv=10,
        iid=True,
        n_jobs=-1
    )
    opt = opt.fit(X_train, y_train)
    best_scores = opt.best_score_
    best_param = opt.best_params_
    
    return best_scores, best_param
    