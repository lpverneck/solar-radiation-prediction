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
from sklearn.feature_selection import SelectKBest, f_regression
from src import tasks
from params import *

t = time.time()
# warnings.filterwarnings("ignore")


L = []


# Scores used
scoring = [
    "neg_root_mean_squared_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "r2",
]


# Selecting data folder
# data_dir = os.getcwd() + "/data/raw"
# G.col instructions
# data_dir = os.getcwd() + "/drive/My Drive/data/raw"
# files = os.listdir(data_dir)
# files.pop(0)  # G.col (-)
data_dir, files = tasks.server_select(opt="loc")


# Independent and generalized executions (executions = 30)
for run in range(1, 31):

    print("=" * 45 + "\nExecution nº:", run)
    # Setting random number generator seed
    root = (run + 1550) * run

    # Iteration over the different stations
    for station in files:

        d = {}
        station_name = station[:-5]

        X_train = pd.read_json(
            data_dir[:-3] + "splitted\\" + station_name + "_xtrain.json"
        )
        y_train = pd.read_json(
            data_dir[:-3] + "splitted\\" + station_name + "_ytrain.json", typ="series"
        )
        X_test = pd.read_json(
            data_dir[:-3] + "splitted\\" + station_name + "_xtest.json"
        )
        y_test = pd.read_json(
            data_dir[:-3] + "splitted\\" + station_name + "_ytest.json", typ="series"
        )

        print("'\tStation:", station_name, end="")

        # Pipeline creation
        pipeline = Pipeline(
            [
                ("poly_features", PolynomialFeatures()),
                ("features_select", SelectKBest(score_func=f_regression)),
                ("scale", StandardScaler()),
                ("ridge_reg", Ridge()),
            ]
        )

        # Cross-Validation procedure
        grid = GridSearchCV(
            pipeline,
            cv=KFold(n_splits=10, shuffle=True, random_state=root),
            param_grid=param_grid,
            refit="neg_root_mean_squared_error",
            scoring=scoring,
            iid=False,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)

        # Save the fitted model
        joblib.dump(
            grid.best_estimator_,
            data_dir[:-8] + "models\\" + station_name + "_" + str(run) + "_model.pkl",
            compress=1,
        )

        # Predict test instances using the refit model
        y_pred = grid.predict(X_test)

        # Save the results for each station
        d = {
            "exec": run,
            "seed": root,
            "station": station_name,
            "best params": grid.best_params_,
            "best score": grid.best_score_,
            "best estimator": grid.best_estimator_,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "best index": grid.best_index_,
            "cv_results": pd.DataFrame(grid.cv_results_),
        }

        L.append(d)
        tasks.time_display(t, "mid")
        print(" - Completed.")


print("\nDone!")


# Save the final results into a .json file
results = pd.DataFrame(L)
results.to_json("results.json")


tasks.time_display(t, "end")
