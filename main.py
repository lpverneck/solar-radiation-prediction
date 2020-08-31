import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from src import tasks
from hyperparameters import *

t = time.time()

# >>>>>>>>>>>>>>>>>>>>>>>>> #
standart = param_grid
# >>>>>>>>>>>>>>>>>>>>>>>>> #

# Select data folder
data_dir, files = tasks.set_directory()

# Select folder to save models and the final results
models_dir, res = tasks.mode_select(opt="test")

L = []

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
            data_dir[:-3] + "processed/" + station_name + "_xtrain.json"
        )
        y_train = pd.read_json(
            data_dir[:-3] + "processed/" + station_name + "_ytrain.json",
            typ="series",
        )
        X_test = pd.read_json(
            data_dir[:-3] + "processed/" + station_name + "_xtest.json"
        )
        y_test = pd.read_json(
            data_dir[:-3] + "processed/" + station_name + "_ytest.json",
            typ="series",
        )

        print("'\tStation:", station_name, end="")

        # Pipeline creation
        pipeline = Pipeline(
            [
                ("poly_features", PolynomialFeatures()),
                (
                    "features_select",
                    SelectKBest(score_func=mutual_info_regression),
                ),
                ("scale", StandardScaler()),
                ("ridge_reg", Ridge()),
            ]
        )

        # >>>>>>>>>>>>>>>>>>>>>>>>> #
        param_grid = hyperparameter_check(
            st_name=station_name,
            std=standart,
            alpha=alpha_param,
            feats=n_feats,
        )
        # >>>>>>>>>>>>>>>>>>>>>>>>> #

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
            data_dir[:-8]
            + models_dir
            + station_name
            + "_"
            + str(run)
            + "_model.pkl",
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
results.to_json(res)


tasks.time_display(t, "end")
