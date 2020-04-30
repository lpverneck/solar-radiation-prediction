import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures


t = time.time()
warnings.filterwarnings("ignore")


def time_display(t, flag=""):
    """Generates a display that shows the elapsed time of computation.

    Parameters
    ----------
    t : initial time in seconds since the Epoch.

    flag : {'mid', 'end'}
        Flag to chosse the position in the code.
        - 'mid' chooses the display in the middle of the computation.
        - 'end' chooses the display in the end of the computation.
    """
    elapsed = time.time() - t
    if (flag == "end"):
        print("="*45 + "\nTotal Elapsed Time:\n")
        print(int(elapsed / 3600), 'hours', int((elapsed % 3600) / 60),
              'minutes', int((elapsed % 3600) % 60), 'seconds')
        print("="*45 + "\n")
    elif (flag == "mid"):
        print(' - [', int(elapsed / 3600), 'h', int((elapsed % 3600) / 60),
              'm', int((elapsed % 3600) % 60), 's]', end='')


# Initial parameters
L = []
main_values = [0, 0.0001, 0.001, 0.01, 0.0125, 0.1, 0.125, 0.175, 0.3, 0.5]
# comp_values = np.linspace(1, 5, 5).tolist()
comp_values = np.linspace(1, 5, 10).tolist()
alpha_param = main_values + comp_values


# Values for GridSearchCV to iterate over
param_grid = {
    'poly_features__degree': [1, 2, 3, 4],
    'poly_features__interaction_only': [True, False],
    'poly_features__include_bias': [True, False],
    'ridge_reg__alpha': alpha_param,
    'ridge_reg__fit_intercept': [True],
    'ridge_reg__normalize': [True]
}


# Scores used
scoring = [
    'neg_root_mean_squared_error',
    'neg_mean_squared_error',
    'neg_mean_absolute_error',
    'r2'
]


# Selecting data folder
data_dir = os.getcwd() + "\\data\\raw"
# G.col instructions
# data_dir = os.getcwd() + "/drive/My Drive/data/raw"
files = os.listdir(data_dir)
files.pop(0)  # G.col (-)


# Independent and generalized executions (executions = 30)
for exec in range(1, 31):

    print("="*45 + "\nExecution nº:", exec)
    # Setting random number generator seed
    root = (exec + 1550) * exec

    # Iteration over the different stations
    for station in files:

        d = {}
        station_name = station[:-5]

        X_train = pd.read_json(data_dir[:-3] + "splitted\\" + station_name +
                               "_xtrain.json")
        y_train = pd.read_json(data_dir[:-3] + "splitted\\" + station_name +
                               "_ytrain.json", typ='series')
        X_test = pd.read_json(data_dir[:-3] + "splitted\\" + station_name +
                              "_xtest.json")
        y_test = pd.read_json(data_dir[:-3] + "splitted\\" + station_name +
                              "_ytest.json", typ='series')

        print("'\tStation:", station_name, end='')

        # Pipeline creation
        pipeline = Pipeline([
            ('poly_features', PolynomialFeatures()),
            ('ridge_reg', Ridge())
        ])

        # Cross-Validation procedure
        grid = GridSearchCV(pipeline, cv=KFold(n_splits=10, shuffle=True,
                                               random_state=root),
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
             'seed': root,
             'station': station_name,
             'best params': grid.best_params_,
             'best score': grid.best_score_,
             'best estimator': grid.best_estimator_,
             'y_true': y_test.to_numpy(),
             'y_pred': y_pred,
             'best index': grid.best_index_,
             'cv_results': pd.DataFrame(grid.cv_results_)
        }

        L.append(d)
        time_display(t, "mid")
        print(" - Completed.")


print("\nDone!")


# Save the final results into a .json file
results = pd.DataFrame(L)
results.to_json('results.json')


time_display(t, "end")
