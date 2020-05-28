import numpy as np

# Parameters setup
guess_values = [0, 0.0001, 0.001, 0.01, 0.0125]
comp_values = np.linspace(1, 5, 24).tolist()
alpha_param = guess_values + comp_values


# Values for GridSearchCV to iterate over
param_grid = {
    "poly_features__degree": [1],
    "poly_features__interaction_only": [True, False],
    "poly_features__include_bias": [True, False],
    "ridge_reg__alpha": [0, 1],
    "ridge_reg__fit_intercept": [True],
    "ridge_reg__normalize": [False],
    "features_select__k": [20, 7, 3],
}
