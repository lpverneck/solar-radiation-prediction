import numpy as np


# Scores used for training
scoring = [
    "neg_root_mean_squared_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "r2",
]

# ======================================================================

# s1 params
# g_values = [
#     0,
#     0.0001,
#     0.001,
#     0.01,
#     0.0125,
#     0.1,
#     0.125,
#     0.175,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
# ]
# comp_values = np.linspace(1, 5, 24).tolist()
# alpha_param = g_values + comp_values

# param_grid = {
#     "poly_features__degree": [1, 2, 3, 4],
#     "poly_features__interaction_only": [True, False],
#     "poly_features__include_bias": [True, False],
#     "ridge_reg__alpha": alpha_param,
#     "ridge_reg__fit_intercept": [True],
#     "ridge_reg__normalize": [False],
# }

# ======================================================================

# s2 params
# alpha_param = [0, 0.0001, 0.001, 0.01, 0.0125, 0.1, 0.125, 0.175, 0.2, 0.3, 1]
# n_feats = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

# param_grid = {
#     "poly_features__degree": [4],
#     "poly_features__interaction_only": [True, False],
#     "poly_features__include_bias": [True, False],
#     "ridge_reg__alpha": alpha_param,
#     "ridge_reg__fit_intercept": [True],
#     "ridge_reg__normalize": [False],
#     "features_select__k": n_feats,
# }

# ======================================================================

# fast test params
alpha_param = [0, 1]
n_feats = [20, 30]

param_grid = {
    "poly_features__degree": [2],
    "poly_features__interaction_only": [True, False],
    "poly_features__include_bias": [True, False],
    "ridge_reg__alpha": alpha_param,
    "ridge_reg__fit_intercept": [True],
    "ridge_reg__normalize": [False],
    "features_select__k": n_feats,
}
