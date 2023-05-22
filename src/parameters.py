""" parameters.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This file contains the parameter grids used in classify.py when hyperparameter tuning is done.
    Its contents are imported into classify.py.
"""

logistic_param_grid = {"C": [0.05, 0.1, 0.5, 1, 2, 3, 4],
                      "tol": [0.0001, 0.001, 0.01, 0.1, 1],
                      "penalty": ["l1", "l2"],
                      "solver": ["liblinear", "saga"],
                      "intercept_scaling": [0.5, 1, 2]}

mlp_param_grid = {"hidden_layer_sizes": [(10,), (20,), (30,), (50,), (100,), (125,), (150,), (10,10), (20,20), (25,25), (10, 10, 10), (20, 20, 20), (25, 25, 25)]}

logistic_param_grid_sm = {"C": [2, 3],
                          "tol": [0.001, 1],
                          "penalty": ["l1", "l2"],
                          "solver": ["liblinear"]}

mlp_param_grid_sm = {"hidden_layer_sizes": [(10,), (20,), (10, 10)]}