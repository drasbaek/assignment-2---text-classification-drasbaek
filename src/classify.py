""" classify.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This file classifies the vectorized Fake News Dataset by using a Logistic Regression or a Multi-Layer Perceptron (Neural Network).
    Hyperparameter tuning is done by using GridSearchCV for both model types.
    The best model, as determined by GridSearchCV, is saved to the models directory and the classification report is saved to the out directory.

Usage:
    $ python src/classify.py -m logistic -g full
"""

# import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import argparse
import numpy as np
from sklearn import metrics
import sys
import joblib
import logging
sys.path.append("..")
from parameters import *


def define_paths():
    '''
    Define paths for input and output data.

    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   model_outpath (pathlib.PosixPath): Path to output data for models.
    -   report_outpath (pathlib.PosixPath): Path to output data for reports.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "in"

    # define output dir for models
    model_outpath = path.parents[1] / "models"

    # define output dir for reports
    report_outpath = path.parents[1] / "out"

    return inpath, model_outpath, report_outpath


def input_parse():
    '''
    Parses input arguments.
    It is possible to specify the model to use and the size of the grid to use for hyperparameter tuning.

    Returns:
    -   args (argparse.Namespace): Parsed arguments.
    '''

    # initialize parser
    parser = argparse.ArgumentParser()

    # add model arguments
    parser.add_argument("-m", "--model", default="logistic", help="model to use")
    parser.add_argument("-g", "--grid_size", default="full", help="use small grid for hyperparameter tuning")

    # parse arguments
    args = parser.parse_args()

    return args


def input_checker(args):
    '''
    Check if the input arguments are correct and return error message if not.
    
    Args:
    -   args (argparse.Namespace): Parsed arguments.
    '''

    # check if model and grid size are correct
    if args.model not in ["logistic", "mlp"]:
        raise ValueError("Model must be either logistic or mlp.")
    
    if args.grid_size not in ["full", "small"]:
        raise ValueError("Small grid must be either true or false.")


def read_data(inpath):
    '''
    Load the vectorized data from the data directory.

    Args:
    -   inpath (pathlib.PosixPath): Path to input data.

    '''

    # inform data is being loaded
    logging.info("(1/4) Loading vectorized data")

    # load data from the data directory
    data = np.load(inpath / "vectorized_data.npz", allow_pickle=True)

    # assign data to appropriate variables
    X_train_feats = data["X_train_feats"].item()
    X_test_feats = data["X_test_feats"].item()
    y_train = data["y_train"]
    y_test = data["y_test"]

    return X_train_feats, X_test_feats, y_train, y_test


def select_grid(args):
    '''
    Select the grid to use for hyperparameter tuning.
    It depends on the model and the desired size of the grid.

    Args:
    -   args (argparse.Namespace): Parsed arguments.

    Returns:
    -   param_grid (dict): Dictionary of hyperparameters to use for grid search.
    '''

    # define grid names
    param_grid_mapping = {
    ("small", "logistic"): logistic_param_grid_sm,
    ("small", "mlp"): mlp_param_grid_sm,
    ("full", "logistic"): logistic_param_grid,
    ("full", "mlp"): mlp_param_grid
    }

    # get correct grid from mapping using args
    param_grid = param_grid_mapping.get((args.grid_size, args.model))

    return param_grid


def grid_search(X_train_feats, y_train, args):
    '''
    Compare combinations of the hyperparameters using grid search to find the optimal model.

    Args:
    -   X_train_feats (ndarray): Matrix of training features.
    -   y_train (ndarray): Array of training labels.
    -   args (argparse.Namespace): Parsed arguments.

    Returns:
    -   best_model (MLPClassifier or LogisticRegression): Best model found by grid search.
    -   best_params (dict): Best parameters found by grid search.
    '''

    # inform grid search is being done
    logging.info("(2/4) Performing grid search")

    # select grid
    param_grid = select_grid(args)

    # initialize model
    if args.model == "logistic":
        clf = LogisticRegression(max_iter=5000)
    elif args.model == "mlp":
        clf = MLPClassifier(max_iter=5000)

    # initialize grid search
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", verbose=1)

    # do grid search
    grid_search.fit(X_train_feats, y_train)

    # define outpits
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


def evaluate_model(best_model, X_test_feats, y_test):
    '''
    Evaluate the best model on the test set to produce classification report.

    Args:
    -   best_model (MLPClassifier or LogisticRegression): Best model found by grid search.
    -   X_test_feats: Matrix of test features.
    -   y_test: Array of test labels.

    Returns:
    -   classification_report (str): Classification report for the best model as a string.

    '''

    # inform model is being evaluated
    logging.info("(3/4) Evaluating best model")

    # do predictions on test set
    y_pred = best_model.predict(X_test_feats)

    return metrics.classification_report(y_test, y_pred)


def clf_report_to_txt(classification_report, best_params, args, report_outpath):
    '''
    Convert classification report to a text file and save it to the output directory.
    Adds a line with the best model parameters to the classification report also.

    Args:
    -   classification_report (str): Classification report for the best model as a string.
    -   best_model (MLPClassifier or LogisticRegression): Best model found by grid search.
    -   args (argparse.Namespace): Parsed arguments.
    -   report_outpath (pathlib.PosixPath): Path to output data.
    '''

    # inform outputs are being saved
    logging.info("(4/4) Saving outputs!")

    # create string with report name
    report_name = f'{args.model}_report.txt'
    
    # write report to file with specification for model at the top
    with open(report_outpath / report_name, "w") as file:
        file.write(f"Best parameters for {args.model} model: {best_params}, \n {classification_report}")


def save_model(best_model, args, model_outpath):
    '''
    Save model to models folder with name by model type.

    Args:
    -   best_model (MLPClassifier or LogisticRegression): Best model found by grid search.
    -   args (argparse.Namespace): Parsed arguments.
    -   model_outpath (pathlib.PosixPath): Path to models folder.
    '''

    # define model name
    model_name = f"{args.model}_model"

    # dump model
    joblib.dump(best_model, model_outpath / f"{model_name}.joblib")


def main():
    # define paths
    inpath, model_outpath, report_outpath = define_paths()
    
    # config logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # parse input
    args = input_parse()

    # check input
    input_checker(args)

    # inform that run is starting
    logging.info(f"====== Starting classification for {args.model} model ======")

    # load data
    X_train_feats, X_test_feats, y_train, y_test = read_data(inpath)

    # do grid search and get best model
    best_model, best_params = grid_search(X_train_feats, y_train, args)

    # evaluate best model
    classification_report = evaluate_model(best_model, X_test_feats, y_test)

    # save classification report for best model
    clf_report_to_txt(classification_report, best_params, args, report_outpath)

    # save best model object
    save_model(best_model, args, model_outpath)

    # inform user that script is done and outputs are saved
    logging.info("Classification complete! Classification report can be found at {} and model can be found at {}".format(report_outpath / f"{args.model}_report.txt", model_outpath / f"{args.model}_model.joblib") + "\n" + " ")

# run main
if __name__ == "__main__":
    main()