""" vectorize.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This file vectorizes the Fake News Dataset by using a CountVectorizer or TfidfVectorizer.
    The vectorized data is saved as an npz file so that it can be used for classify.py.

Usage:
    $ python src/vectorize.py -v tfidf -f 3000 -n (1,2)
"""


# import libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import pandas as pd
import os
import argparse
import numpy as np
import logging


def define_paths():
    '''
    Define paths for input and output data.

    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to output data.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "in"

    # define output dir
    outpath = path.parents[1] / "models"

    return inpath, outpath


def input_parse():
    '''
    Parses input arguments.
    It is possible to specify the vectorizer type to use, the maximum number of features and the ngram range.

    Returns:
    -   args (argparse.Namespace): Parsed arguments.
    '''
    
    # initialize parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-v", "--vectorizer", default="tfidf", help="vectorizer to use")
    parser.add_argument("-f", "--max_features", default=3000, help="number of features for vectorizer")
    parser.add_argument("-n", "--ngram", default=(1,2), help="ngram range for vectorizer")

    # parse arguments
    args = parser.parse_args()

    return args


def input_checker(args):
    '''
    Check if the input arguments are correct and return error message if not.
    
    Args:
    -   args (argparse.Namespace): Parsed arguments.
    '''

    # check if vectorizer is correct
    if args.vectorizer not in ["bow", "tfidf"]:
        raise ValueError("Vectorizer must be 'bow' or 'tfidf'")
    
    # check if max_features is correct
    if int(args.max_features) < 1:
        raise ValueError("max_features must be > 0")
    
    # check if ngram is a tuple with two integers
    if not isinstance(args.ngram, tuple) or len(args.ngram) != 2 or not all(isinstance(n, int) for n in args.ngram):
        raise ValueError("ngram must be a tuple with two integers")
    

def read_data(filename):
    '''
    Read data from csv file.

    Args:
    -   filename (str): Path to csv file.

    Returns:
    -   X (pandas.core.series.Series): Series with text data
    -   y (pandas.core.series.Series): Series with labels
    '''

    # inform that data is being read
    logging.info("(1/3) Reading data from {}".format(filename))

    # read data
    data = pd.read_csv(filename)

    # split data
    X = data["text"]
    y = data["label"]

    return X, y


def vectorize_data(X_train, X_test, args):
    '''
    Vectorizing the data using a CountVectorizer or TfidfVectorizer.

    Args:
    -   X_train (pandas.core.series.Series): Series with text data for training
    -   X_test (pandas.core.series.Series): Series with text data for testing
    -   args (argparse.Namespace): Parsed arguments.

    Returns:
    -   vectorizer: Fitted vectorizer.
    -   X_train_feats (ndarray): Vectorized training data.
    -   X_test_feats (ndarray): Vectorized testing data.
    '''

    # initialize vectorizer
    if args.vectorizer == "bow":
        vectorizer = CountVectorizer(ngram_range = args.ngram,
                             lowercase =  True,       
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = int(args.max_features))
    
    elif args.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range = args.ngram,
                             lowercase =  True,       
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = int(args.max_features))
    
    # inform that vectorizer is being fitted
    logging.info("(2/3) Fitting {} vectorizer".format(args.vectorizer))

    # fit and transform vectorizer
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    return vectorizer, X_train_feats, X_test_feats


def save_data(X_train_feats, X_test_feats, y_train, y_test, args, inpath):
    '''
    Saving the data as npz file (compressed numpy array) for use in classify.py

    Args:
    -   X_train_feats (ndarray): Vectorized training data.
    -   X_test_feats (ndarray): Vectorized testing data.
    -   y_train (pandas.core.series.Series): Series with labels for training
    -   y_test (pandas.core.series.Series): Series with labels for testing
    -   args (argparse.Namespace): Parsed arguments.
    -   inpath (pathlib.PosixPath): Path to input data.
    '''

    # inform that data is being saved
    logging.info("(3/3) Saving data to {}".format(os.path.join(inpath, "vectorized_data.npz")))

    # save data
    np.savez_compressed(os.path.join(inpath, "vectorized_data.npz"),
                        X_train_feats=X_train_feats,
                        X_test_feats=X_test_feats,
                        y_train=y_train,
                        y_test=y_test,
                        vectorizer=args.vectorizer)

def main():
    # get paths
    inpath, outpath = define_paths()

    # config logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # parse input
    args = input_parse()

    # check if inputs are okay
    input_checker(args)

    # inform that run is starting
    logging.info(f"====== Starting vectorization ======")

    # read data
    X, y = read_data(os.path.join(inpath, "fake_or_real_news.csv"))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # vectorize data
    vectorizer, X_train_feats, X_test_feats = vectorize_data(X_train, X_test, args)

    # save vectorizer
    joblib.dump(vectorizer, os.path.join(outpath, "vectorizer.joblib"))

    # save data
    save_data(X_train_feats, X_test_feats, y_train, y_test, args, inpath)

    # inform that run is complete
    logging.info("Vectorization complete! Data can be found as {} and the vectorizer as {}".format(os.path.join(inpath, "vectorized_data.npz"), os.path.join(outpath, "vectorizer.joblib")) + "\n" + " ")

# run main
if __name__ == "__main__":
    main()