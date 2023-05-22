#!/usr/bin/env bash
# create virtual environment called text_classification
python3 -m venv txt_classification_env

# activate virtual environment
source ./txt_classification_env/bin/activate

# install requirements
echo "[INFO] Installing requirements..."
python3 -m pip install -r requirements.txt

# run script for vectorizing
python3 src/vectorize.py

# run script for classifying using logistic regression
python3 src/classify.py -m "logistic"

# run script for classifying using neural network
python3 src/classify.py -m "mlp"

# deactivate virtual environment
deactivate txt_classification_env

