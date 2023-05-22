#!/usr/bin/env bash
# create virtual environment called text_classification
python3 -m venv txt_classification_env

# activate virtual environment
source ./txt_classification_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt