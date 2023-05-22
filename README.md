# Assignment 2: Text Classification

## Repository Overview
1. [Description](#description)
2. [Repository Tree](#tree)
3. [Usage](#gusage)
4. [Modified Usage](#musage)
5. [Results](#results)
6. [Discussion](#discussion)

## Description <a name="description"></a>
This repository includes the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 2 in the course "Language Analytics" at Aarhus University.

The analysis sets out to demarcate real and fake news from the [Fake News Dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). To achieve this, logistic regression and neural network (MLP) classifiers are trained on the task, both optimized through a grid search for hyperparameter tuning.
</br></br>

## Repository Tree <a name="tree"></a>

```
├── README.md                   
├── assign_desc.md              
├── in
│   ├── fake_or_real_news.csv           <----- the Fake News Dataset
│   └── vectorized_data.npz             <----- the vectorized dataset (output for vectorize-data.py)
├── models
│   ├── logistic_model.joblib           
│   ├── mlp_model.joblib                
│   └── vectorizer.joblib               <----- example word vectorizer model (used to produce vectorized_data.npz)
├── out
│   ├── logistic_report.txt             
│   └── mlp_report.txt                  
├── requirements.txt            
├── run.sh 
├── setup.sh
├── src
│   ├── classify.py        <----- script for classification with both models
│   ├── parameters.py      <----- script containing the parameter grids
│   └── vectorize.py       <----- script for vectorization of the dataset

```
<br>

## Usage <a name="gusage"></a>
To run the analysis, you must have Python3 installed and clone this repository. If satisfying these requirements, type the following command from the root directory of the project:
```
bash run.sh
```

This achieves the following:
* Create a virtual environment
* Install requirements to that environment
* Vectorize the dataset (`vectorize.py`)
* Classify the vectorized data using both types of classifiers (`classify.py`)
* Deactivate the environment

The models for vectorization, the logistic regression classifier and the neural network classifier are all found in `models`. The classification reports for the best models for logistic regression and neural network are found in the `out` directory.
</br></br>

## Modified Usage <a name="musage"></a>
### Setup
If wanting to run the analysis part-by-part and use non-default parameters, this is also feasable by adhering to the following instructions. You should also follow this modified usage in case the analysis is too computationally heavy as modifications can fix this. <br>

Firstly, setup the virtual environment and install requirements with the shell script as such:
```
bash setup.sh
```

### Vectorizing the Data
The data should first be vectorized using `vectorize.py`. Here you may specify the following arguments:

| Parameter       | Default Value | Description |
|-----------------|--------------|--------------|
| `--vectorizer`      | "tfdif"      | Vectorizer to use (either "bow" or "tfidf")|
| `--max_features`    | 3000         | Maxiumum features to be identified|
| `--ngram`           | (1,2)        | Which ngrams should be used|

For more information of the parameters, see [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

For instance, it can be done as such:
```
# vectorize using BOW, a maximum of 2000 features and only bigrams
python src/vectorize.py --vectorizer "bow" --max_features 2000 --ngram (2,2)
```
The exemplary file `vectorized_data.npz` have been created using the default parameters.
</br></br>

### Classifying the Data
After having applied your preferred vectorizer, the classification of the data can be done. <br>

This is simply achieved by running `classify.py` with the `--model` argument specifying whether it should be based on Logistic Regression or a Neural Network:
```
# run logistic regression classification
python src/classify-data.py --model "logistic"

# run neural network classification
python src/classify-data.py --model "mlp"
```
<br>

The model parameters for the classifications are selected using a grid search with cross-validation. If you experience slow running times for the grid search, you can use a smaller grid that requires less computiation power using the `--grid_size` argument as such:
```
# run neural network classification with reduced grid
python src/classify.py --model "mlp" --grid_size "small"
```
If you want to further reduce the grids or test new combinations of paramters, please refer to the `parameters.py` file to modify parameter grids.
</br></br>

## Results <a name="results"></a>
The following results are for the two models with the default values for all arguments.

### Logistic Regression Classifier
Model parameters for best model: `'C': 4, 'intercept_scaling': 0.5, 'penalty': 'l1', 'solver': 'saga', 'tol': 0.01` <br>
|            | precision | recall | f1-score | support |
| ----       | --------- | ------ | -------- | ------- |
| FAKE       | 0.91      | 0.93   | 0.92     | 614     |
| REAL       | 0.93      | 0.81   | 0.92     | 653     |
|            |           |        |          |         |
|accuracy    |           |        | 0.92     | 1267    |
|macro avg   | 0.92      | 0.92   | 0.92     | 1267    |
|weighted avg| 0.92      | 0.92   | 0.92     | 1267    |


### Neural Network (MLP) Classifier
Model parameters for best model: `'hidden_layer_sizes': (25, 25)` <br>
|            | precision | recall | f1-score | support |
| ----       | --------- | ------ | -------- | ------- |
| FAKE       | 0.89      | 0.92   | 0.90     | 614     |
| REAL       | 0.92      | 0.89   | 0.91     | 653     |
|            |           |        |          |         |
|accuracy    |           |        | 0.91     | 1267    |
|macro avg   | 0.91      | 0.91   | 0.91     | 1267    |
|weighted avg| 0.91      | 0.91   | 0.91     | 1267    |

<br>

## Discussion <a name="discussion"></a>
The results reveal that both models perform well on the task, achieving F1-scores beyond 0.9. Hence, we conclude that it was possible to a great extent to demarcate real from fake news in the dataset. <br>

Surprisingly perhaps, a slight edge can be found for the Logistic Regression classifier as it achieves an F1-score of 0.92 compared to 0.91 for the Neural Network classifier. Still, it should be emphasized that these results could have swayed the other way given small changes such as using a different vectorizer. This is naturally possible to investigate by following the *Modified Usage* section. Also, altering the parameters grids in `parameters.py` may lead to even better performance for both models.

