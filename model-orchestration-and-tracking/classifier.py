import pandas as pd
# import numpy as np
import argparse
import os
import pickle

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score
from imblearn.metrics import specificity_score
from skopt import BayesSearchCV
from skopt.space import Integer
from lightgbm import LGBMClassifier

import data_preparation as dp 
from model_predictions import make_cv_predictions, make_predictions
import visualisations as vis


def load_train_data(train_data_path):
    # Start by loading the train data
    data = pd.read_csv(train_data_path)

    # Randomly shuffle the data to minimise the effect of randomness on our results
    data = data.sample(frac=1.0, random_state=55)

    return data


def load_new_data(test_data_path):
    new_data = pd.read_csv(test_data_path)
    return new_data


def train_and_tune_model(X, Y, trainX, trainY):
    # Specify model and subsequent hyperparameters to tune
    lgbm = LGBMClassifier(random_state=99)
    parameters = {'clf__n_estimators': Integer(10, 200, prior='uniform'),
                  'clf__max_depth': Integer(2, 8, prior='uniform'),
                  'clf__num_leaves': Integer(20, 60, prior='uniform'),
    }

    # Create data preprocessor and pipeline
    preprocessor = dp.create_data_preprocessor(X.columns)
    pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("feaure_selector", dp.feature_selector()),
            ("clf", lgbm),
    ])

    # Specify scoring metrics
    scoring = {
            "auc"        : "roc_auc",
            "specificity": make_scorer(specificity_score, average="weighted"),
            "recall"     : make_scorer(recall_score, pos_label="good"),
            "accuracy"   : "accuracy",
    }

    # Perform hyperparameter tuning with BayesSearchCV over 10 folds with AUC as refit metric.
    # Try only 5 combinations to speed things up
    gs_lgbm = BayesSearchCV(pipeline, parameters, cv=10, scoring=scoring, 
                           refit="auc", random_state=500, n_iter=5)

    # Fit the BayesSearchCV object to the train data
    gs_lgbm.fit(trainX, trainY)

    # Run nested cross-validation over 10 folds
    lgbm_scores = cross_validate(gs_lgbm, X, Y, cv=10, n_jobs=-1, verbose=1,
                            return_train_score=True, scoring=scoring)
    
    # Return the best model and its CV scores
    return gs_lgbm, lgbm_scores


def main(train_path, test_path, prediction_path):
    # Load the data
    data = load_train_data(train_path)

    # Make simple data preparations
    X, Y = dp.prepare_data(data)

    # Split into train and test sets. 
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=89)

    # Train and tune model
    model, scores = train_and_tune_model(X, Y, trainX, trainY)

    # Make cross validated predictions
    predictions = make_cv_predictions(model, X, Y)

    # Make predictions on new data
    new_data = load_new_data(test_path)
    new_predictions = make_predictions(model, new_data, prediction_path)

    vis.plot_confusion_matrix(Y, predictions)
    vis.plot_ROC_AUC_curve(model, testX, testY)
    vis.plot_cv_scores(scores)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", 
        default="../datasets/cleaned/train/credit_train.csv",
        help="Path to train data",
        )
    parser.add_argument(
        "--test_path", 
        default="../datasets/cleaned/test/credit_test.csv",
        help="Path to unseen test data",
        )
    parser.add_argument(
        "--output_path", 
        default="../datasets/predictions/predictions.csv",
        help="Path to where predictions will be stored",
        )
    args = parser.parse_args()

    main(args.train_path, args.test_path, args.output_path)








