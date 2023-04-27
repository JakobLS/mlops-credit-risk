
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer
from imblearn.metrics import specificity_score
from skopt import BayesSearchCV
from skopt.space import Integer
from lightgbm import LGBMClassifier

from data_preparation import prepare_data, feature_selection
from model_predictions import make_cv_predictions, make_predictions
import visualisations


def load_train_data():
    # Start by loading the train data
    data = pd.read_csv("../datasets/cleaned/train/credit_train.csv")
    return data


def load_new_data():
    new_data = pd.read_csv("../datasets/cleaned/test/credit_test.csv")
    return new_data


def train_and_tune_model(X, Y, trainX, trainY):
    # Specify model and subsequent hyperparameters to tune
    lgbm = LGBMClassifier(random_state=99)
    parameters = {'n_estimators': Integer(10, 200, prior='uniform'),
                  'max_depth': Integer(2, 8, prior='uniform'),
                  'num_leaves': Integer(20, 60, prior='uniform'),
    }

    # Specify scoring metrics
    scoring = {
            "auc"        : "roc_auc",
            "specificity": make_scorer(specificity_score, average="weighted"),
            "recall"     : "recall",
            "accuracy"   : "accuracy",
    }

    # Perform hyperparameter tuning with BayesSearchCV over 10 folds with AUC as refit metric.
    # Try only 5 combinations to speed things up
    gs_lgbm = BayesSearchCV(lgbm, parameters, cv=10, scoring=scoring, 
                        refit="auc", random_state=500, n_iter=5)

    # Fit the BayesSearchCV object to the train data
    gs_lgbm.fit(trainX, trainY)

    # Run nested cross-validation over 10 folds
    lgbm_scores = cross_validate(gs_lgbm, X, Y, cv=10, n_jobs=-1, verbose=1,
                            return_train_score=True, scoring=scoring)
    
    # Return the best model and its CV scores
    return gs_lgbm, lgbm_scores


def main():
    # Load the data
    data = load_train_data()

    # Make certain data preparations and select top features
    data = prepare_data(data)
    X, Y = feature_selection(data)

    # Split into train and test sets. 
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=89)

    # Train and tune model
    model, scores = train_and_tune_model(X, Y, trainX, trainY)

    # Make cross validated predictions
    predictions = make_cv_predictions(model, X, Y)

    # Make predictions on new data
    new_data = load_new_data()
    new_predictions = make_predictions(model, new_data)

    visualisations.plot_confusion_matrix(Y, predictions)
    visualisations.plot_ROC_AUC_curve(model, testX, testY)
    visualisations.plot_cv_scores(scores)




if __name__ == "__main__":
    main()








