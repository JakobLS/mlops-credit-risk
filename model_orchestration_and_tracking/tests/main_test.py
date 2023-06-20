import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
from prefect import flow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model_orchestration_and_tracking.main import train_and_tune_model
from model_orchestration_and_tracking.scripts.data_preparation import load_train_data, prepare_data
from model_orchestration_and_tracking.scripts.model_predictions import make_predictions

base_path = "7.Final-Project/Capstone-Project/MLOps-project/model_orchestration_and_tracking/tests/"
TRAIN_PATH = base_path + "data_for_tests/train/credit_train.csv"
MODEL_PATH = base_path + "model_for_test/model_test.pkl"
TEST_PATH = base_path + "data_for_tests/test/credit_test.csv"
PREDICTION_PATH = base_path + "/data_for_tests/predictions/predictions.csv"



def test_load_train_data():
    data = flow(load_train_data.fn)(TRAIN_PATH)
    assert data.shape == (800, 21)


def test_train_and_tune_model():
    """ Train, tune and 5-fold cross-validate a model on the train set.
        Test whether the cross-validated AUC is at least 77%. 
    """
    # Load the data
    data = pd.read_csv(TRAIN_PATH)

    # Make simple data preparations
    X, Y = prepare_data(data)

    # Split into train and test sets. 
    trainX, _, trainY, _ = train_test_split(X, Y, test_size=0.2, 
                                                  random_state=100)
    
    kwargs = {"n_iter": 5}
    model, scores = flow(train_and_tune_model.fn)(X, Y, trainX, trainY, **kwargs)

    # Save the model
    joblib.dump(model.best_estimator_, MODEL_PATH)

    mean_auc = np.mean(scores["test_auc"]) * 100
    assert mean_auc >= 77


def test_make_predictions():
    """ Test that the model AUC is at least 75% on the test set.
    """
    # Load the recently saved model
    model_pipeline = joblib.load(MODEL_PATH)

    # Load data
    data = pd.read_csv(TEST_PATH)

    predictions = make_predictions(model_pipeline, data, PREDICTION_PATH)
    auc = roc_auc_score(predictions['class'], predictions['prediction_probs']) * 100

    assert auc >= 75


