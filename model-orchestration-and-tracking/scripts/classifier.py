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

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import data_preparation as dp 
from model_predictions import make_cv_predictions, make_predictions
import visualisations as vis



def set_mlflow(experiment_name):
    # Specify MLflow parameters
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)


def load_train_data(train_data_path):
    # Start by loading the train data
    data = pd.read_csv(train_data_path)

    # Randomly shuffle the data to minimise the effect of randomness on our results
    data = data.sample(frac=1.0, random_state=55)

    return data


def load_new_data(test_data_path):
    new_data = pd.read_csv(test_data_path)
    return new_data


def train_and_tune_model(X, Y, trainX, trainY, top_n, experiment_name):
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
    bs_lgbm = BayesSearchCV(pipeline, parameters, cv=10, scoring=scoring, 
                           refit="auc", random_state=500, n_iter=5)

    # Fit the BayesSearchCV object to the train data
    bs_lgbm.fit(trainX, trainY)

    # Run nested cross-validation over 10 folds
    lgbm_scores = cross_validate(bs_lgbm, X, Y, cv=10, n_jobs=-1, verbose=1,
                                 return_train_score=True, scoring=scoring)
    
    # Log results with MLflow
    log_results_with_mlflow(X, Y, bs_lgbm.cv_results_, bs_lgbm.best_estimator_, 
                            top_n, experiment_name, tags, lgbm_scores, test_path, 
                            predictions_path)

    # Return the best model and its CV scores
    return bs_lgbm, lgbm_scores


def log_results_with_mlflow(X, Y, cv_results, best_model, top_n, 
                            experiment_name, tags, cv_scores, test_path, 
                            predictions_path):
    # Extract the top_n best results based on AUC
    top5_df = pd.DataFrame(cv_results).sort_values(
        by='mean_test_auc', 
        ascending=False,
    ).iloc[:top_n, :].reset_index(drop=True)

    # Rename columns from test to validate
    cols = top5_df.columns
    top5_df = top5_df.rename(
        columns=dict(zip(cols, 
                        [c.replace('test', 'val') for c in cols])))
    metrics_columns = [c for c in top5_df.columns if not c.startswith('param')]

    # Log the results with MLflow
    for i in range(top_n):
        with mlflow.start_run() as run:
            # Add tag on run
            mlflow.set_tags(tags=tags)

            # Store parameters
            mlflow.log_params(dict(top5_df['params'][i]))

            # Store metrics
            mlflow.log_metrics(top5_df[metrics_columns].iloc[i, :].to_dict())

            if i == 0:
                # Log and save the best model
                # best_model_path = "model"
                mlflow.sklearn.log_model(
                    sk_model=best_model, 
                    artifact_path="model",
                )
    
    # Add the best model to the Model Register
    register_best_model(experiment_name)


def register_best_model(experiment_name):
    """ 
    Register the best model in the Model Registry.
    """

    # Get current experiment name
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Find the run with highest AUC
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.mean_test_auc DESC"],
    )

    if len(best_run) == 0:
        raise "No models found. Can't register in the Model Register."
    
    # Register the best model
    model_uri=f"runs:/{best_run[0].info.run_id}/model" 
    mlflow.register_model(
        model_uri=model_uri,
        name=f"best-model-{experiment_name}"
    )


def main(train_path, test_path, output_path, 
         top_n, experiment_name, tags):
    
    # Convert tags to a dictionary
    tags = ast.literal_eval(tags)

    # Load the data
    data = load_train_data(train_path)

    # Make simple data preparations
    X, Y = dp.prepare_data(data)

    # Split into train and test sets. 
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=89)

    # Train and tune model
    model, scores = train_and_tune_model(X, Y, trainX, trainY, top_n, experiment_name)

    # Make cross validated predictions
    predictions = make_cv_predictions(model, X, Y)

    # Make predictions on new data
    new_data = load_new_data(test_path)
    new_predictions = make_predictions(model, new_data, output_path)

    vis.plot_confusion_matrix(Y, predictions)
    vis.plot_ROC_AUC_curve(model, testX, testY)
    vis.plot_cv_scores(scores)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", 
        default="../../datasets/cleaned/train/credit_train.csv",
        help="Path to train data",
        )
    parser.add_argument(
        "--test_path", 
        default="../../datasets/cleaned/test/credit_test.csv",
        help="Path to unseen test data",
        )
    parser.add_argument(
        "--output_path", 
        default="../../datasets/predictions/predictions.csv",
        help="Path to where predictions will be stored",
        )
    parser.add_argument(
        "--experiment_name",
        default="Credit Risk Prediction Model",
        help="MLFlow experiement name"
    )
    parser.add_argument(
        "--top_n",
        default=3,
        help="The top n best models to track with MLflow"
    )
    parser.add_argument(
        "--tags",
        default="{'model': 'LightGBM'}",
        help="Tags for identifying the model and/or run"
    )
    args = vars(parser.parse_args())


    set_mlflow(args['experiment_name'])
    main(**args)






