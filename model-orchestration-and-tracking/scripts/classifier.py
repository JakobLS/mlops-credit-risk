import pandas as pd
import argparse
import os
import pickle
import ast

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score, accuracy_score, roc_auc_score
from imblearn.metrics import specificity_score
from skopt import BayesSearchCV
from skopt.space import Integer
from lightgbm import LGBMClassifier

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature

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

    
def train_and_tune_model(X, Y, trainX, trainY, **args):
    args.pop('train_path', None)

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
    log_results_with_mlflow(X, Y, trainX, bs_lgbm.cv_results_, bs_lgbm.best_estimator_, 
                            lgbm_scores, **args)

    # Return the best model and its CV scores
    return bs_lgbm, lgbm_scores


def log_results_with_mlflow(X, Y, trainX, cv_results, best_model, cv_scores, top_n, 
                            test_path, predictions_path, experiment_name, tags, description):
    # Convert tags to a dictionary
    tags = ast.literal_eval(tags)

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
        with mlflow.start_run(description) as run:
            # Add tag on run
            mlflow.set_tags(tags=tags)

            # Store parameters
            mlflow.log_params(dict(top5_df['params'][i]))

            # Store metrics
            mlflow.log_metrics(top5_df[metrics_columns].iloc[i, :].to_dict())

            if i == 0:
                # Log and save the best model. Include signature
                signature = infer_signature(trainX, best_model.predict(trainX))
                mlflow.sklearn.log_model(
                    sk_model=best_model, 
                    artifact_path="model",
                    signature=signature,
                )

                # Make cross validated predictions and plot the results
                predictions = make_cv_predictions(best_model, X, Y)
                vis.plot_confusion_matrix(Y, predictions, log_to_mlflow=True,
                                          title="Confusion Matrix on Validation Set")
                vis.plot_cv_scores(cv_scores, log_to_mlflow=True)

                # Make predictions on test data and log metrics
                test_data = log_test_metrics_to_mlflow(best_model, 
                                                       test_path, predictions_path)

                # Log ROC AUC curve
                vis.plot_ROC_AUC_curve(
                    best_model, 
                    test_data[test_data.columns.difference(['class'])], 
                    test_data['class'], 
                    log_to_mlflow=True,
                    title="ROC AUC Curve on Test Data",
                )
    
    # Add the best model to the Model Register
    register_best_model(experiment_name)


def log_test_metrics_to_mlflow(model, test_path, predictions_path):
    """ 
    Function for making predictions on test data and logging metrics.
    """
    # Load test data and make predictions
    test_data = load_new_data(test_path)
    test_predictions = make_predictions(model, test_data, predictions_path)
    y_true = test_predictions['class'].apply(lambda x: 1 if x == 'good' else 0)
    y_preds = test_predictions['predictions'].apply(lambda x: 1 if x == 'good' else 0)

    # Log metrics
    mlflow.log_metrics({'test_accuracy': accuracy_score(y_true, y_preds),
                        'test_auc': roc_auc_score(y_true, 
                                                  test_predictions['prediction_probs']),
                        'test_recall': recall_score(y_true, y_preds),
                        'test_specificity': specificity_score(y_true, y_preds, 
                                                                average="weighted")
                        })
    
    return test_data


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
        order_by=["metrics.mean_val_auc DESC"],
    )

    if len(best_run) == 0:
        raise "No models found. Can't register in the Model Register."
    
    # Register the best model
    model_uri=f"runs:/{best_run[0].info.run_id}/model" 
    mlflow.register_model(
        model_uri=model_uri,
        name=f"best-model-{experiment_name}"
    )


def main(train_path, test_path, predictions_path, top_n, 
         experiment_name, tags, description):

    # Load the data
    data = load_train_data(train_path)

    # Make simple data preparations
    X, Y = dp.prepare_data(data)

    # Split into train and test sets. 
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=89)

    # Train and tune model
    model, scores = train_and_tune_model(X, Y, trainX, trainY, **args)




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
        "--predictions_path", 
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
        default=5,
        help="The top n best models to track with MLflow"
    )
    parser.add_argument(
        "--tags",
        default="{'model': 'LightGBM', 'developer': 'Jakob'}",
        help="Tags for identifying the model and/or run"
    )
    parser.add_argument(
        "--description",
        default="Training model for Predicting the Credit Risk of an Individual",
        help="A description of the experiment."
    )
    args = vars(parser.parse_args())


    set_mlflow(args['experiment_name'])
    main(**args)






