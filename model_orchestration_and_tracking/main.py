import pandas as pd
import argparse
import os
import pickle
import ast
from datetime import timedelta
from dotenv import load_dotenv

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

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.infrastructure import Process

try:
    import scripts.data_preparation as dp
    import scripts.model_predictions as mp
    import scripts.visualisations as vis
except:
    import model_orchestration_and_tracking.scripts.data_preparation as dp
    import model_orchestration_and_tracking.scripts.model_predictions as mp
    import model_orchestration_and_tracking.scripts.visualisations as vis

# Set a unified random_state across the file
random_state = 100

# The .env variable is only used for local deployment
load_dotenv('.env')
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "datasets/cleaned/train/credit_train.csv")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "datasets/cleaned/test/credit_test.csv")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "datasets/predictions/predictions.csv")
REGISTRY_PREDICTIONS_PATH = os.getenv("REGISTRY_PREDICTIONS_PATH", 
                                      "datasets/predictions/registry_predictions.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5050")
EXPERIMENT_NAME = "Credit Risk Prediction Model"


def set_mlflow(experiment_name):
    # Specify MLflow parameters
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    
@task(name="train_and_tune_the_model")
def train_and_tune_model(X, Y, trainX, trainY, **kwargs):
    # Specify model and subsequent hyperparameters to tune
    lgbm = LGBMClassifier(random_state=random_state)
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
    n_iter = kwargs['n_iter']
    bs_lgbm = BayesSearchCV(pipeline, parameters, cv=10, scoring=scoring, 
                           refit="auc", random_state=random_state, n_iter=n_iter)

    # Fit the BayesSearchCV object to the train data
    bs_lgbm.fit(trainX, trainY)

    # Run nested cross-validation over 10 folds
    lgbm_scores = cross_validate(bs_lgbm, X, Y, cv=10, n_jobs=-1, verbose=1,
                                 return_train_score=True, scoring=scoring)

    # Return the best model and its CV scores
    return bs_lgbm, lgbm_scores


@task(name="log_all_results_with_mlflow")
def log_results_with_mlflow(X, Y, trainX, cv_results, best_model, cv_scores, 
                            test_path, **kwargs):
    # Convert tags to a dictionary
    tags = ast.literal_eval(kwargs['tags'])

    # Create dataframe and rename columns from test to validate
    top5_df = pd.DataFrame(cv_results)
    cols = top5_df.columns
    top5_df = top5_df.rename(
        columns=dict(zip(cols, 
                        [c.replace('test', 'val') for c in cols])))
    metrics_columns = [c for c in top5_df.columns if not c.startswith('param')]

    # Extract the top_n best results based on validation AUC
    top_n = kwargs['top_n']
    top5_df = top5_df.sort_values(
        by='mean_val_auc', 
        ascending=False,
    ).iloc[:top_n, :].reset_index(drop=True)

    # Log the results with MLflow
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    for i in range(top_n):
        with mlflow.start_run(description=kwargs['description'], 
                              experiment_id=experiment_id) as run:
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
                predictions = mp.make_cv_predictions(best_model, X, Y)
                vis.plot_confusion_matrix(Y, predictions, log_to_mlflow=True,
                                          title="Confusion Matrix on Validation Set")
                vis.plot_cv_scores(cv_scores, log_to_mlflow=True)

                # Make predictions on test data and log metrics
                test_data = log_test_metrics_to_mlflow(best_model, 
                                                       test_path, 
                                                       kwargs['predictions_path'],
                                                       )

                # Log ROC AUC curve
                vis.plot_ROC_AUC_curve(
                    best_model, 
                    test_data[test_data.columns.difference(['class'])], 
                    test_data['class'], 
                    log_to_mlflow=True,
                    title="ROC AUC Curve on Test Data",
                )


def log_test_metrics_to_mlflow(model, test_path, predictions_path):
    """ 
    Function for making predictions on test data and logging metrics.
    """
    # Load test data and make predictions
    test_data = dp.load_new_data(test_path)
    test_predictions = mp.make_predictions(model, test_data, predictions_path)
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


@task(name="register_the_best_model")
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
        name=experiment_name,
    )


@flow(name="recurring_credit_risk_model",
      task_runner=SequentialTaskRunner)
def main(train_path, test_path, registry_predictions_path,
         experiment_name, **kwargs):
    
    # Set the mlflow experiment name
    set_mlflow(experiment_name)
    
    # Load the data
    data = dp.load_train_data(train_path)

    # Make simple data preparations
    X, Y = dp.prepare_data(data)

    # Split into train and test sets. 
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, 
                                                  random_state=random_state)

    # Train and tune model
    model, scores = train_and_tune_model(X, Y, trainX, trainY, **kwargs)

    # Log results with MLflow
    log_results_with_mlflow(X, Y, trainX, model.cv_results_, model.best_estimator_, 
                            scores, test_path, **kwargs)
    
    # Add the best model to the Model Register
    register_best_model(experiment_name)

    # # Make predictions with model from Model Registry in production stage
    # predictions = mp.make_predictions_with_model_registry_model(
    #     model_name=experiment_name, 
    #     data_path=test_path, 
    #     output_path=registry_predictions_path,
    #     stage="Production",
    # )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", 
        default=TRAIN_DATA_PATH,
        help="Path to train data",
        )
    parser.add_argument(
        "--test_path", 
        default=TEST_DATA_PATH,
        help="Path to unseen test data",
        )
    parser.add_argument(
        "--predictions_path", 
        default=PREDICTIONS_PATH,
        help="Path to where predictions will be stored",
        )
    parser.add_argument(
        "--registry_predictions_path", 
        default=REGISTRY_PREDICTIONS_PATH,
        help="Path to where predictions made by the model fetched from the Model Registry are stored",
        )
    parser.add_argument(
        "--experiment_name",
        default=EXPERIMENT_NAME,
        help="MLFlow experiement name"
    )
    parser.add_argument(
        "--top_n",
        default=5,
        help="The top n best models to track with MLflow"
    )
    parser.add_argument(
        "--n_iter",
        default=5,
        help="The number of parameter combinations to try during hyper-parameter tuning. Higher values will take longer but can also yield better models."
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
    kwargs = vars(parser.parse_args())

    deployment = Deployment.build_from_flow(
        flow=main,
        name="model_training_and_prediction_weekly",
        parameters={'kwargs': {**kwargs}},
        schedule=CronSchedule(cron="0 3 * * 1", timezone="Europe/Madrid"), # Run it at 03:00 am every Monday
        infrastructure=Process(working_dir=os.getcwd()), # Run flows from current local directory
        version=1,
        work_queue_name="credit_risk_model_queue",
        # tags=['dev'],
    )

    deployment.apply()


