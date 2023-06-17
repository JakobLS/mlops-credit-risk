import json
import os
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import storage

import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.infrastructure import Process
from pymongo import MongoClient

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (DataDriftTab, 
                                      ClassificationPerformanceTab)
from evidently.model_profile import Profile
from evidently.model_profile.sections import (DataDriftProfileSection,
                                              ClassificationPerformanceProfileSection)

# Environmental variables will only be fetched from .env file when run locally
REFERENCE_DATA = os.getenv("REFERENCE_DATA", "datasets/cleaned/reference/reference1.csv")
REPORT_TIME_WINDOW_MINUTES = int(os.getenv("REPORT_TIME_WINDOW_MINUTES", 180))
EVIDENTLY_TIME_WIDTH_MINS = int(os.getenv("EVIDENTLY_TIME_WIDTH_MINS", 720))
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://mongo.:27017/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5050")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Credit Risk Prediction Model")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", 
                                           "/secrets/mlops-credit-risk-secret.json")

# Define MongoDB client
mongo_client = MongoClient(MONGODB_ADDRESS)



@task(name="set_mlflow")
def set_mlflow():
    # Specify MLflow parameters
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


@task
def load_reference_data(reference_path):
    """ Load the reference data (including target) and the model. Make predictions.
        Evidently requires both the target and prediction to be present in both the
        reference data and new data. 
    """
    try:
        # Load the model and reference data
        model = mlflow.sklearn.load_model(model_uri=f"models:/{EXPERIMENT_NAME}/Production")
        reference_data = pd.read_csv(f"gs://mlops-credit-risk/{reference_path}")

        # Prepare the data by cleaning the column names and excluding the class
        reference_data.columns = reference_data.columns.str.strip()
        X = reference_data[reference_data.columns.difference(['target'])]

        # Make predictions
        reference_data['prediction'] = model.predict(X)

        return reference_data
    
    except Exception as e:
        print("\n***\nNo predictions were made; No model in Production stage or incorrect Model URL.\n***\n")
        print(e)


def load_mongo_data_between(collection_name, from_dt, to_dt):
    """ Fetch data from a MongoDB collection between two dates
    """
    collection = mongo_client.get_database("credit_risk_service").get_collection(collection_name)
    results = collection.find({'created_at': {'$gte': from_dt, '$lt': to_dt}})
    return list(results)


@task
def fetch_recent_data(from_dt, to_dt, fetch_by_date=True, nbr_samples=100):
    """ Fetch data from MongoDB.
    """
    data = pd.DataFrame()
    if fetch_by_date:
        data = load_mongo_data_between("credit_risk_data", from_dt, to_dt)
        data = pd.DataFrame(data)

    # If there are fewer than 30 samples for the selected time frame, 
    # fetch by number of samples
    if len(data.index) < 30:
        db = mongo_client.get_database("credit_risk_service")
        data = db.get_collection("credit_risk_data").find()

        # By default, use the most recent 100 samples
        data = pd.DataFrame(list(data)[-nbr_samples:])

    return data


@task
def run_evidently(reference_data, new_data):
    """ Create an Evidently Dashboard by comparing statistics between the reference and live data.
    """
    # Specify Profile
    profile = Profile(sections=[DataDriftProfileSection(), 
                                ClassificationPerformanceProfileSection()])
    mapping = ColumnMapping(prediction='prediction',
                            target='target',
                            categorical_features=[
                                'checking_status',
                                'credit_history',
                                'purpose',
                                'savings_status',
                                'employment',
                                'personal_status',
                                'other_parties',
                                'property_magnitude',
                                'other_payment_plans',
                                'housing',
                                'job',
                                'own_telephone',
                                'foreign_worker',
                            ],
                            numerical_features=[
                                'duration', 
                                'credit_amount',
                                'installment_commitment',
                                'residence_since',
                                'age',
                                'existing_credits',
                                'num_dependents',
                            ],
                            datetime_features=[])
    profile.calculate(reference_data, new_data, mapping)

    # Create Dashboard
    dashboard = Dashboard(tabs=[DataDriftTab(),
                                ClassificationPerformanceTab(verbose_level=2)])
    dashboard.calculate(reference_data, new_data, mapping)

    return json.loads(profile.json()), dashboard


def upload_report_to_gcp_bucket(filename):
    storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    bucket = storage_client.get_bucket("mlops-credit-risk")
    blob = bucket.blob(f"reports/{filename}.html")

    with open("evidently_report.html", "rb") as html_report:
        blob.upload_from_file(html_report)


@task
def save_report(profile, report, unique_name=False):
    # Save the profile to MongoDB
    db = mongo_client.get_database("credit_risk_service")
    db.get_collection("evidently_report").insert_one(profile)

    # If unique_name is set, use current time in the name
    if unique_name:
        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M")
        filename = f"evidently_report_{current_time}"
    else: 
        filename = "evidently_report"
    
    # Save the html report. We need to make a work around by first saving it locally
    report.save("evidently_report.html")
    upload_report_to_gcp_bucket(filename=filename)


@flow(task_runner=SequentialTaskRunner)
def evidently_report():
    set_mlflow()
    reference_data = load_reference_data(REFERENCE_DATA)
    from_dt = datetime.now() - timedelta(minutes=EVIDENTLY_TIME_WIDTH_MINS)
    new_data = fetch_recent_data(
        from_dt, 
        datetime.now(), 
        fetch_by_date=True, 
        nbr_samples=100,
    )
    profile, report = run_evidently(reference_data, new_data)
    save_report(profile, report, unique_name=True)




if __name__ == "__main__":

    deployment = Deployment.build_from_flow(
        flow=evidently_report,
        name="generate_evidently_report",
        schedule=IntervalSchedule(
            interval=timedelta(minutes=REPORT_TIME_WINDOW_MINUTES),
            timezone='Europe/Madrid'),
        infrastructure=Process(working_dir=os.getcwd()), # Run flows from current local directory
        version=1,
        work_queue_name="evidently_report_queue",
    )

    deployment.apply()


