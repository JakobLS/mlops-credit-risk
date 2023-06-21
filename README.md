
# MLOps Project - Credit Risk Prediction

The purpose of this project is to build out a simple classification model to predict the credit risk of bank customers. The resulting model will then be deployed to production using MLOps best practices.

The dataset can be downloaded from Kaggle via [this](https://www.kaggle.com/btolar1/weka-german-credit) link.

The dataset contains 1000 entries with 20 categorial/symbolic attributes. Each entry represents a person who takes a credit from a bank where the person is classified as having good or bad (`class`) credit risk according to a set of attributes.

The advantage with using such a small dataset is that we get to experiment faster, using fewer resources and that we get to address other problems we often don't face when working on larger datasets. Additionally, many companies - in particular startups - have limited datasets to work with in the first place. This would better simulate a situation like that. 



## Technologies


- Cloud: GCP
- Experiment Tracking: MLFlow
- Workflow Orchestration: Prefect
- Containerisation: Docker and Docker Compose
- Model Deployment: Flask, Docker, Dash, MLFlow
- Monitoring: Evidently, Grafana, Prometheus, MongoDB
- Best Practices: Unit and Integration Tests, Makefile
- (CI/CD: GitHub Actions)


## Implementation plan

- [x] Build notebook with initial model development
    - [x] Data Preparation
    - [x] Exploratory Data Analysis
    - [x] Model Pipeline for Tuning and Training
    - [x] Evaluate and Analyse Model Performance
- [x] Experiment Tracking
- [x] Workflow Orchestration
- [x] Web Service to Expose the Model Predictions
- [x] Model Monitoring
- [x] Tests (partially)
    - [x] Unit Tests
    - [x] Integration Tests
- [] CI/CD and [x] Makefile
- [x] GCP Cloud Deployment (partially)
- [x] Use Docker containers for each service


## Future Work

Each one of the implemented steps above can be further developed and improved on. However, some good practices have not been implemented at all and should be considered for more mature projects. These include:

- Complete deployment on the cloud. Currently everything is deployed in separate Docker containers. These can fairly easily be moved to the cloud as are.
- Host the generated Evidently reports for easier inspection. 
- Adding CI/CD
- Adding IaC



<br>

# Project Overview

Start all services by executing 

```bash
docker compose -f docker-compose.yml up --build
``` 

This will create a separate Docker container for each service. 

In order to add model training and Evidently report generation Flows to the queue, run the following commands: 

```bash
make model-train-flow
``` 

and 

```bash
make evidently-report-flow
``` 

At this stage, you can access the different services via the following urls:

- MLFlow: http://localhost:5051/
- Prefect UI: http://localhost:4200/
- Model UI, aka **Risk-O-Meter** (allows you to send data to the model and receive predictions. It's a simple simulation of how a bank clerk might use such a system): http://localhost:9696/
- Prometheus: http://localhost:9091/ (user/password: admin/admin)
- Grafana Dashboard: http://localhost:3000/


<br>


## Model Development

The model was developed in a Jupter Notebook ([Model-development.ipynb](model_development/Model-development.ipynb)), and includes data cleaning, data visualisation, basic feature selection, model tuning and model evaluation using 10-fold cross-validation. 

Three different algorithms are evaluated, Logistic Regression, Random Forest and LightGBM. LightGB was chosen due to its higher specificty and comparable other metrics. It was also significantly faster to train and resulted in a lighter model compared to the second best model. 

Below are 10-fold cross-validated results for each of the evaluated models. 

![](images/model-comparison.jpeg)


<br><br>

## MLFlow Model Registry

MLFlow is used to register model training runs. Artifacts, which includes models, plots and statistics, are stored in a Google Cloud Bucket, while the server is run in a local Docker container and accessible via the url specified above. 

After each model training, tuning and cross-validation run, the 5 best models are logged with their corresponding accuracy, AUC, recall and specificity on both the validation and test sets. To speed things up, currently also only 5 different parameter combinations are evaluated, but this can easily be adjusted.

![](images/mlflow-experiments-list.png)

The single best model is then registred in the Model Registry and it's up to a human evaluator the decide whether to move it into production stage or not. 

<br>

![](images/mlflow-model-registry-artifacts2.png)


<br>

## Prefect Workflow Orchestration

Prefect is used to orchestrate the runs, or DAGs. The model is currently set to automatically retrain every Sunday night to mitigate model degradation. Evidently reports are subsequently generated every 3 hours and stored in a Google Cloud Bucket where they can be downloaded and inspected.


![](images/prefect-ui.png)

The model training and report generation flows are deployed in different Docker containers.

<br>

## Model Service UI

In order to create a simple simulation of how a bank clerk might use a decision system like this, there's also a UI. The **Risk-O-Meter**, as it's called, allows the user to upload a .csv file with data about a list of clients and receive a prediction about their credit risk. 

- <span style="color:green">Green</span> credit risk means that the risk is lower for the bank and that they should make an offer to the client.
- <span style="color:red">Red</span> means that the risk is currently a little too high.

The model's confidence in its prediction for each client is also displayed for the clerk to better asses the prediction.


<br>

![](images/model-ui.png)


## Monitoring

- Grafana is used to monitor data drift and Prometheus stores these metrics.
- Evidently is used to calculate data drift and for more advanced monitoring analysis. The reports are stored in html format in a Google Cloud Bucket and locally in MongoDB.

![](images/grafana-dashboard.png)

<br>

[Evidently Report example](monitoring/reports/evidently_report.html)

![](images/evidently-report-large.png)


<br>


## Tests with Pytest

Some basic unit and integration tests are also available. These mainly check that the trained model returns predictions that are roughly accurate. However, this allows us to be fairly confident in that the entire data pre-processing pipeline and model training steps are correct, we are using correct package versions etc. 

Run the tests by initiating them in the VSCode UI, or by executing the following script:

```bash
python -m pytest model_orchestration_and_tracking/tests/*
```

