
# MLOps Project - Credit Risk Prediction

The purpose of this project is to build out a simple classification model to predict the credit risk of bank customers. The resulting model will then be deployed to production using MLOps best practices.

The dataset can be downloaded from Kaggle via [this](https://www.kaggle.com/btolar1/weka-german-credit) link.

The dataset contains 1000 entries with 20 categorial/symbolic attributes. Each entry represents a person who takes a credit from a bank where the person is classified as having good or bad (`class`) credit risk according to a set of attributes.

The advantage with using such a small dataset is that we get to experiment faster, using fewer resources and that we get to address other problems we often don't face when working on larger datasets. Additionally, many companies - in particular startups - have limited datasets to work with in the first place. This would better simulate a situation like that. 



## Technologies


- Cloud: GCP
- Experiment Tracking: MLFlow
- Workflow Orchestration: Prefect
- Monitoring: Evidently, Grafana, Prometheus
- CI/CD: GitHub Actions


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
- [] Tests
    - [] Unit Tests
    - [] Integration Tests
- [] CI/CD and Makefile
- [] GCP Cloud Deployment
- [] Use Docker containers for each service


