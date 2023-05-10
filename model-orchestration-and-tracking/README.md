# Model Tracking

Before we're able to log anything with MLflow, we need to start the tracking server. Run one of the following commands depending on your preferences. 

1. Start the MLflow tracking server and UI in this folder using the following command. Only allows tracking of metrics. 

    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts
    ```

2. Alternatively, if you want to store everything, including metrics, artifacts and models in individual files, run the following:

    ```bash
    mlflow ui --backend-store-uri mlruns --default-artifact-root artifacts
    ```


