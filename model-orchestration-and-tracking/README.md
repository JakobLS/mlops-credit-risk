# Model Tracking


Start the MLflow tracking server and UI in this folder using the following command:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts
```

Alternatively, if you want to store metrics, artifacts and models individually in files rather than in a database:

```bash
mlflow ui --backend-store-uri mlruns --default-artifact-root artifacts
```


