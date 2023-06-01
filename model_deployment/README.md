# Model Deployment

The model has been deployed as a web service using Flask and Dash. The resulting dashboard allows the user to interact with the model by submitting CSV or Excel files for prediction. This is a rather simple simulation of how a bank clerk might use a system like this. 

Also, note that we have currently hard-coded the model version to be used in the deployment. To streamline this process, we should aim for storing the model and artifacts in the cloud (such as an S3 bucket or Cloud Storage) which we then can retrieve directly. This would allow us to automatically train and then deploy the best model on a continuous basis. 


## Local Deployment

### Using a Development Server

In order to start the dashboard, run the following command in the Terminal from the `../model_orchestration_and_tracking` folder. 

```bash
python ../model_deployment/app.py
```

This starts a server on your local machine on port 8090. Head over to `http://127.0.0.1:8090/` using a browser and start interacting with the model.


### Using a Production Server

In the same `../model_orchestration_and_tracking` folder as above, run the following command:

```bash
export PYTHONPATH=${PWD%/*}
```

This will allow the production server to find the files needed. 

If not already installed, install `gunicorn`, which we will use as production server, using `pip install gunicorn`. Then, run the following command in this same folder to start the app:

```bash
gunicorn --bind=localhost:8090 model_deployment.app:server
```

The app is then accessible via the same url as before. 


## Docker Deployment

From the main project folder, where the `Dockerfile` is located, run:

```bash
docker build -t credit-risk .
```
Followed by

```bash
docker run -it --rm -p 8090:8090 credit-risk:latest
```

This will start up the server. Navigate to `http://0.0.0.0:8090/` to see and test out the live app! 


<br><br>

## Simple Initial Test 

In order to test the model, run the following in this folder:

```bash
python test_main.py
```

Accuracy, Precision, Recall and some other metrics should be printed to the terminal. They should be aligned with those from `test_app.py`.


