# Model Deployment

The model has been deployed as a web service using Flask and Dash. The resulting dashboard allows the user to interact with the model by submitting CSV or Excel files for prediction. This is a rather simple simulation of how a bank clerk might use a system like this. 


## Local Deployment

In order to start the dashboard, run the following command in the Terminal from the `../model_orchestration_and_tracking` folder. 

```bash
python ../model_deployment/app.py
```

This starts a server on your local machine on port 8090. Head over to `http://127.0.0.1:8090/` using a browser and start interacting with the model.


## Docker Deployment


