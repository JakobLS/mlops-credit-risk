import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from sklearn.metrics import classification_report


TEST_DATA_PATH = "../datasets/cleaned/test/credit_test.csv"
app = Flask('credit-risk-prediction')



def load_test_data():
    new_data = pd.read_csv(TEST_DATA_PATH)
    return new_data


def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['class'])]
    Y = data['class']
    
    return X, Y


def make_predictions_with_model_registry_model(model_name):
    # Load the data and prepare it
    data = load_test_data()
    X, Y = prepare_data(data)

    # Load the model and make predictions
    try:
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/Production")
        data['predictions'] = model.predict(X)
        data['prediction_probs'] = model.predict_proba(X)[:, 1]
        
    except:
        raise 

    return data


@app.route('/test', methods=['GET', 'POST'])
def predict_endpoint():
    predictions = make_predictions_with_model_registry_model(
        model_name="Credit Risk Prediction Model"
    )

    cl_report = classification_report(predictions['class'], 
                                      predictions['predictions'], 
                                      digits=3, 
                                      output_dict=True,
    )

    print(cl_report)
    return jsonify(cl_report)


if __name__ == "__main__":
    app.run(debug=True, port=9696)

