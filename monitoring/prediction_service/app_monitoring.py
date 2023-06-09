import base64
import io
import os
import pandas as pd
import numpy as np
import requests

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import mlflow
from pymongo import MongoClient


MODEL_LOCATION = os.getenv('MODEL_LOCATION', 'model/')
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE_ADDRESS', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

# Define MongoDB client and collection
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("credit_risk_service")
collection = db.get_collection("credit_risk_data")

credit_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = credit_app.server
credit_app.title = "Credit Risk"

credit_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H2("Risk-O-Meter"), 
            width={'size': 12, 
                   'offset': 0, 
                   'order': 0}
        ), 
    style = {'textAlign': 'center', 
             'paddingBottom': '1%'}
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Client Files', 
                   style={'font-weight': '800'},
            )
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'width': '100%',
                                             'height': '80%'}),
])


def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['class'])]
    Y = data['class']
    
    return X, Y


def save_to_db(records):
    if len(records.index) > 1:
        collection.insert_many(records.to_dict('records'))
    else:
        collection.insert_one(records.to_dict('records'))


def send_to_evidently_service(records):
    # Evidently expects a list of records. Prepare the data accordingly
    if len(records) <= 1:
        records = [records.to_dict('records')]
    else:
        records = records.to_dict('records')
    # Post to Evidently, to the location "/iterate/<dataset>" 
    # (check line 216 in evidently_service.py)
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/credit_risk", json=records)


def make_predictions_with_model_registry_model(data):
    """ Function for retreiving a model from the Model Registry 
        and make predictions with it. 
    """

    # Prepare the data
    X, Y = prepare_data(data)

    # Load the model and make predictions
    try:
        # Load model
        model = mlflow.sklearn.load_model(model_uri=MODEL_LOCATION)
        data['predicted_risk'] = model.predict(X)
        data['prediction_probs'] = np.amax(model.predict_proba(X), axis=1).round(4)

        # Move the prediction column first
        preds = data.pop("predicted_risk")
        data.insert(0, "predicted_risk", preds)

        # Store the predictions and send them to Evidently for monitoring
        save_to_db(data)
        send_to_evidently_service(data)

        return data
    
    except Exception as e:
        print(e)

        return data


def parse_content(content, filename):
    _, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            "There was an error processing this file. Make sure you're either uploading a CSV or an Excel file."
        ])
    
    # Make predictions with the model
    predictions = make_predictions_with_model_registry_model(
        data=df,
    )

    return html.Div([
        html.H5(filename, style={'margin': '2vh 0'}),

        dash_table.DataTable(
            data=predictions.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in predictions.columns],
            filter_action='native',
            sort_action="native",
            sort_mode="multi",
            page_current=0,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'whiteSpace': 'normal'},
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{predicted_risk} eq "good"',
                        'column_id': 'predicted_risk'
                    },
                    'backgroundColor': '#F0FFF0',
                },
                {
                    'if': {
                        'filter_query': '{predicted_risk} eq "bad"',
                        'column_id': 'predicted_risk'
                    },
                    'backgroundColor': '#FFE4E1',
                },
                ],
        ),
    ])


@credit_app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(content, names):
    if content is not None:
        children = parse_content(content, names)
        return children




if __name__ == "__main__":
    credit_app.run_server(debug=True, host='0.0.0.0', port=9696)
