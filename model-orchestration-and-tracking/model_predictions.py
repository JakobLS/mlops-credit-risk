from sklearn.model_selection import cross_val_predict
from data_preparation import prepare_data



def make_cv_predictions(model_pipeline, X, Y):
    # Make cross-validated predictions 
    predictions = cross_val_predict(model_pipeline, X, Y, cv=10, n_jobs=-1, verbose=3)
    return predictions


def save_model_predictions(predictions):
    predictions.to_csv("../datasets/predictions/predictions.csv", index=False)


def make_predictions(model_pipeline, data):
    # Prepare the data as before
    X, Y = prepare_data(data)

    # Make and store model predictions
    data['predictions'] = model_pipeline.predict(X)
    save_model_predictions(data)

    return data




