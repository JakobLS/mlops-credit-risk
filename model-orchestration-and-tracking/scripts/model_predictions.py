from sklearn.model_selection import cross_val_predict
from data_preparation import prepare_data



def make_cv_predictions(model_pipeline, X, Y):
    # Make cross-validated predictions 
    predictions = cross_val_predict(model_pipeline, X, Y, cv=10, n_jobs=-1, verbose=3)
    return predictions


def save_model_predictions(predictions, prediction_path):
    # Save mdoel predictions
    predictions.to_csv(prediction_path, index=False)


def make_predictions(model_pipeline, data, prediction_path):
    # Prepare the data as before
    X, Y = prepare_data(data)

    # Make and store model predictions
    data['predictions'] = model_pipeline.predict(X)
    data['prediction_probs'] = model_pipeline.predict_proba(X)[:, 1]
    save_model_predictions(data, prediction_path)

    return data




