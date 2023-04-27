
from sklearn.model_selection import cross_val_predict
from data_preparation import prepare_data



def make_cv_predictions(model, X, Y):
    # Make cross-validated predictions 
    predictions = cross_val_predict(model, X, Y, cv=10, n_jobs=-1, verbose=3)
    return predictions


def save_model_predictions(predictions):
    predictions.to_csv("../datasets/predictions/predictions.csv", index=False)


def make_predictions(model, data):
    # Specify the final columns based on previous work
    final_columns = [
        'age', 
        'checking_status', 
        'credit_amount', 
        'credit_history', 
        'duration',
        'installment_commitment', 
        'personal_status', 
        'property_magnitude',
        'purpose', 
        'savings_status',
        'class',
    ]
    data = data[final_columns]
    data = prepare_data(data)

    # Select features and target variable
    X = data[data.columns.difference(['class'])]
    Y = data['class']

    # Make and store model predictions
    data['predictions'] = model.predict(X)
    save_model_predictions(data)

    return data




