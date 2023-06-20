import pandas as pd
from sklearn.metrics import classification_report


PREDICTIONS_PATH = "../datasets/predictions/registry_predictions.csv"

def load_predictions():
    new_data = pd.read_csv(PREDICTIONS_PATH)
    return new_data


def calculate_KPIs():
    # Get the predictions
    predictions = load_predictions()

    # Calculate some KPIs
    cl_report = classification_report(predictions['class'], 
                                      predictions['predictions'], 
                                      digits=3, 
                                      output_dict=True,
    )

    print(cl_report)


if __name__ == "__main__":
    calculate_KPIs()



