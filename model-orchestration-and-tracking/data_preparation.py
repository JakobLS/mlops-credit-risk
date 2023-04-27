
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2




def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Randomly shuffle the data to minimise the effect of randomness on our results
    data = data.sample(frac=1.0, random_state=55)

    # Select all categorical columns
    categorical = data.select_dtypes(include=['object']).columns

    # Define encoder. As we are dealing with categorical variables, make sure that the output are integers
    oe = OrdinalEncoder(dtype=np.int64)

    # Transform the categorical columns
    data[categorical] = oe.fit_transform(data[categorical])
    
    return data


def feature_selection(data):
    # Select features and target variable
    X = data[data.columns.difference(['class'])]
    Y = data['class']

    # Select the top 10 most important features using chi-squared
    selector = SelectKBest(chi2, k=10).fit(X, Y)
    X = pd.DataFrame(selector.transform(X), 
                    columns=selector.get_feature_names_out()) 
    
    return X, Y


