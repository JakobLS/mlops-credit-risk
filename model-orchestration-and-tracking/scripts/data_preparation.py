import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['class'])]
    Y = data['class']
    
    return X, Y


def feature_selector():
    # Select the top 10 most important features using chi-squared
    selector = SelectKBest(chi2, k=10)
    return selector


def create_data_preprocessor(all_feature_columns):
    # Select categorical and continuous columns
    numerical_columns = ['duration', 'credit_amount', 'age']
    categorical_columns = [c for c in all_feature_columns if c not in numerical_columns]

    # Define the steps in the categorical processor.
    categorical_processor = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(dtype=np.int64, 
                                            handle_unknown='use_encoded_value', 
                                            unknown_value=-1)),
    ])

    # # Define the steps in the numerical processor (mainly for Logistic Regression)
    # numerical_processor = Pipeline(steps=[
    #         ('numerical_selector', FeatureSelector(numerical)),
    #         # ('Standardscaler', StandardScaler()),
    # ])

    # Combine the categorical and numerical processors into a combined processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical_processor', categorical_processor, categorical_columns),
            # ('numerical_processor', numerical_processor, numerical_columns),
            ('passthrough', 'passthrough', numerical_columns),
            ],
        remainder='drop', verbose_feature_names_out=False)
    
    return preprocessor





