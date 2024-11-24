from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from enum import Enum

class Numeric_Handler_Type(Enum):
    No_preprocess = 0
    Standard_deviation = 1
    Normalization = 2
class Categoric_Handler_Type(Enum):
    No_preprocess = 0
    One_Hot_encoder = 1

def get_number_of_datatype(X_data):
    # Split the features in X_data into numeric_data and categorical_data.
    numeric, categoric = [], []
    for feature in X_data:
        val = float(X_data.loc[0, feature])
        if isinstance(val, float) and val.is_integer():
                categoric.append(feature)
        else:   numeric.append(feature)
    return numeric, categoric

def preprocess_numeric_data(numeric_data, numeric_handler):

    # standard deviation
    if numeric_handler == Numeric_Handler_Type.Standard_deviation:
        imputer = SimpleImputer(strategy = "mean")
        imputed_numeric_data = imputer.fit_transform(numeric_data)
        
        scaler = StandardScaler()
        scaled_numeric_data = scaler.fit_transform(imputed_numeric_data)
        return scaled_numeric_data
    
    elif numeric_handler == Numeric_Handler_Type.Normalization:
        # normalization
        MinMaxModel = MinMaxScaler().fit(numeric_data)
        normalized_numeric_data = MinMaxModel.transform(numeric_data)
        return normalized_numeric_data
    
    # No preprocess
    return numeric_data

def preprocess_categoric_data(X_data, categoric_features, categoric_data, categoric_handler):

    if len(categoric_features) == 0:    return X_data
    
    if categoric_handler == Categoric_Handler_Type.One_Hot_encoder:
        # One-Hot encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(categoric_data)

        new_columns = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categoric_data.columns))
        X_data = X_data.drop(columns = categoric_features).reset_index(drop = True)
        X_data = pd.concat([X_data, new_columns], axis = 1).reset_index(drop = True)
    elif isinstance(categoric_handler, Numeric_Handler_Type): 
        # handle categoric data as numeric data
        X_data[categoric_features] = X_data[categoric_features].astype(np.float32)
        X_data.loc[:, categoric_features] = preprocess_numeric_data(X_data[categoric_features], categoric_handler)

    # No preprocess
    return X_data

def align_columns(X_train, X_test):

    train_cols = X_train.columns
    test_cols = X_test.columns
    all_cols = test_cols.union(train_cols)

    test_missing_cols = train_cols.difference(test_cols)
    train_missing_cols = test_cols.difference(train_cols)

    for col in train_missing_cols:
        X_train[col] = 0
    for col in test_missing_cols:
        X_test[col] = 0  
    X_train = X_train[all_cols].copy()
    X_test = X_test[all_cols].copy()
    
    return X_train, X_test

def preprocess_features(X_data, numeric_features, categoric_features, numeric_handler, categoric_handler):
    X_data.loc[:, numeric_features] = preprocess_numeric_data(X_data[numeric_features], numeric_handler)
    X_data = preprocess_categoric_data(X_data, categoric_features, X_data.loc[:, categoric_features], categoric_handler)
    return X_data

def preprocess_data(dataset_names, X_trains, y_trains, X_tests, numeric_handler = Numeric_Handler_Type.No_preprocess, categoric_handler = Categoric_Handler_Type.No_preprocess):
    
    copy_X_trains, copy_X_tests, copy_Y_trains = [], [], []
    # preprocess
    for i, _ in enumerate(dataset_names):
        X_train, X_test, Y_train = X_trains[i].copy(deep = True), X_tests[i].copy(deep = True), y_trains[i].copy(deep = True)

        numeric_features, categoric_features = get_number_of_datatype(X_train)

        X_train = preprocess_features(X_train, numeric_features, categoric_features, numeric_handler, categoric_handler)
        X_test = preprocess_features(X_test, numeric_features, categoric_features, numeric_handler, categoric_handler)

        if categoric_handler == Categoric_Handler_Type.One_Hot_encoder:
            X_train, X_test = align_columns(X_train, X_test)

        copy_X_trains.append(X_train)
        copy_X_tests.append(X_test)
        copy_Y_trains.append(Y_train)

    return copy_X_trains, copy_Y_trains, copy_X_tests

if __name__ == "__main__":
    pass