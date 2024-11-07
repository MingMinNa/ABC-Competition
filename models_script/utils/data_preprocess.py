from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def get_number_of_datatype(X_data):
    # Split the features in X_data into numeric_data and categorical_data.
    numeric, categoric = [], []
    for feature in X_data:
        val = float(X_data.loc[0, feature])
        if isinstance(val, float) and val.is_integer():
                categoric.append(feature)
        else:   numeric.append(feature)
    return numeric, categoric

def preprocess_numeric_data(numeric_data):
    imputer = SimpleImputer(strategy = "mean")
    imputed_numeric_data = imputer.fit_transform(numeric_data)
    
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(imputed_numeric_data)
    return scaled_numeric_data

def preprocess_categoric_data(categoric_data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encoder.fit_transform(categoric_data)
    return pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categoric_data.columns))

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

def preprocess_features(X_data, numeric_features, categoric_features):
    X_data.loc[:, numeric_features] = preprocess_numeric_data(X_data[numeric_features])
    new_columns = preprocess_categoric_data(X_data.loc[:, categoric_features])
    X_data = X_data.drop(columns = categoric_features).reset_index(drop = True)
    X_data = pd.concat([X_data, new_columns], axis = 1).reset_index(drop = True)
    return X_data

def preprocess_data(dataset_names, X_trains, y_trains, X_tests):
    
    # preprocess
    for i, _ in enumerate(dataset_names):
        X_train, X_test, Y_train = X_trains[i], X_tests[i], y_trains[i]

        numeric_features, categoric_features = get_number_of_datatype(X_train)

        X_train = preprocess_features(X_train, numeric_features, categoric_features)
        X_test = preprocess_features(X_test, numeric_features, categoric_features)
        X_trains[i], X_tests[i] = align_columns(X_train, X_test)
        y_trains[i] = Y_train

    return X_trains, y_trains, X_tests

if __name__ == "__main__":
    pass