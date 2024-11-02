import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess


def main(TEST_MODE = True):
    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # preprocess
    dataset_datatype_cnt = []
    for i, _ in enumerate(dataset_names):
        numeric_features, categoric_features = data_preprocess.get_number_of_datatype(X_data = X_trains[i])
        dataset_datatype_cnt.append((numeric_features, categoric_features))
    
    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    models=[]
    aucs = []
    for i in range(len(dataset_names)):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, dataset_datatype_cnt[i][0]),
                ("cat", categorical_transformer, dataset_datatype_cnt[i][1]),
            ]
        )

        # Define the model
        model = RandomForestClassifier(random_state = 42)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        if TEST_MODE:
            tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(X_trains[i], y_trains[i], test_size=0.2, random_state=42)
            tmp_y_train = tmp_y_train.squeeze()
            tmp_y_test = tmp_y_test.squeeze()
            # Combine preprocessing steps
            
            pipeline.fit(tmp_X_train, tmp_y_train)
            tmp_y_prob = pipeline.predict_proba(tmp_X_test)[:, 1]
            auc = roc_auc_score(tmp_y_test, tmp_y_prob)
            aucs.append(auc)
        else:
            y_trains[i] = y_trains[i].squeeze()
            pipeline.fit(X_trains[i], y_trains[i])
        models.append(pipeline)
    y_predicts=[]
    for i in range(len(dataset_names)):
        y_predict_proba=models[i].predict_proba(X_tests[i])[:, 1]
        df = pd.DataFrame(y_predict_proba, columns=['y_predict_proba'])
        y_predicts.append(df)

    file_handler.save_auc_result(aucs, "RF")
    file_handler.save_predict(y_predicts, dataset_names)

if __name__ == "__main__":
    main(TEST_MODE = True)