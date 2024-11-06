import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess

RANDOM_SEED = 42

# 0.725659
def main(TEST_MODE = True):
    RANDOM_SEED = 42

    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()
    dataset_datatype_cnt = []
    for i, _ in enumerate(dataset_names):
        numeric_features, categoric_features = data_preprocess.get_number_of_datatype(X_data=X_trains[i])
        dataset_datatype_cnt.append((numeric_features, categoric_features))

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    models = []
    best_aucs = []
    
    param_grid = {
        'classifier__n_estimators': [50, 70, 100],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    for i in tqdm(range(len(dataset_names))):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, dataset_datatype_cnt[i][0]),
                ("cat", categorical_transformer, dataset_datatype_cnt[i][1]),
            ]
        )

        model = RandomForestClassifier(random_state = RANDOM_SEED)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        grid_search = GridSearchCV(pipeline, param_grid, cv = 3, scoring = "roc_auc", n_jobs=-1)
        grid_search.fit(X_trains[i], y_trains[i].squeeze())
        
        best_model = grid_search.best_estimator_
        models.append(best_model)

    y_predicts = []
    for i in range(len(dataset_names)):
        y_predict_proba = models[i].predict_proba(X_tests[i])[:, 1]
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    if TEST_MODE: 
        file_handler.save_auc_result(best_aucs, "RF")
    file_handler.save_predict(y_predicts, dataset_names)
    return

if __name__ == "__main__":
    main(TEST_MODE = True)