import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import  GridSearchCV
import lightgbm as lgb
from scipy.stats import randint, uniform

try:    
    from .utils import file_handler, data_preprocess
    from .utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
except: 
    from utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type


def build_model(X_train, y_train, RANDOM_SEED = 42):

    ravel_y_train = y_train.copy().to_numpy().ravel()

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 30, 50],
        'lambda_l1': [0, 1, 2],
        'lambda_l2': [0, 1, 2],
    }
    
    lgb_clf = lgb.LGBMClassifier(
        objective = 'binary',
        metric = 'auc',
        boosting_type = 'gbdt',
        random_state = RANDOM_SEED,
        verbose = -1
    )

    grid_search = GridSearchCV(
        estimator = lgb_clf,
        param_grid = param_grid,
        scoring = make_scorer(roc_auc_score),  
        cv = 5,      
        n_jobs = -1  
    )

    grid_search.fit(X_train, ravel_y_train)

    best_params = grid_search.best_params_
    print("[LightGBM]Best Parameters from CV:", best_params)

    final_model = lgb.LGBMClassifier(
        objective = 'binary',
        metric = 'auc',
        boosting_type = 'gbdt',
        random_state = RANDOM_SEED,
        verbose = -1,
        **best_params 
    )

    final_model.fit(X_train, ravel_y_train)

    return final_model

# numeric(No_preprocess), categoric(No_preprocess):         0.828752 [Competition_data(lightGBM).zip]
# numeric(std), categoric(one-hot):                         0.831177 [Competition_data(lightGBM_2).zip]
def main(RANDOM_SEED = 42):

    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests,
                                                                  numeric_handler = Numeric_Handler_Type.Standardization,
                                                                  categoric_handler = Categoric_Handler_Type.One_Hot_encoder)

    y_predicts = []
    for i in tqdm(range(len(dataset_names))):
        model = build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        y_pred_proba = model.predict_proba(X_tests[i], num_iteration = model._best_iteration)[:, 1]
        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    file_handler.save_predict(y_predicts, dataset_names)
    return 

if __name__ == "__main__":
    main(RANDOM_SEED = 42)
