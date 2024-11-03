import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import lightgbm as lgb
from scipy.stats import randint, uniform

try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess


def main(TEST_MODE = True):
    RANDOM_SEED = 42
    # 讀取資料集
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # 預處理資料
    for i, _ in enumerate(dataset_names):
        X_train, X_test, Y_train = X_trains[i], X_tests[i], y_trains[i]

        numeric_features, categoric_features = data_preprocess.get_number_of_datatype(X_train)

        X_train = data_preprocess.preprocess_features(X_train, numeric_features, categoric_features)
        X_test = data_preprocess.preprocess_features(X_test, numeric_features, categoric_features)
        X_trains[i], X_tests[i] = data_preprocess.align_columns(X_train, X_test)
        y_trains[i] = Y_train

    models = []
    aucs = []
    for i in tqdm(range(len(dataset_names))):
        if TEST_MODE:
            tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
                X_trains[i], y_trains[i], test_size = 0.2, random_state = RANDOM_SEED)
        else:
            tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = X_trains[i], X_trains[i], y_trains[i], y_trains[i]

        # 定義超參數分佈
        params = {
            'objective': 'binary',  # 二元分類
            'metric': 'auc',        # 使用 AUC 作為評估指標
            'boosting_type': 'gbdt', # 傳統的梯度提升決策樹
            'learning_rate': 0.01,
            'num_leaves': 31,
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
        
        # 將數據轉換為 LightGBM 的數據格式
        train_data = lgb.Dataset(tmp_X_train, label = tmp_y_train)
        test_data = lgb.Dataset(tmp_X_test, label = tmp_y_test, reference = train_data)

        # 訓練模型
        model = lgb.train(
            params,
            train_data,
            valid_sets = [train_data, test_data],
            valid_names = ['train', 'valid'],
            num_boost_round = 100,
        )
        y_pred_proba = model.predict(tmp_X_test, num_iteration=model.best_iteration)
        auc = roc_auc_score(tmp_y_test, y_pred_proba)
        aucs.append(auc)
        models.append(model)

    y_predicts = []
    for i in range(len(dataset_names)):
        y_pred_proba = model.predict(tmp_X_test, num_iteration=model.best_iteration)
        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    file_handler.save_auc_result(aucs, "LightGBM")
    file_handler.save_predict(y_predicts, dataset_names)
    return 

if __name__ == "__main__":
    main(TEST_MODE = False)
