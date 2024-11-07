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


def build_model(X_train, y_train, RANDOM_SEED = 42):

    # test version
    # tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split( X_train, y_train, test_size = 0.2, random_state = RANDOM_SEED)
    
    # final version
    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = X_train, X_train, y_train, y_train

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,    # 降低學習率
        'num_leaves': 30,         # 降低 num_leaves 以減少模型複雜度
        'lambda_l1': 1,           # 增加 L1 正則化
        'lambda_l2': 1,           # 增加 L2 正則化
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

    return model

# 0.703477(fix)
def main(RANDOM_SEED = 42):

    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests)

    y_predicts = []
    for i in tqdm(range(len(dataset_names))):
        model = build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        y_pred_proba = model.predict(X_tests[i], num_iteration = model.best_iteration)
        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    file_handler.save_predict(y_predicts, dataset_names)
    return 

if __name__ == "__main__":
    main(RANDOM_SEED = 42)
