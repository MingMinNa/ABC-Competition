import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier

try:    from .utils import file_handler, data_preprocess, const
except: from utils import file_handler, data_preprocess, const

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)           
        self.fc3 = nn.Linear(32, 1)             

    def forward(self, x):
        x = F.relu(self.fc1(x))                
        x = self.dropout1(x)                    
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))         
        return x

class MergeNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MergeNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)            
        self.fc2 = nn.Linear(64, 1)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))         
        return x


def XGBoost(X_train, y_train, TEST_MODE = True, RANDOM_SEED = 42):

    tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
            X_train, y_train, test_size = 0.2, random_state = RANDOM_SEED)
    # Define XGBoost model with specified parameters
    xgb_model = XGBClassifier(random_state = RANDOM_SEED, eval_metric = "logloss", tree_method = "hist", early_stopping_rounds = 10)

    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        "n_estimators": randint(300, 700),
        "max_depth": randint(4, 12),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.5),
        "min_child_weight": randint(1, 10),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1)
    }

    # Set up cross-validation
    cross_val = StratifiedKFold(n_splits = 3, shuffle = True, random_state = RANDOM_SEED)

    # Perform RandomizedSearchCV for hyperparameter tuning
    randomized_search_cv = RandomizedSearchCV(
        estimator = xgb_model,
        param_distributions = param_distributions,
        n_iter = 50,
        scoring = "roc_auc",
        cv = cross_val,
        n_jobs = -1,
        verbose = 1,
        random_state = RANDOM_SEED
    )

    # Fit the model with training data
    randomized_search_cv.fit(X_train, y_train, eval_set=[(tmp_X_test, tmp_y_test)], verbose=False)

    best_xgb_model = randomized_search_cv.best_estimator_
    return best_xgb_model

def LightGBM(X_train, y_train, TEST_MODE = True, RANDOM_SEED = 42):
    if TEST_MODE:
        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
            X_train, y_train, test_size = 0.2, random_state = RANDOM_SEED)
    else:
        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = X_train, X_train, y_train, y_trains[i]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,   # 降低學習率
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

def catBoost(X_train, y_train, TEST_MODE = True, RANDOM_SEED = 42):

    # 初始化 CatBoost 模型
    model = CatBoostClassifier(eval_metric="AUC", random_seed = RANDOM_SEED, verbose=0, iterations=100)

    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        "depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "l2_leaf_reg": uniform(1, 10),
        "bagging_temperature": uniform(0, 1),
        "random_strength": uniform(0, 1),
    }

    # Perform RandomizedSearchCV to find the best
    rand_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter = 100,
        scoring = "roc_auc",
        cv = StratifiedKFold(n_splits=3),
        verbose = 1,
        n_jobs = -1,
        random_state = RANDOM_SEED,
    )

    rand_search.fit(
        X_train,
        y_train,
        early_stopping_rounds = 50,
    )

    # 輸出最佳參數和最佳分數
    print(f'最佳參數: {rand_search.best_params_}')
    print(f'最佳交叉驗證分數: {rand_search.best_score_:.4f}')
    # print(rand_search.best_estimator_.predict_proba(X_train)[:, 1])
    return rand_search.best_estimator_

def simple_DL(X_train, y_train, TEST_MODE = True, RANDOM_SEED = 42):

    torch.manual_seed(RANDOM_SEED)
    
    X = X_train
    y = y_train
        
    # skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = RANDOM_SEED)

    # for train_index, test_index in skf.split(X, y):
    #     tmp_X_train, tmp_X_test = X.iloc[train_index], X.iloc[test_index]
    #     tmp_y_train, tmp_y_test = y.iloc[train_index], y.iloc[test_index]
    #     train_tensor = TensorDataset(torch.FloatTensor(tmp_X_train.values), torch.FloatTensor(tmp_y_train.values))
    #     train_loader = DataLoader(train_tensor, batch_size = 32, shuffle = True)
        
    #     input_size = X.shape[1]
    #     model = SimpleNN(input_size)
        
    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
    #     # Train the model
    #     model.train()
    #     for epoch in range(50):  # You can adjust this based on your dataset
    #         for inputs, labels in train_loader:
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = criterion(outputs.squeeze(), labels.squeeze())
    #             loss.backward()
    #             optimizer.step()
    
    input_size = X.shape[1]
    model = SimpleNN(input_size)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    train_tensor = TensorDataset(torch.FloatTensor(X.values), torch.FloatTensor(y.values))
    train_loader = DataLoader(train_tensor, batch_size = 32, shuffle = True)
    model.train()
    for epoch in range(50):  # You can adjust this based on your dataset
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
    return model

def merge_DL(dataset_names, X_trains, y_trains, X_tests, RANDOM_SEED = 42):
    
    torch.manual_seed(RANDOM_SEED)

    y_predicts = []
    for i in tqdm(range(len(dataset_names))):
        simpleDL_model = simple_DL(X_train = X_trains[i], y_train = y_trains[i], TEST_MODE = False, RANDOM_SEED = RANDOM_SEED)
        XGB_model = XGBoost(X_train = X_trains[i], y_train = y_trains[i], TEST_MODE = False, RANDOM_SEED = RANDOM_SEED)
        LightGBM_model = LightGBM(X_train = X_trains[i], y_train = y_trains[i], TEST_MODE = False, RANDOM_SEED = RANDOM_SEED)
        catBoost_model = catBoost(X_train = X_trains[i], y_train = y_trains[i], TEST_MODE = False, RANDOM_SEED = RANDOM_SEED)
        
        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(X_trains[i], y_trains[i], train_size = 0.8, random_state = RANDOM_SEED)
        simpleDL_model.eval()
        with torch.no_grad():
            simple_DL_y_prob = simpleDL_model(torch.FloatTensor(tmp_X_train.values)).numpy()
        
        XGB_y_prob = XGB_model.predict_proba(tmp_X_train)[:, 1]
        LightGBM_y_prob = LightGBM_model.predict(tmp_X_train, num_iteration = LightGBM_model.best_iteration)
        catBoost_y_prob = catBoost_model.predict_proba(tmp_X_train)[:, 1]

        train_tensor = TensorDataset(torch.FloatTensor(pd.DataFrame({
            'DL': simple_DL_y_prob[:, 0], 
            'XGB': XGB_y_prob, 
            'LightGBM': LightGBM_y_prob,
            'catBoost': catBoost_y_prob}).values), torch.FloatTensor(tmp_y_train.values))
        train_loader = DataLoader(train_tensor, batch_size = 32, shuffle = True)

        merge_model = MergeNN()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(merge_model.parameters(), lr = 0.001)
        for epoch in range(50):  # You can adjust this based on your dataset
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = merge_model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

        merge_model.eval()
        simpleDL_model.eval()
        with torch.no_grad():
            simple_DL_y_prob = simpleDL_model(torch.FloatTensor(tmp_X_test.values)).numpy()
            XGB_y_prob = XGB_model.predict_proba(tmp_X_test)[:, 1]
            LightGBM_y_prob = LightGBM_model.predict(tmp_X_test, num_iteration = LightGBM_model.best_iteration)
            catBoost_y_prob = catBoost_model.predict_proba(tmp_X_test)[:, 1]
            tmp_y_prob = merge_model(torch.FloatTensor(pd.DataFrame({
                                    'DL': simple_DL_y_prob[:, 0], 
                                    'XGB': XGB_y_prob,
                                    'LightGBM': LightGBM_y_prob,
                                    'catBoost': catBoost_y_prob}).values)).numpy()
            auc = roc_auc_score(tmp_y_test, tmp_y_prob)
            print(f"{i} {auc}")

            simple_DL_y_prob = simpleDL_model(torch.FloatTensor(X_tests[i].values)).numpy()
            XGB_y_prob = XGB_model.predict_proba(X_tests[i])[:, 1]
            LightGBM_y_prob = LightGBM_model.predict(X_tests[i], num_iteration = LightGBM_model.best_iteration)
            catBoost_y_prob = catBoost_model.predict_proba(X_tests[i])[:, 1]
            y_predict_proba = merge_model(torch.FloatTensor(pd.DataFrame({
                                         'DL': simple_DL_y_prob[:, 0], 
                                         'XGB': XGB_y_prob,
                                         'LightGBM': LightGBM_y_prob,
                                         'catBoost': catBoost_y_prob}).values)).numpy()
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

        del LightGBM_model, XGB_model, merge_model, simpleDL_model, catBoost_model

        # del simpleDL_model, XGB_model
    file_handler.save_predict(y_predicts, dataset_names)


# with preprocess of numeric and categoric features: 0.854
# without all preprocess: 0.872
# without preprocess of categoric features: 0.861
if __name__ == '__main__':
    
    RANDOM_SEED = 42
    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # preprocess

    # no preprocessing has the better result (?)
    # for i, _ in enumerate(dataset_names):
    #     X_train, X_test, Y_train = X_trains[i], X_tests[i], y_trains[i]
    #     numeric_features, categoric_features = data_preprocess.get_number_of_datatype(X_train)
    #     X_train = data_preprocess.preprocess_features(X_train, numeric_features, categoric_features)
    #     X_test = data_preprocess.preprocess_features(X_test, numeric_features, categoric_features)
    #     X_trains[i], X_tests[i] = data_preprocess.align_columns(X_train, X_test)
    #     y_trains[i] = Y_train

    merge_DL(dataset_names, X_trains, y_trains, X_tests, RANDOM_SEED)
    quit()


    
    