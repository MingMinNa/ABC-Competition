import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint


def preprocess_data(X_train, X_val, X_test):
    numerical_feats = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_feats = X_train.select_dtypes(include=["object", "category"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    X_train_num = num_imputer.fit_transform(X_train[numerical_feats])
    X_val_num = num_imputer.transform(X_val[numerical_feats])
    X_test_num = num_imputer.transform(X_test[numerical_feats])

    if len(categorical_feats) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = cat_imputer.fit_transform(X_train[categorical_feats])
        X_val_cat = cat_imputer.transform(X_val[categorical_feats])
        X_test_cat = cat_imputer.transform(X_test[categorical_feats])

        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_train_cat_encoded = onehot_encoder.fit_transform(X_train_cat)
        X_val_cat_encoded = onehot_encoder.transform(X_val_cat)
        X_test_cat_encoded = onehot_encoder.transform(X_test_cat)
    else:
        X_train_cat_encoded = np.array([]).reshape(len(X_train_num), 0)
        X_val_cat_encoded = np.array([]).reshape(len(X_val_num), 0)
        X_test_cat_encoded = np.array([]).reshape(len(X_test_num), 0)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_val_num_scaled = scaler.transform(X_val_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_processed = np.hstack((X_train_num_scaled, X_train_cat_encoded))
    X_val_processed = np.hstack((X_val_num_scaled, X_val_cat_encoded))
    X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))

    return X_train_processed, X_val_processed, X_test_processed


def main():
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Competition_data")

    dataset_names = []
    X_trains = []
    y_trains = []
    X_tests = []

    for folder_name in sorted(os.listdir(base_path)):
        dataset_path = os.path.join(base_path, folder_name)
        if os.path.isdir(dataset_path):
            dataset_names.append(folder_name)
            X_trains.append(pd.read_csv(f"{dataset_path}/X_train.csv", header=0))
            y_trains.append(pd.read_csv(f"{dataset_path}/y_train.csv", header=0))
            X_tests.append(pd.read_csv(f"{dataset_path}/X_test.csv", header=0))

    models = []
    auc_results = []

    for i in range(len(dataset_names)):
        print(f"Process data set: {dataset_names[i]}")

        X_train = X_trains[i]
        y_train = y_trains[i]
        X_test = X_tests[i]

        # 使用 train_test_split 進行資料分割
        tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        X_train_processed, X_val_processed, X_test_processed = preprocess_data(tmp_X_train, tmp_X_val, X_test)
        tmp_y_train = tmp_y_train.values.ravel()
        tmp_y_val = tmp_y_val.values.ravel()

        xgb = XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", early_stopping_rounds=10)

        param_distributions = {
            "n_estimators": randint(300, 700),  # 樹的數量
            "max_depth": randint(4, 12),  # 樹的最大深度
            "learning_rate": uniform(0.01, 0.29),  # 學習率
            "subsample": uniform(0.6, 0.4),  # 用於訓練每棵樹的樣本比例，防止過擬合
            "colsample_bytree": uniform(0.6, 0.4),  # 每棵樹使用的特徵比例，防止過擬合
            "gamma": uniform(0, 0.5),  # 分裂節點所需的最小損失減少量，控制模型的保守性
            "min_child_weight": randint(1, 10),  # 子節點所需的最小權重和，防止過於複雜的模型
            "reg_alpha": uniform(0, 1),  # L1 正則化項的權重，減少模型複雜度
            "reg_lambda": uniform(0, 1),  # L2 正則化項的權重，防止過擬合
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        randomized_search = RandomizedSearchCV(
            estimator=xgb,  # 要調參的模型
            param_distributions=param_distributions,  # 超參數的分佈範圍
            n_iter=50,  # 隨機搜尋中參數組合的嘗試次數
            scoring="roc_auc",  # 評估指標
            cv=skf,  # 使用前面定義的分層 K 折交叉驗證
            n_jobs=-1,  # 使用所有可用的 CPU 核心進行並行運算
            verbose=1,  # 設置詳細程度，1表示顯示進度條
            random_state=42,  # 設置隨機種子以確保結果可重現
        )

        # 執行隨機搜尋，尋找最佳超參數
        randomized_search.fit(X_train_processed, tmp_y_train, eval_set=[(X_val_processed, tmp_y_val)], verbose=False)

        print(f"Best hyperparameter: {randomized_search.best_params_}")
        print(f"Best CV AUC: {randomized_search.best_score_}")

        best_model = randomized_search.best_estimator_
        y_val_pred_proba = best_model.predict_proba(X_val_processed)[:, 1]  # 使用最佳模型預測驗證集的概率
        val_auc = roc_auc_score(tmp_y_val, y_val_pred_proba)  # 計算驗證集的AUC分數
        print(f"{dataset_names[i]} Validation AUC: {val_auc:.4f}")

        models.append(best_model)
        auc_results.append({"Dataset": dataset_names[i], "Validation AUC": val_auc})

        full_X_train_processed, _, X_test_processed = preprocess_data(X_train, X_train, X_test)
        full_y_train = y_train.values.ravel()

        final_model = XGBClassifier(
            **randomized_search.best_params_, random_state=42, eval_metric="logloss", tree_method="hist"
        )

        # 初始化最終模型，使用最佳超參數
        final_model.fit(full_X_train_processed, full_y_train)

        y_pred_proba = final_model.predict_proba(X_test_processed)[:, 1]

        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        df.to_csv(os.path.join(base_path, dataset_names[i], "y_predict.csv"), index=False, header=True)
        print(f"{dataset_names[i]} saved.\n")

    auc_df = pd.DataFrame(auc_results)
    print("AUC result: ")
    print(auc_df)

    average_auc = auc_df["Validation AUC"].mean()
    print(f"\nAverage AUC: {average_auc:.4f}")


if __name__ == "__main__":
    main()
