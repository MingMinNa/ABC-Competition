import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint


def preprocess_data(X_train, X_test):
    # Separate numeric and categorical features
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("passthrough", "passthrough"),
        ]
    )

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit and transform the training data, transform the test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor


def main():
    base_path = "./Competition_data"

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

    ensemble_models = []
    auc_results = []

    for i in range(len(dataset_names)):
        print(f"Processing data set: {dataset_names[i]}")

        X_train = X_trains[i]
        y_train = y_trains[i]
        X_test = X_tests[i]

        # Split the training data into training and validation sets
        tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Preprocess the data
        tmp_X_train_processed, tmp_X_val_processed, preprocessor = preprocess_data(tmp_X_train, tmp_X_val)
        X_train_processed, X_test_processed, _ = preprocess_data(X_train, X_test)

        # 定義各個基模型
        xgb = XGBClassifier(eval_metric="auc", random_state=42)

        lgbm = LGBMClassifier(objective="binary", random_state=42)

        catboost = CatBoostClassifier(eval_metric="AUC", silent=True, random_state=42)

        # 定義各個模型的超參數分佈
        param_distributions_xgb = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 5),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(1, 2),
        }

        param_distributions_catboost = {
            "iterations": randint(100, 500),
            "depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "l2_leaf_reg": uniform(1, 10),
            "border_count": randint(32, 255),
        }

        rand_search_xgb = RandomizedSearchCV(
            xgb,
            param_distributions_xgb,
            n_iter=50,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        rand_search_catboost = RandomizedSearchCV(
            catboost,
            param_distributions_catboost,
            n_iter=50,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        print("Tuning XGBoost...")
        rand_search_xgb.fit(tmp_X_train_processed, tmp_y_train.values.ravel())
        best_xgb = rand_search_xgb.best_estimator_

        print("Tuning CatBoost...")
        rand_search_catboost.fit(tmp_X_train_processed, tmp_y_train.values.ravel())
        best_catboost = rand_search_catboost.best_estimator_

        ensemble = VotingClassifier(
            estimators=[
                ("xgb", best_xgb),
                ("catboost", best_catboost),
            ],
            voting="soft",
            weights=[1, 1],
            n_jobs=-1,
        )

        # 訓練集成模型
        ensemble.fit(tmp_X_train_processed, tmp_y_train.values.ravel())

        # 評估集成模型
        y_val_pred_proba = ensemble.predict_proba(tmp_X_val_processed)[:, 1]
        val_auc = roc_auc_score(tmp_y_val, y_val_pred_proba)
        auc_results.append({"Dataset": dataset_names[i], "Validation AUC": val_auc})
        print(f"{dataset_names[i]} Validation AUC: {val_auc:.4f}")

        # 對測試集進行預測
        y_pred_proba = ensemble.predict_proba(X_test_processed)[:, 1]

        # 保存預測結果
        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        df.to_csv(f"{base_path}/{dataset_names[i]}/y_predict.csv", index=False, header=True)
        print(f"{dataset_names[i]} saved.\n")

    # 顯示 AUC 結果
    auc_df = pd.DataFrame(auc_results)
    print("AUC result: ")
    print(auc_df)

    average_auc = auc_df["Validation AUC"].mean()
    print(f"\nAverage AUC: {average_auc:.4f}\n")


if __name__ == "__main__":
    main()
