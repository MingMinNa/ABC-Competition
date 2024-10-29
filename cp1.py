import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


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

    models = []
    auc_results = []

    for i in range(len(dataset_names)):
        print(f"Process data set: {dataset_names[i]}")

        X_train = X_trains[i]
        y_train = y_trains[i]
        X_test = X_tests[i]

        tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        X_train_processed, X_val_processed, X_test_processed = preprocess_data(tmp_X_train, tmp_X_val, X_test)
        tmp_y_train = tmp_y_train.values.ravel()
        tmp_y_val = tmp_y_val.values.ravel()

        xgb = XGBClassifier(random_state=42, eval_metric="logloss")

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1],
            "colsample_bytree": [0.8, 1],
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=xgb, param_grid=param_grid, scoring="roc_auc", cv=skf, n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train_processed, tmp_y_train)

        print(f"Best hyperparameter: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_}")

        best_model = grid_search.best_estimator_
        y_val_pred_proba = best_model.predict_proba(X_val_processed)[:, 1]
        val_auc = roc_auc_score(tmp_y_val, y_val_pred_proba)
        print(f"{dataset_names[i]} Validation AUC: {val_auc:.4f}")

        models.append(best_model)
        auc_results.append({"Dataset": dataset_names[i], "Validation AUC": val_auc})

        full_X_train_processed, _, _ = preprocess_data(X_train, X_train, X_test)
        full_y_train = y_train.values.ravel()
        best_model.fit(full_X_train_processed, full_y_train)

        y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

        df = pd.DataFrame(y_pred_proba, columns=["target"])
        df.to_csv(f"{base_path}/{dataset_names[i]}/y_predict.csv", index=False, header=True)

    auc_df = pd.DataFrame(auc_results)
    print("Every AUC:")
    print(auc_df)

    average_auc = auc_df["Validation AUC"].mean()
    print(f"\nAverage AUC: {average_auc:.4f}")


if __name__ == "__main__":
    main()
