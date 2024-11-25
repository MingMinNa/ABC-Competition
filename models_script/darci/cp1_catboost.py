import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint


def preprocess_data(X_train, X_test):
    # Separate numeric and categorical features
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

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

    return X_train_processed, X_test_processed, categorical_features.tolist()

# 0.836335
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
        print(f"Processing data set: {dataset_names[i]}")

        X_train = X_trains[i]
        y_train = y_trains[i]
        X_test = X_tests[i]

        # Split the training data into training and validation sets
        tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Preprocess the data
        tmp_X_train_processed, tmp_X_val_processed, categorical_features = preprocess_data(tmp_X_train, tmp_X_val)
        X_train_processed, X_test_processed, _ = preprocess_data(X_train, X_test)

        # Define the model
        catboost = CatBoostClassifier(eval_metric="AUC", random_seed=42, verbose=0, iterations=1000)

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
            catboost,
            param_distributions,
            n_iter=100,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3),
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        rand_search.fit(
            tmp_X_train_processed,
            tmp_y_train.values.ravel(),
            eval_set=[(tmp_X_val_processed, tmp_y_val)],
            early_stopping_rounds=50,
            cat_features=[X_train.columns.get_loc(col) for col in categorical_features],
        )

        # Best model
        best_model = rand_search.best_estimator_
        models.append(best_model)

        # Predict probabilities on the validation set
        y_val_pred_proba = best_model.predict_proba(tmp_X_val_processed)[:, 1]
        val_auc = roc_auc_score(tmp_y_val, y_val_pred_proba)
        auc_results.append({"Dataset": dataset_names[i], "Validation AUC": val_auc})

        # Predict probabilities on the test set
        y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

        # Save the predictions
        df = pd.DataFrame(y_pred_proba, columns=["y_predict_proba"])
        df.to_csv(os.path.join(base_path, dataset_names[i], "y_predict.csv"), index=False, header=True)
        print(f"{dataset_names[i]} saved.\n")

    # Display AUC results
    auc_df = pd.DataFrame(auc_results)
    print("AUC result: ")
    print(auc_df)

    average_auc = auc_df["Validation AUC"].mean()
    print(f"\nAverage AUC: {average_auc:.4f}")


if __name__ == "__main__":
    main()
