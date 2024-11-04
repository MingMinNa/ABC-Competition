from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
import pandas as pd

try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess

# 0.814357
def main(TEST_MODE = True):
    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # preprocess
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
                X_trains[i], y_trains[i], test_size=0.2, random_state=42)
        else:
            tmp_X_train, tmp_y_train = X_trains[i], y_trains[i]

        # Define XGBoost model with specified parameters
        xgb_model = XGBClassifier(random_state = 42, eval_metric = "logloss", tree_method = "hist", early_stopping_rounds = 10)

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
        cross_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Perform RandomizedSearchCV for hyperparameter tuning
        randomized_search_cv = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=50,
            scoring="roc_auc",
            cv=cross_val,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        # Fit the model with training data
        randomized_search_cv.fit(tmp_X_train, tmp_y_train, eval_set=[(tmp_X_test, tmp_y_test)], verbose=False)

        # Evaluate the best model on the test data
        best_xgb_model = randomized_search_cv.best_estimator_
        if TEST_MODE:
            y_test_pred_proba = best_xgb_model.predict_proba(tmp_X_test)[:, 1]
            test_auc_score = roc_auc_score(tmp_y_test, y_test_pred_proba)
            aucs.append(test_auc_score)

        models.append(best_xgb_model)

    # Save AUC results if in test mode
    if TEST_MODE:
        file_handler.save_auc_result(aucs, "XGBoost")

    # Make predictions on test data
    y_predicts = []
    for i in range(len(dataset_names)):
        y_predict_proba = models[i].predict_proba(X_tests[i])[:, 1]
        df = pd.DataFrame(y_predict_proba, columns=['y_predict_proba'])
        y_predicts.append(df)

    # Save predictions to CSV files
    file_handler.save_predict(y_predicts, dataset_names)

if __name__ == "__main__":
    main(TEST_MODE=True)
