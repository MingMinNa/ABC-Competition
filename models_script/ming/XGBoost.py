from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
import pandas as pd

try:    
    from .utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
except: 
    from utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type


def build_model(X_train, y_train, RANDOM_SEED = 42):
    
    # Define XGBoost model with specified parameters
    xgb_model = XGBClassifier(random_state = RANDOM_SEED, eval_metric = "logloss", tree_method = "hist")

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
        verbose = 0,
        random_state = RANDOM_SEED
    )
    randomized_search_cv.fit(X_train, y_train)

    best_params = randomized_search_cv.best_params_

    best_xgb_model = XGBClassifier(
        random_state = RANDOM_SEED,
        eval_metric = "logloss",
        tree_method = "hist",
        **best_params  # insert the best hyperparameters
    )

    # train model with all train data
    best_xgb_model.fit(X_train, y_train)

    return best_xgb_model

# numeric(No_preprocess), categoric(No_preprocess):         0.834567 [Competition_data(xgboost).zip]
# numeric(std), categoric(one-hot):                         0.810903 [Competition_data(xgboost_2).zip]
def main(RANDOM_SEED = 42):
    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests,
                                                                  numeric_handler = Numeric_Handler_Type.No_preprocess,
                                                                  categoric_handler = Categoric_Handler_Type.No_preprocess)

    y_predicts = []
    for i in tqdm(range(len(dataset_names))):
        model = build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        y_predict_proba = model.predict_proba(X_tests[i])[:, 1]
        df = pd.DataFrame(y_predict_proba, columns=['y_predict_proba'])
        y_predicts.append(df)

        del model

    # Save predictions to CSV files
    file_handler.save_predict(y_predicts, dataset_names)

if __name__ == "__main__":
    main(RANDOM_SEED = 42)
