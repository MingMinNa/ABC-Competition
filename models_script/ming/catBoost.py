import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier

try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess


def build_model(X_train, y_train, RANDOM_SEED = 42):
    numeric_features, _ = data_preprocess.get_number_of_datatype(X_data = X_train)

    # Too slow and the result is not better
    # model = CatBoostClassifier(eval_metric = "AUC", random_seed = RANDOM_SEED, verbose=0, iterations=100,
    #                             cat_features = list(range(len(numeric_features), X_train.shape[1])))

    model = CatBoostClassifier(eval_metric = "AUC", random_seed = RANDOM_SEED, verbose=0, iterations=100)

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
    print(f'[catBoost]最佳參數: {rand_search.best_params_}')
    print(f'[catBoost]最佳交叉驗證分數: {rand_search.best_score_:.4f}')
    return rand_search.best_estimator_

# 0.858505(fix)
def main(RANDOM_SEED = 42):
    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests)
    dataset_names = dataset_names[:1]
    y_predicts = []
    for i in tqdm(range(len(dataset_names))):
        model = build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        y_predict_proba = model.predict_proba(X_tests[i])[:, 1]
        df = pd.DataFrame(y_predict_proba, columns=['y_predict_proba'])
        y_predicts.append(df)

        del model

    # Save predictions to CSV files
    file_handler.save_predict(y_predicts, dataset_names)

if __name__ == '__main__':
    main(RANDOM_SEED = 42)