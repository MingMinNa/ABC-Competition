import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier

try:    
    from .utils import file_handler, data_preprocess
    from .utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
except: 
    from utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type


def build_model(X_train, y_train, RANDOM_SEED = 42):
    
    # numeric_features, _ = data_preprocess.get_number_of_datatype(X_data = X_train)
    
    base_model = CatBoostClassifier(
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose = 0,  
        # cat_features = list(range(len(numeric_features), X_train.shape[1])),
        iterations=100
    )

    param_distributions = {
        "depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "l2_leaf_reg": uniform(1, 10),
        "bagging_temperature": uniform(0, 1),
        "random_strength": uniform(0, 1),
    }

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    rand_search.fit(X_train, y_train, early_stopping_rounds=50)
    
    print(f'[CatBoost] 最佳參數: {rand_search.best_params_}')
    print(f'[CatBoost] 最佳交叉驗證分數: {rand_search.best_score_:.4f}')

    best_params = rand_search.best_params_
    final_model = CatBoostClassifier(
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=0,  
        iterations=100,
        **best_params  
    )

    final_model.fit(X_train, y_train)

    return final_model

# numeric(No_preprocess), categoric(No_preprocess):         0.866916 [Competition_data(catboost).zip]
# numeric(std), categoric(one-hot):                         0.858284 [Competition_data(catboost_2).zip]

# use "cat_features = list(range(len(numeric_features), X_train.shape[1]))" 
# numeric(No_preprocess), categoric(No_preprocess):         0.857915 [Competition_data(catboost_cat).zip]
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

if __name__ == '__main__':
    main(RANDOM_SEED = 42)