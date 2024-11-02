import os
import pandas as pd

try:    from . import const
except: import const

def remove_all_predict():
    for folder_name in os.listdir(const.DATA_FOLDER):
        predict_file = os.path.join(const.DATA_FOLDER, folder_name, "y_predict.csv")
        if os.path.exists(predict_file) is False: continue
        os.remove(predict_file)
    return

def load_dataset():

    dataset_names = []
    X_trains = []
    y_trains = []
    X_tests = []
    for folder_name in os.listdir(const.DATA_FOLDER):
        x_train_path = os.path.join(const.DATA_FOLDER, folder_name, "X_train.csv")
        y_train_path = os.path.join(const.DATA_FOLDER, folder_name, "y_train.csv")
        x_test_path = os.path.join(const.DATA_FOLDER, folder_name, "X_test.csv")

        dataset_names.append(folder_name)
        X_trains.append(pd.read_csv(x_train_path, header = 0))
        y_trains.append(pd.read_csv(y_train_path, header = 0))
        X_tests.append(pd.read_csv(x_test_path, header = 0))
    return dataset_names, X_trains, y_trains, X_tests

def save_predict(y_predicts, dataset_names):
    for idx,dataset_name in enumerate(dataset_names):
        predict_path = os.path.join(const.DATA_FOLDER, dataset_name, "y_predict.csv")
        df = y_predicts[idx]
        df.to_csv(predict_path, index = False,header = True)
    return

def save_auc_result(aucs, file_name):
    file_path = os.path.join(const.RESULT_FOLDER, file_name + ".csv")
    result_file = pd.DataFrame(columns = ['dataset', 'auc'])
    for idx, auc in enumerate(aucs):
        result_file.loc[idx, 'dataset'] = idx + 1
        result_file.loc[idx, 'auc'] = auc
    result_file.to_csv(file_path, index = False)
    return
 
if __name__ == "__main__":
    remove_all_predict()