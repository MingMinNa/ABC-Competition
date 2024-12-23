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
    for idx in range(len(y_predicts)):
        predict_path = os.path.join(const.DATA_FOLDER, dataset_names[idx], "y_predict.csv")
        print(predict_path)
        df = y_predicts[idx]
        df.to_csv(predict_path, index = False,header = True)
    return
 
if __name__ == "__main__":
    remove_all_predict()