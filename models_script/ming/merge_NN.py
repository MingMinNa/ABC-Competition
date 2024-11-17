import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:    
    from .utils import file_handler, data_preprocess
    from .utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
    from . import simple_NN, XGBoost, catBoost, lightGBM
except: 
    from utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
    import simple_NN, XGBoost, catBoost, lightGBM


class MergeNN(nn.Module):
    def __init__(self, input_size):
        super(MergeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)            
        self.fc2 = nn.Linear(64, 1)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))         
        return x

def main(RANDOM_SEED = 42):

    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # preprocess. But no preprocessing has the better result (?)
    # preprocess_X_trains, preprocess_y_trains, preprocess_X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests)
    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests, 
                                                                  numeric_handler = Numeric_Handler_Type.No_preprocess,
                                                                  categoric_handler = Categoric_Handler_Type.No_preprocess)
    torch.manual_seed(RANDOM_SEED)

    y_predicts = []
    for i in tqdm(range(len(dataset_names))):

        # models with preprocessed data
        '''
        preprocess_simpleNN_model = simple_NN.build_model(X_train = preprocess_X_trains[i], y_train = preprocess_y_trains[i], RANDOM_SEED = RANDOM_SEED)
        preprocess_XGB_model = XGBoost.build_model(X_train = preprocess_X_trains[i], y_train = preprocess_y_trains[i], RANDOM_SEED = RANDOM_SEED)
        preprocess_LightGBM_model = lightGBM.build_model(X_train = preprocess_X_trains[i], y_train = preprocess_y_trains[i], RANDOM_SEED = RANDOM_SEED)
        preprocess_catBoost_model = catBoost.build_model(X_train = preprocess_X_trains[i], y_train = preprocess_y_trains[i], RANDOM_SEED = RANDOM_SEED)
        
        preprocess_tmp_X_train, preprocess_tmp_X_test, preprocess_tmp_y_train, preprocess_tmp_y_test = train_test_split(\
            preprocess_X_trains[i], preprocess_y_trains[i], train_size = 0.8, random_state = RANDOM_SEED)

        preprocess_simpleNN_model.eval()
        with torch.no_grad():
            preprocess_simpleNN_prob = preprocess_simpleNN_model(torch.FloatTensor(preprocess_tmp_X_train.values)).numpy()
        
        preprocess_XGB_prob = preprocess_XGB_model.predict_proba(preprocess_tmp_X_train)[:, 1]
        preprocess_LightGBM_prob = preprocess_LightGBM_model.predict(preprocess_tmp_X_train, num_iteration = preprocess_LightGBM_model.best_iteration)
        preprocess_catBoost_prob = preprocess_catBoost_model.predict_proba(preprocess_tmp_X_train)[:, 1]
        '''


        # models with raw data
        simpleNN_model = simple_NN.build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        XGB_model = XGBoost.build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        LightGBM_model = lightGBM.build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        catBoost_model = catBoost.build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)

        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(\
            X_trains[i], y_trains[i], train_size = 0.8, random_state = RANDOM_SEED)
        
        simpleNN_model.eval()
        with torch.no_grad():
            simpleNN_y_prob = simpleNN_model(torch.FloatTensor(tmp_X_train.values)).numpy()
        
        XGB_y_prob = XGB_model.predict_proba(tmp_X_train)[:, 1]
        LightGBM_y_prob = LightGBM_model.predict(tmp_X_train, num_iteration = LightGBM_model.best_iteration)
        catBoost_y_prob = catBoost_model.predict_proba(tmp_X_train)[:, 1]

        train_tensor = TensorDataset(torch.FloatTensor(pd.DataFrame({
            'NN': simpleNN_y_prob[:, 0], 
            'XGB': XGB_y_prob, 
            'LightGBM': LightGBM_y_prob,
            'catBoost': catBoost_y_prob,
            # 'preprocess_NN': preprocess_simpleNN_prob[:, 0],
            # 'preprocess_XGB':preprocess_XGB_prob,
            # 'preprocess_LightGBM':preprocess_LightGBM_prob,
            # 'preprocess_catBoost':preprocess_catBoost_prob
            }).values), torch.FloatTensor(tmp_y_train.values))
        train_loader = DataLoader(train_tensor, batch_size = 32, shuffle = True)

        merge_model = MergeNN(4)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(merge_model.parameters(), lr = 0.001)

        for epoch in range(50): 
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = merge_model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

        merge_model.eval()
        simpleNN_model.eval()
        with torch.no_grad():
            # preprocess_simpleNN_prob = preprocess_simpleNN_model(torch.FloatTensor(preprocess_tmp_X_test.values)).numpy()
            # preprocess_XGB_prob = preprocess_XGB_model.predict_proba(preprocess_tmp_X_test)[:, 1]
            # preprocess_LightGBM_prob = preprocess_LightGBM_model.predict(preprocess_tmp_X_test, num_iteration = preprocess_LightGBM_model.best_iteration)
            # preprocess_catBoost_prob = preprocess_catBoost_model.predict_proba(preprocess_tmp_X_test)[:, 1]

            simpleNN_y_prob = simpleNN_model(torch.FloatTensor(tmp_X_test.values)).numpy()
            XGB_y_prob = XGB_model.predict_proba(tmp_X_test)[:, 1]
            LightGBM_y_prob = LightGBM_model.predict(tmp_X_test, num_iteration = LightGBM_model.best_iteration)
            catBoost_y_prob = catBoost_model.predict_proba(tmp_X_test)[:, 1]

            tmp_y_prob = merge_model(torch.FloatTensor(pd.DataFrame({
                                    'NN': simpleNN_y_prob[:, 0], 
                                    'XGB': XGB_y_prob,
                                    'LightGBM': LightGBM_y_prob,
                                    'catBoost': catBoost_y_prob,
                                    # 'preprocess_NN': preprocess_simpleNN_prob[:, 0],
                                    # 'preprocess_XGB':preprocess_XGB_prob,
                                    # 'preprocess_LightGBM':preprocess_LightGBM_prob,
                                    # 'preprocess_catBoost':preprocess_catBoost_prob
                                    }).values)).numpy()
            auc = roc_auc_score(tmp_y_test, tmp_y_prob)
            print(f"{i} {auc}")

            # preprocess_simpleNN_prob = preprocess_simpleNN_model(torch.FloatTensor(preprocess_X_tests[i].values)).numpy()
            # preprocess_XGB_prob = preprocess_XGB_model.predict_proba(preprocess_X_tests[i])[:, 1]
            # preprocess_LightGBM_prob = preprocess_LightGBM_model.predict(preprocess_X_tests[i], num_iteration = preprocess_LightGBM_model.best_iteration)
            # preprocess_catBoost_prob = preprocess_catBoost_model.predict_proba(preprocess_X_tests[i])[:, 1]

            simpleNN_y_prob = simpleNN_model(torch.FloatTensor(X_tests[i].values)).numpy()
            XGB_y_prob = XGB_model.predict_proba(X_tests[i])[:, 1]
            LightGBM_y_prob = LightGBM_model.predict(X_tests[i], num_iteration = LightGBM_model.best_iteration)
            catBoost_y_prob = catBoost_model.predict_proba(X_tests[i])[:, 1]

            y_predict_proba = merge_model(torch.FloatTensor(pd.DataFrame({
                                         'DL': simpleNN_y_prob[:, 0], 
                                         'XGB': XGB_y_prob,
                                         'LightGBM': LightGBM_y_prob,
                                         'catBoost': catBoost_y_prob,
                                        #  'preprocess_NN': preprocess_simpleNN_prob[:, 0],
                                        #  'preprocess_XGB':preprocess_XGB_prob,
                                        #  'preprocess_LightGBM':preprocess_LightGBM_prob,
                                        #  'preprocess_catBoost':preprocess_catBoost_prob
                                         }).values)).numpy()
        
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

        # del preprocess_LightGBM_model, preprocess_XGB_model, preprocess_simpleNN_model, preprocess_catBoost_model
        del LightGBM_model, XGB_model, simpleNN_model, catBoost_model
        del merge_model

    file_handler.save_predict(y_predicts, dataset_names)


# (1) numeric(No_preprocess), categoric(No_preprocess):         0.875328
# (2) numeric(No_preprocess), categoric(Normalization):         0.874448
# (3) numeric(No_preprocess), categoric(One-Hot encoder):       0.871739
# (4) numeric(Std), categoric(No_preprocess):                   0.869394
# (5) numeric(Std), categoric(Normalization):                   0.868946
# (6) numeric(Std), categoric(One-Hot encoder):                 0.86681
# (7) numeric(Normalization), categoric(No_preprocess):         0.773912
# (8) numeric(Normalization), categoric(Normalization):         0.773445
# (9) numeric(Normalization), categoric(One-Hot encoder):       0.769017
if __name__ == '__main__':
    main(RANDOM_SEED = 200)
    quit()
