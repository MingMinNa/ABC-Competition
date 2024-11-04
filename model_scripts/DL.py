import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import matplotlib.pyplot as plt
try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)           
        self.fc3 = nn.Linear(32, 1)             

    def forward(self, x):
        x = F.relu(self.fc1(x))                
        x = self.dropout1(x)                    
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))         
        return x

def main(TEST_MODE = True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    # preprocess
    for i, _ in enumerate(dataset_names):
        X_train, X_test, Y_train = X_trains[i], X_tests[i], y_trains[i]

        numeric_features, categoric_features = data_preprocess.get_number_of_datatype(X_train)

        X_train = data_preprocess.preprocess_features(X_train, numeric_features, categoric_features)
        X_test = data_preprocess.preprocess_features( X_test, numeric_features, categoric_features)
        X_trains[i], X_tests[i] = data_preprocess.align_columns(X_train, X_test)
        y_trains[i] = Y_train

    torch.manual_seed(42)
    models = []
    aucs = []
    
    for i in tqdm(range(len(dataset_names))):
        X = X_trains[i]
        y = y_trains[i]
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        fold_aucs = []

        for train_index, test_index in skf.split(X, y):
            tmp_X_train, tmp_X_test = X.iloc[train_index], X.iloc[test_index]
            tmp_y_train, tmp_y_test = y.iloc[train_index], y.iloc[test_index]

            train_tensor = TensorDataset(torch.FloatTensor(tmp_X_train.values), torch.FloatTensor(tmp_y_train.values))
            train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
            
            input_size = tmp_X_train.shape[1]
            model = SimpleNN(input_size)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr = 0.001)
            
            # Train the model
            model.train()
            for epoch in range(70):  # You can adjust this based on your dataset
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    loss.backward()
                    optimizer.step()

            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                tmp_y_prob = model(torch.FloatTensor(tmp_X_test.values)).numpy()
                auc = roc_auc_score(tmp_y_test, tmp_y_prob)
                fold_aucs.append(auc)

        # Store the average AUC for the current dataset
        aucs.append(np.mean(fold_aucs))
        models.append(model)

    # Make predictions on the test set
    y_predicts = []
    for i in range(len(dataset_names)):
        model = models[i]
        model.eval()
        with torch.no_grad():
            y_predict_proba = model(torch.FloatTensor(X_tests[i].values)).numpy()
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    if TEST_MODE: file_handler.save_auc_result(aucs, "DL")
    file_handler.save_predict(y_predicts, dataset_names)
    return

if __name__ == "__main__":
    main(TEST_MODE = False)
    # test_single_dataset(2)