import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:    from .utils import file_handler, data_preprocess
except: from utils import file_handler, data_preprocess

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.fc2 = nn.Linear(64, 32)           
        self.fc3 = nn.Linear(32, 1)           

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函數
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 輸出層使用sigmoid激活函數
        return x

def main(TEST_MODE = True):

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
        if TEST_MODE:
            tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = train_test_split(
                X_trains[i], y_trains[i], test_size = 0.2, random_state = 42)
        else:
            tmp_X_train, tmp_y_train = X_trains[i], y_trains[i]
            
        train_tensor = TensorDataset(torch.FloatTensor(tmp_X_train.values), torch.FloatTensor(tmp_y_train.values))
        train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)
        
        input_size = tmp_X_train.shape[1] 
        model = SimpleNN(input_size)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
        model.train()
        for epoch in range(100):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()

        if TEST_MODE:
            model.eval()
            with torch.no_grad():
                tmp_y_prob = model(torch.FloatTensor(tmp_X_test.values)).numpy()
            auc = roc_auc_score(tmp_y_test, tmp_y_prob)
            aucs.append(auc)
            
        models.append(model)

    y_predicts = []
    for i in range(len(dataset_names)):
        model = models[i]
        model.eval()
        with torch.no_grad():
            y_predict_proba = model(torch.FloatTensor(X_tests[i].values)).numpy()
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

    file_handler.save_auc_result(aucs, "DL")
    file_handler.save_predict(y_predicts, dataset_names)

if __name__ == "__main__":
    main(TEST_MODE = True)