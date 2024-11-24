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

try:    
    from .utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type
except: 
    from utils import file_handler, data_preprocess
    from utils.data_preprocess import Numeric_Handler_Type, Categoric_Handler_Type

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

def build_model(X_train, y_train, RANDOM_SEED = 42):

    torch.manual_seed(RANDOM_SEED)
    
    X = X_train
    y = y_train

    input_size = X.shape[1]
    model = SimpleNN(input_size)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    train_tensor = TensorDataset(torch.FloatTensor(X.values), torch.FloatTensor(y.values))
    train_loader = DataLoader(train_tensor, batch_size = 32, shuffle = True)
    model.train()
    for epoch in range(50): 
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

    return model

# numeric(No_preprocess), categoric(No_preprocess):         0.843949 [Competition_data(simple).zip]
# numeric(std), categoric(one-hot):                         0.843499 [Competition_data(simple_2).zip]
def main(RANDOM_SEED = 42):

    # get dataset
    dataset_names, X_trains, y_trains, X_tests = file_handler.load_dataset()

    X_trains, y_trains, X_tests = data_preprocess.preprocess_data(dataset_names, X_trains, y_trains, X_tests,
                                                                  numeric_handler = Numeric_Handler_Type.No_preprocess,
                                                                  categoric_handler = Categoric_Handler_Type.No_preprocess)

    torch.manual_seed(RANDOM_SEED)
    y_predicts = []
    
    for i in tqdm(range(len(dataset_names))):
        model = build_model(X_train = X_trains[i], y_train = y_trains[i], RANDOM_SEED = RANDOM_SEED)
        model.eval()
        with torch.no_grad():
            y_predict_proba = model(torch.FloatTensor(X_tests[i].values)).numpy()
        df = pd.DataFrame(y_predict_proba, columns=["y_predict_proba"])
        y_predicts.append(df)

        del model

    file_handler.save_predict(y_predicts, dataset_names)
    return

if __name__ == "__main__":
    main(RANDOM_SEED = 42)