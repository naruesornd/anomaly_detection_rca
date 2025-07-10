# ===================== 1. 定义模型 (适配LSTM) ========== #
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from data_loader import time_series_loader

class LSTM_Dropout_Net(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2, num_layers=1):
        super(LSTM_Dropout_Net, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        return self.fc1(lstm_out)

# ===================== 2. DFS 工具函数 (适配LSTM) ===================== #
def fine_feature_selection(dp, top_k_features, target_columns, test_size=0.2, random_state=42, dropout_threshold=0.05):
    if dp.df[top_k_features + target_columns].isnull().sum().sum() > 0:
        df_dropna = dp.df[top_k_features + target_columns].dropna()
    else:
        df_dropna = dp.df[top_k_features + target_columns]

    feature_num = len(top_k_features)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X = scaler_x.fit_transform(df_dropna[top_k_features])
    y = scaler_y.fit_transform(df_dropna[target_columns])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_loader = time_series_loader(X_train, y_train, seq_len=12, batch_size=64, shuffle=True)
    test_loader = time_series_loader(X_test, y_test, seq_len=12, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Dropout_Net(input_size=feature_num, hidden_size=64, dropout=0.2, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model = model_training(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)

    # ========== Drop Feature Selection: 逐个特征置零，比较验证集损失变化 ========== #
    print("\n\n[Drop Feature Selection]\n")
    baseline_loss = evaluate_model(model, test_loader, criterion, device)
    drop_results = {}

    for idx, col in enumerate(top_k_features):
        X_test_dropped = X_test.copy()
        X_test_dropped[:, idx] = 0
        dropped_loader = time_series_loader(X_test_dropped, y_test, seq_len=12, batch_size=64, shuffle=False)
        drop_loss = evaluate_model(model, dropped_loader, criterion, device)
        delta = drop_loss - baseline_loss
        drop_results[col] = delta
        print(f"Drop {col}: loss delta = {delta:.6f}")

    # 过滤贡献小于阈值的特征
    selected_features = [f for f, d in drop_results.items() if d > dropout_threshold]
    print(f"\n最终选定特征数: {len(selected_features)}")
    return selected_features

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item()
    return total_loss / len(dataloader)

def model_training(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss/len(train_dataloader):.4f}  Test Loss: {val_loss:.4f}")
    return model

def dataset_to_tensor(dataset):
    samples = []
    for X, _ in dataset:
        samples.append(X)
    return torch.stack(samples)
