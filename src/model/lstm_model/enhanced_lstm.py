import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import ipywidgets as widgets
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display
from data_loader import time_series_loader,test_time_series_loader
import joblib  # 或使用 pickle
# 增强的LSTM模型
class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_dim]
        
        return self.fc(context)
def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val)
            total_loss += loss.item() * X_val.size(0)
    return total_loss / len(val_loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100, patience = 5):
    #early stopping
    best_val_loss = float('inf')
    epoch_without_improvement = 0
    best_model_weights = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_train.size(0)
        val_loss = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss/len(train_loader.dataset):.4f}  Test Loss: {val_loss:.4f}")

        #early_stopping
        if val_loss <best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
        else:
            epoch_without_improvement +=1

        if epoch_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_weights)
            break
    return model

def lstm_model(dp, selected_features, target_col, feature_name,test_size=0.2, random_state=42,  num_epochs=100, patience=5):
    
    if dp.df[selected_features + target_col ].isnull().sum().sum() > 0:
        df_dropna = dp.df[selected_features + target_col ].dropna()
    else:
        df_dropna = dp.df[selected_features + target_col ]

    X_raw = df_dropna[selected_features].values
    y_raw = df_dropna[target_col].values

    split_idx = int(len(X_raw) * (1 - test_size))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=random_state, shuffle=False)

    train_data_export = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=selected_features + [target_col])
    val_data_export = pd.DataFrame(np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1), columns=selected_features + [target_col])
    test_data_export = pd.DataFrame(np.concatenate([X_test_raw, y_test_raw.reshape(-1, 1)], axis=1), columns=selected_features + [target_col])
    train_data_export.to_csv(f'../data/model_data/train_data_export_{feature_name}.csv', index=False)
    val_data_export.to_csv(f'../data/model_data/val_data_export_{feature_name}.csv', index=False)
    test_data_export.to_csv(f'../data/model_data/test_data_export_{feature_name}.csv', index=False)

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    scaler_x_path = f'../data/model_data/scaler_x_{feature_name}.pkl'
    scaler_x_export = StandardScaler().fit(X_train_raw)
    joblib.dump(scaler_x_export, scaler_x_path)  # 或者用 pickle.dump()
    print(f"Scaler已保存到: {scaler_x_path}")
    X_test = scaler_x.transform(X_test_raw)
    X_val = scaler_x.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)
    y_val = scaler_y.transform(y_val)
    scaler_y_path = f'../data/model_data/scaler_y_{feature_name}.pkl'
    scaler_y_export = StandardScaler().fit(y_train_raw)
    joblib.dump(scaler_y_export, scaler_y_path)  # 或者用 pickle.dump()
    print(f"Scaler已保存到: {scaler_y_path}")

    
    train_loader = time_series_loader(X_train, y_train, seq_len=12, batch_size=64, shuffle=False)
    val_loader = time_series_loader(X_val, y_val, seq_len=12, batch_size=64, shuffle=False)
    test_loader = time_series_loader(X_test, y_test, seq_len=12, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedLSTM(input_dim=X_train.shape[1], hidden_dim=128, output_dim=1, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss()  # 对异常值更鲁棒
    train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs, patience=patience)
    model_weights_path = f"../model/model_weights_{feature_name}.pth"
    torch.save(model.state_dict(), model_weights_path)
    model_analysis(test_loader, model_weights_path, scaler_x, scaler_y, target_col, device, feature_name)

def model_analysis(test_loader, model_weights_path, scaler_x, scaler_y, target_col, device, feature_name):
    model = EnhancedLSTM(input_dim=test_loader.dataset.X.shape[1], hidden_dim=128, output_dim=1, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # 收集预测和真实值
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_preds.append(preds)
            y_trues.append(y_batch.numpy())

    # 合并结果并确保长度一致
    y_pred = np.concatenate(y_preds, axis=0)
    y_true = np.concatenate(y_trues, axis=0)
    
    # 检查长度是否一致
    assert len(y_pred) == len(y_true), f"预测值({len(y_pred)})和真实值({len(y_true)})长度不一致"

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"R2 Score: {r2:.4f}, MAE: {mae:.4f}")

    # 绘制对比曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制前200个样本更清晰（可根据需要调整）
    plot_samples = min(200, len(y_true))
    plt.plot(y_true[:plot_samples], 'b-', label='True Values', alpha=0.7, linewidth=1.5)
    plt.plot(y_pred[:plot_samples], 'r--', label='Predictions', alpha=0.8, linewidth=1.2)
    
    # 添加标注
    plt.title(f'True vs Predicted Values\nR2: {r2:.4f} | MAE: {mae:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 突出显示异常区域
    diff = np.abs(y_true[:plot_samples] - y_pred[:plot_samples])
    threshold = np.mean(diff) + 2 * np.std(diff)
    anomalies = np.where(diff > threshold)[0]
    plt.scatter(anomalies, y_true[anomalies], c='yellow', s=50, 
                edgecolors='red', label='Large Errors', zorder=3)
    
    plt.tight_layout()
    plt.savefig(f'../img/true_vs_predicted_plant1_{feature_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


