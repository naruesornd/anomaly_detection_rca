import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 1. Dropout 模型定义 ===================== #
class DropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
# ===================== 2. DFS 工具函数 ===================== #
"""
重要性分析
"""
def dropout_feature_importance(model, X_tensor, n_samples=100):
    model.train()
    all_outputs = []
    for _ in range(n_samples):
        output = model(X_tensor).detach().cpu().numpy()
        all_outputs.append(output)
    all_outputs = np.stack(all_outputs, axis=0)
    return np.var(all_outputs, axis=0).mean()
"""
重要性评估,每次随机mask掉一个特征,计算模型输出的方差
"""
def evaluate_feature_importance(model, X_tensor, n_samples=100):
    n_features = X_tensor.shape[1]
    base_score = dropout_feature_importance(model, X_tensor, n_samples)
    importances = []
    for i in range(n_features):
        X_masked = X_tensor.clone()
        X_masked[:, i] = 0
        masked_score = dropout_feature_importance(model, X_masked, n_samples)
        importances.append(base_score - masked_score)
    return np.array(importances)